//===- bolt/Passes/RelocationRecovery.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/RelocationRecovery.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/DataExtractor.h"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltCategory;
extern cl::opt<bool> Instrument;
cl::opt<bool> AggressiveRelocationRecovery(
    "aggressive-relocation-recovery",
    cl::desc("also recover function pointers from general data sections; "
             "requires --recover-relocations"),
    cl::Hidden, cl::cat(BoltCategory));
} // namespace opts

namespace llvm {
namespace bolt {
namespace {

/// Information collected for one ADRP instruction before rewriting it.
/// Page is the absolute page computed by the original ADRP. Adds contains the
/// ADD instructions that use this ADRP result to form the same target address;
/// Symbol names that target when it can be recovered unambiguously.
struct AdrpDefinition {
  MCInst *ADRP;
  uint64_t Page;
  MCSymbol *Symbol = nullptr;
  SmallVector<MCInst *, 4> Adds;
  bool ClearOffsetAfterRecovery;
};

Error recoveryError(const BinaryFunction &BF, StringRef Reason) {
  return createFatalBOLTError(Twine("cannot recover references in ") +
                              BF.getPrintName() + ": " + Reason);
}

/// Follow an ADRP definition through the CFG and collect ADD uses that compute
/// one target. Record all definitions and uses before rewriting operands
/// because multiple definitions can reach the same ADD. When the target is not
/// unique, preserve the input page in the ADRP and leave the ADD unchanged.
Error collectAdrpDefUseChains(
    BinaryFunction &BF, SmallVectorImpl<AdrpDefinition> &Definitions,
    DenseMap<MCInst *, BinaryBasicBlock *> &AddBlocks) {
  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock &BB : BF) {
    for (auto I = BB.begin(); I != BB.end(); ++I) {
      MCInst &ADRP = *I;
      if (!BC.MIB->isADRP(ADRP) || !ADRP.getOperand(1).isImm())
        continue;
      const std::optional<uint32_t> Offset = BC.MIB->getOffset(ADRP);
      if (!Offset)
        return recoveryError(BF, "missing ADRP input offset");
      const uint64_t Page = ((BF.getAddress() + *Offset) & ~uint64_t(4095)) +
                            uint64_t(ADRP.getOperand(1).getImm()) * 4096;
      AdrpDefinition Definition{&ADRP,
                                Page,
                                nullptr,
                                {},
                                !BF.requiresPreciseAddressMap() &&
                                    !opts::Instrument};
      const unsigned Reg = ADRP.getOperand(0).getReg();
      SmallVector<BinaryBasicBlock *, 8> Worklist;
      SmallPtrSet<BinaryBasicBlock *, 8> Visited;
      Worklist.push_back(&BB);
      bool First = true;
      bool Complete = true;
      std::optional<uint64_t> Target;
      while (!Worklist.empty()) {
        BinaryBasicBlock *Current = Worklist.pop_back_val();
        auto Begin = First ? std::next(I) : Current->begin();
        const bool IsInitial = First;
        First = false;
        if (!IsInitial && !Visited.insert(Current).second)
          continue;
        bool Killed = false;
        for (auto II = Begin; II != Current->end(); ++II) {
          MCInst &Inst = *II;
          if (BC.MIB->isPseudo(Inst))
            continue;
          if (BC.MIB->isCall(Inst)) {
            Complete = false;
            Killed = true;
            break;
          }
          const bool Uses = BC.MIB->hasUseOfPhysReg(Inst, Reg);
          const bool Defines = BC.MIB->hasDefOfPhysReg(Inst, Reg);
          if (Uses) {
            if (!BC.MIB->isAddXri(Inst) || Inst.getOperand(1).getReg() != Reg ||
                !Inst.getOperand(2).isImm() || !Inst.getOperand(3).isImm() ||
                Inst.getOperand(3).getImm()) {
              Complete = false;
            } else {
              const uint64_t Address = Page + Inst.getOperand(2).getImm();
              if (Target && *Target != Address)
                Complete = false;
              Target = Address;
              Definition.Adds.push_back(&Inst);
              AddBlocks.try_emplace(&Inst, Current);
            }
          }
          if (Defines) {
            Killed = true;
            break;
          }
        }
        if (!Killed)
          for (BinaryBasicBlock *Succ : Current->successors())
            Worklist.push_back(Succ);
      }
      if (Target) {
        if (BinaryFunction *Dest = BC.getBinaryFunctionAtAddress(*Target)) {
          Definition.Symbol = Dest->getSymbol();
        } else if (BinaryFunction *Dest =
                       BC.getBinaryFunctionContainingAddress(*Target)) {
          if (!Dest->isInConstantIsland(*Target)) {
            if (const BinaryBasicBlock *Entry =
                    Dest->getBasicBlockAtOffset(*Target - Dest->getAddress()))
              Definition.Symbol = Dest->getSecondaryEntryPointSymbol(*Entry);
          }
        } else if (BC.getJumpTableContainingAddress(*Target)) {
          Definition.Symbol =
              BC.getOrCreateGlobalSymbol(*Target, "JUMP_TABLE/");
        }
      }
      if (!Complete)
        Definition.Symbol = nullptr;
      Definitions.push_back(std::move(Definition));
    }
  }
  return Error::success();
}

/// Walk backward from Use through its predecessor blocks and find the
/// definition of Reg on every path. Succeed only when each path reaches an ADRP
/// before a call, a non-ADRP definition of Reg, or a function-entry boundary.
/// Scan UseBlock only before Use; if a loop reaches it again, scan the complete
/// block. Each complete predecessor block is visited at most once.
bool collectReachingAdrpDefinitions(
    BinaryContext &BC, BinaryBasicBlock &UseBlock, MCInst &Use, unsigned Reg,
    SmallPtrSetImpl<MCInst *> &ReachingDefinitions) {
  struct BlockPosition {
    BinaryBasicBlock *Block;
    MCInst *Stop;
  };
  SmallVector<BlockPosition, 8> Worklist{{&UseBlock, &Use}};
  SmallPtrSet<BinaryBasicBlock *, 8> VisitedFullBlocks;

  while (!Worklist.empty()) {
    const BlockPosition Position = Worklist.pop_back_val();
    BinaryBasicBlock *Block = Position.Block;
    auto End = Block->end();
    if (Position.Stop) {
      End = llvm::find_if(*Block,
                          [&](MCInst &Inst) { return &Inst == Position.Stop; });
      if (End == Block->end())
        return false;
    } else if (!VisitedFullBlocks.insert(Block).second) {
      continue;
    }

    bool FoundDefinition = false;
    for (auto I = std::make_reverse_iterator(End), E = Block->rend(); I != E;
         ++I) {
      MCInst &Inst = *I;
      if (BC.MIB->isPseudo(Inst))
        continue;
      if (BC.MIB->isCall(Inst))
        return false;
      if (!BC.MIB->hasDefOfPhysReg(Inst, Reg))
        continue;
      if (!BC.MIB->isADRP(Inst) || Inst.getOperand(0).getReg() != Reg ||
          !Inst.getOperand(1).isImm())
        return false;
      ReachingDefinitions.insert(&Inst);
      FoundDefinition = true;
      break;
    }
    if (FoundDefinition)
      continue;
    if (Block->isEntryPoint() || Block->isLandingPad() || Block->pred_empty())
      return false;
    for (BinaryBasicBlock *Pred : Block->predecessors())
      Worklist.push_back({Pred, nullptr});
  }
  return !ReachingDefinitions.empty();
}

Error recoverAArch64InstructionReferences(BinaryContext &BC) {
  SmallVector<AdrpDefinition, 16> Definitions;
  DenseMap<MCInst *, BinaryBasicBlock *> AddBlocks;
  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BC.shouldEmit(BF))
      continue;
    if (Error E = collectAdrpDefUseChains(BF, Definitions, AddBlocks))
      return E;
  }

  DenseMap<MCInst *, AdrpDefinition *> DefinitionsByInstruction;
  SmallPtrSet<MCInst *, 16> CandidateAdds;
  for (AdrpDefinition &Definition : Definitions) {
    DefinitionsByInstruction.try_emplace(Definition.ADRP, &Definition);
    CandidateAdds.insert_range(Definition.Adds);
  }

  DenseMap<MCInst *, MCSymbol *> CandidateAddTargets;
  SmallPtrSet<MCInst *, 16> UnsafeAdds;
  for (MCInst *Add : CandidateAdds) {
    auto BlockIt = AddBlocks.find(Add);
    if (BlockIt == AddBlocks.end()) {
      UnsafeAdds.insert(Add);
      continue;
    }

    const unsigned Reg = Add->getOperand(1).getReg();
    SmallPtrSet<MCInst *, 4> ReachingDefinitions;
    if (!collectReachingAdrpDefinitions(BC, *BlockIt->second, *Add, Reg,
                                        ReachingDefinitions)) {
      UnsafeAdds.insert(Add);
      continue;
    }

    std::optional<uint64_t> Page;
    MCSymbol *Symbol = nullptr;
    bool Valid = true;
    for (MCInst *ADRP : ReachingDefinitions) {
      auto DefinitionIt = DefinitionsByInstruction.find(ADRP);
      if (DefinitionIt == DefinitionsByInstruction.end() ||
          !DefinitionIt->second->Symbol ||
          !llvm::is_contained(DefinitionIt->second->Adds, Add)) {
        Valid = false;
        break;
      }
      const AdrpDefinition &Definition = *DefinitionIt->second;
      if ((Page && *Page != Definition.Page) ||
          (Symbol && Symbol != Definition.Symbol)) {
        Valid = false;
        break;
      }
      Page = Definition.Page;
      Symbol = Definition.Symbol;
    }
    if (!Valid || !Symbol)
      UnsafeAdds.insert(Add);
    else
      CandidateAddTargets.try_emplace(Add, Symbol);
  }

  // An ADRP may feed several ADDs, and an ADD may be reached by several
  // ADRPs. Their rewrites therefore cannot be decided independently. If one
  // ADD is unsafe, make every ADRP that can reach it fall back to its original
  // page and mark every other ADD fed by those ADRPs as unsafe. Repeat until
  // no additional ADRP or ADD becomes unsafe, then rewrite the remaining pairs.
  bool Changed;
  do {
    Changed = false;
    for (AdrpDefinition &Definition : Definitions) {
      if (!Definition.Symbol ||
          !llvm::any_of(Definition.Adds,
                        [&](MCInst *Add) { return UnsafeAdds.contains(Add); }))
        continue;
      Definition.Symbol = nullptr;
      for (MCInst *Add : Definition.Adds)
        Changed |= UnsafeAdds.insert(Add).second;
    }
  } while (Changed);

  DenseMap<MCInst *, MCSymbol *> AddTargets;
  for (const auto &[Add, Symbol] : CandidateAddTargets)
    if (!UnsafeAdds.contains(Add))
      AddTargets.try_emplace(Add, Symbol);

  for (AdrpDefinition &Definition : Definitions) {
    // Without a recovered pair target, encode the absolute input page rather
    // than retaining a PC-relative immediate whose meaning changes when the
    // instruction moves.
    MCSymbol *Symbol = Definition.Symbol;
    if (!Symbol)
      Symbol = BC.registerNameAtAddress("__BOLT_zero_addr", 0, 0, 0);
    int64_t Value;
    BC.MIB->replaceImmWithSymbolRef(
        *Definition.ADRP, Symbol, Definition.Symbol ? 0 : Definition.Page,
        BC.Ctx.get(), Value, ELF::R_AARCH64_ADR_PREL_PG_HI21);
    if (Definition.ClearOffsetAfterRecovery)
      BC.MIB->clearOffset(*Definition.ADRP);
  }
  for (const auto &[Add, Symbol] : AddTargets) {
    if (!Symbol)
      continue;
    int64_t Value;
    BC.MIB->replaceImmWithSymbolRef(*Add, Symbol, 0, BC.Ctx.get(), Value,
                                    ELF::R_AARCH64_ADD_ABS_LO12_NC);
  }
  return Error::success();
}

/// Reconstruct absolute relocations for aligned words that exactly equal a
/// movable function entry. Conservative mode scans ELF pointer arrays;
/// aggressive mode also scans selected general data sections.
Error reconstructDataRelocations(BinaryContext &BC) {
  struct PointerReference {
    uint64_t Address;
    uint64_t Target;
    MCSymbol *Symbol;
  };
  SmallVector<PointerReference, 16> References;
  for (BinarySection &Section : BC.sections()) {
    const StringRef Name = Section.getName();
    const bool IsArray = Name == ".init_array" || Name == ".fini_array";
    const bool IsAggressiveSection = Name == ".data.rel.ro" ||
                                     Name == ".data" || Name == ".rodata" ||
                                     Name == ".tdata";
    if (!IsArray &&
        (!opts::AggressiveRelocationRecovery || !IsAggressiveSection))
      continue;
    DataExtractor Data(Section.getContents(), BC.AsmInfo->isLittleEndian());
    uint64_t Offset = (8 - Section.getAddress() % 8) % 8;
    while (Data.isValidOffsetForDataOfSize(Offset, 8)) {
      const uint64_t Address = Section.getAddress() + Offset;
      const uint64_t Target = Data.getU64(&Offset);
      if (BC.getDynamicRelocationAt(Address) ||
          Section.getRelocationAt(Address - Section.getAddress()))
        continue;
      BinaryFunction *BF = BC.getBinaryFunctionAtAddress(Target);
      if (!BF || BF->isPLTFunction() || !BC.shouldEmit(*BF))
        continue;
      References.push_back({Address, Target, BF->getSymbol()});
    }
  }
  const uint32_t Type = BC.isAArch64()
                            ? static_cast<uint32_t>(ELF::R_AARCH64_ABS64)
                            : static_cast<uint32_t>(ELF::R_X86_64_64);
  for (const PointerReference &Ref : References)
    BC.addRelocation(Ref.Address, Ref.Symbol, Type, 0, Ref.Target);
  return Error::success();
}

} // namespace

Error RelocationRecovery::runOnFunctions(BinaryContext &BC) {
  assert(BC.RecoverRelocations && "recovery pass requires explicit opt-in");
  if (BC.isAArch64())
    if (Error E = recoverAArch64InstructionReferences(BC))
      return E;

  return reconstructDataRelocations(BC);
}

} // namespace bolt
} // namespace llvm
