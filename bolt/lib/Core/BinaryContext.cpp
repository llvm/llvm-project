//===- bolt/Core/BinaryContext.cpp - Low-level context --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BinaryContext class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/NameResolver.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <unordered_set>

using namespace llvm;

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

namespace opts {

cl::opt<bool> NoHugePages("no-huge-pages",
                          cl::desc("use regular size pages for code alignment"),
                          cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool>
PrintDebugInfo("print-debug-info",
  cl::desc("print debug info when printing functions"),
  cl::Hidden,
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

cl::opt<bool> PrintRelocations(
    "print-relocations",
    cl::desc("print relocations when printing functions/objects"), cl::Hidden,
    cl::cat(BoltCategory));

static cl::opt<bool>
PrintMemData("print-mem-data",
  cl::desc("print memory data annotations when printing functions"),
  cl::Hidden,
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

BinaryContext::BinaryContext(std::unique_ptr<MCContext> Ctx,
                             std::unique_ptr<DWARFContext> DwCtx,
                             std::unique_ptr<Triple> TheTriple,
                             const Target *TheTarget, std::string TripleName,
                             std::unique_ptr<MCCodeEmitter> MCE,
                             std::unique_ptr<MCObjectFileInfo> MOFI,
                             std::unique_ptr<const MCAsmInfo> AsmInfo,
                             std::unique_ptr<const MCInstrInfo> MII,
                             std::unique_ptr<const MCSubtargetInfo> STI,
                             std::unique_ptr<MCInstPrinter> InstPrinter,
                             std::unique_ptr<const MCInstrAnalysis> MIA,
                             std::unique_ptr<MCPlusBuilder> MIB,
                             std::unique_ptr<const MCRegisterInfo> MRI,
                             std::unique_ptr<MCDisassembler> DisAsm)
    : Ctx(std::move(Ctx)), DwCtx(std::move(DwCtx)),
      TheTriple(std::move(TheTriple)), TheTarget(TheTarget),
      TripleName(TripleName), MCE(std::move(MCE)), MOFI(std::move(MOFI)),
      AsmInfo(std::move(AsmInfo)), MII(std::move(MII)), STI(std::move(STI)),
      InstPrinter(std::move(InstPrinter)), MIA(std::move(MIA)),
      MIB(std::move(MIB)), MRI(std::move(MRI)), DisAsm(std::move(DisAsm)) {
  Relocation::Arch = this->TheTriple->getArch();
  RegularPageSize = isAArch64() ? RegularPageSizeAArch64 : RegularPageSizeX86;
  PageAlign = opts::NoHugePages ? RegularPageSize : HugePageSize;
}

BinaryContext::~BinaryContext() {
  for (BinarySection *Section : Sections)
    delete Section;
  for (BinaryFunction *InjectedFunction : InjectedBinaryFunctions)
    delete InjectedFunction;
  for (std::pair<const uint64_t, JumpTable *> JTI : JumpTables)
    delete JTI.second;
  clearBinaryData();
}

/// Create BinaryContext for a given architecture \p ArchName and
/// triple \p TripleName.
Expected<std::unique_ptr<BinaryContext>>
BinaryContext::createBinaryContext(const ObjectFile *File, bool IsPIC,
                                   std::unique_ptr<DWARFContext> DwCtx) {
  StringRef ArchName = "";
  StringRef FeaturesStr = "";
  switch (File->getArch()) {
  case llvm::Triple::x86_64:
    ArchName = "x86-64";
    FeaturesStr = "+nopl";
    break;
  case llvm::Triple::aarch64:
    ArchName = "aarch64";
    FeaturesStr = "+all";
    break;
  case llvm::Triple::riscv64:
    ArchName = "riscv64";
    // RV64GC
    FeaturesStr = "+m,+a,+f,+d,+zicsr,+zifencei,+c,+relax";
    break;
  default:
    return createStringError(std::errc::not_supported,
                             "BOLT-ERROR: Unrecognized machine in ELF file");
  }

  auto TheTriple = std::make_unique<Triple>(File->makeTriple());
  const std::string TripleName = TheTriple->str();

  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(std::string(ArchName), *TheTriple, Error);
  if (!TheTarget)
    return createStringError(make_error_code(std::errc::not_supported),
                             Twine("BOLT-ERROR: ", Error));

  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return createStringError(
        make_error_code(std::errc::not_supported),
        Twine("BOLT-ERROR: no register info for target ", TripleName));

  // Set up disassembler.
  std::unique_ptr<MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCTargetOptions()));
  if (!AsmInfo)
    return createStringError(
        make_error_code(std::errc::not_supported),
        Twine("BOLT-ERROR: no assembly info for target ", TripleName));
  // BOLT creates "func@PLT" symbols for PLT entries. In function assembly dump
  // we want to emit such names as using @PLT without double quotes to convey
  // variant kind to the assembler. BOLT doesn't rely on the linker so we can
  // override the default AsmInfo behavior to emit names the way we want.
  AsmInfo->setAllowAtInName(true);

  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, "", FeaturesStr));
  if (!STI)
    return createStringError(
        make_error_code(std::errc::not_supported),
        Twine("BOLT-ERROR: no subtarget info for target ", TripleName));

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    return createStringError(
        make_error_code(std::errc::not_supported),
        Twine("BOLT-ERROR: no instruction info for target ", TripleName));

  std::unique_ptr<MCContext> Ctx(
      new MCContext(*TheTriple, AsmInfo.get(), MRI.get(), STI.get()));
  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(*Ctx, IsPIC));
  Ctx->setObjectFileInfo(MOFI.get());
  // We do not support X86 Large code model. Change this in the future.
  bool Large = false;
  if (TheTriple->getArch() == llvm::Triple::aarch64)
    Large = true;
  unsigned LSDAEncoding =
      Large ? dwarf::DW_EH_PE_absptr : dwarf::DW_EH_PE_udata4;
  if (IsPIC) {
    LSDAEncoding = dwarf::DW_EH_PE_pcrel |
                   (Large ? dwarf::DW_EH_PE_sdata8 : dwarf::DW_EH_PE_sdata4);
  }

  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, *Ctx));

  if (!DisAsm)
    return createStringError(
        make_error_code(std::errc::not_supported),
        Twine("BOLT-ERROR: no disassembler info for target ", TripleName));

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));
  if (!MIA)
    return createStringError(
        make_error_code(std::errc::not_supported),
        Twine("BOLT-ERROR: failed to create instruction analysis for target ",
              TripleName));

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> InstructionPrinter(
      TheTarget->createMCInstPrinter(*TheTriple, AsmPrinterVariant, *AsmInfo,
                                     *MII, *MRI));
  if (!InstructionPrinter)
    return createStringError(
        make_error_code(std::errc::not_supported),
        Twine("BOLT-ERROR: no instruction printer for target ", TripleName));
  InstructionPrinter->setPrintImmHex(true);

  std::unique_ptr<MCCodeEmitter> MCE(
      TheTarget->createMCCodeEmitter(*MII, *Ctx));

  // Make sure we don't miss any output on core dumps.
  outs().SetUnbuffered();
  errs().SetUnbuffered();
  dbgs().SetUnbuffered();

  auto BC = std::make_unique<BinaryContext>(
      std::move(Ctx), std::move(DwCtx), std::move(TheTriple), TheTarget,
      std::string(TripleName), std::move(MCE), std::move(MOFI),
      std::move(AsmInfo), std::move(MII), std::move(STI),
      std::move(InstructionPrinter), std::move(MIA), nullptr, std::move(MRI),
      std::move(DisAsm));

  BC->LSDAEncoding = LSDAEncoding;

  BC->MAB = std::unique_ptr<MCAsmBackend>(
      BC->TheTarget->createMCAsmBackend(*BC->STI, *BC->MRI, MCTargetOptions()));

  BC->setFilename(File->getFileName());

  BC->HasFixedLoadAddress = !IsPIC;

  BC->SymbolicDisAsm = std::unique_ptr<MCDisassembler>(
      BC->TheTarget->createMCDisassembler(*BC->STI, *BC->Ctx));

  if (!BC->SymbolicDisAsm)
    return createStringError(
        make_error_code(std::errc::not_supported),
        Twine("BOLT-ERROR: no disassembler info for target ", TripleName));

  return std::move(BC);
}

bool BinaryContext::forceSymbolRelocations(StringRef SymbolName) const {
  if (opts::HotText &&
      (SymbolName == "__hot_start" || SymbolName == "__hot_end"))
    return true;

  if (opts::HotData &&
      (SymbolName == "__hot_data_start" || SymbolName == "__hot_data_end"))
    return true;

  if (SymbolName == "_end")
    return true;

  return false;
}

std::unique_ptr<MCObjectWriter>
BinaryContext::createObjectWriter(raw_pwrite_stream &OS) {
  return MAB->createObjectWriter(OS);
}

bool BinaryContext::validateObjectNesting() const {
  auto Itr = BinaryDataMap.begin();
  auto End = BinaryDataMap.end();
  bool Valid = true;
  while (Itr != End) {
    auto Next = std::next(Itr);
    while (Next != End &&
           Itr->second->getSection() == Next->second->getSection() &&
           Itr->second->containsRange(Next->second->getAddress(),
                                      Next->second->getSize())) {
      if (Next->second->Parent != Itr->second) {
        errs() << "BOLT-WARNING: object nesting incorrect for:\n"
               << "BOLT-WARNING:  " << *Itr->second << "\n"
               << "BOLT-WARNING:  " << *Next->second << "\n";
        Valid = false;
      }
      ++Next;
    }
    Itr = Next;
  }
  return Valid;
}

bool BinaryContext::validateHoles() const {
  bool Valid = true;
  for (BinarySection &Section : sections()) {
    for (const Relocation &Rel : Section.relocations()) {
      uint64_t RelAddr = Rel.Offset + Section.getAddress();
      const BinaryData *BD = getBinaryDataContainingAddress(RelAddr);
      if (!BD) {
        errs() << "BOLT-WARNING: no BinaryData found for relocation at address"
               << " 0x" << Twine::utohexstr(RelAddr) << " in "
               << Section.getName() << "\n";
        Valid = false;
      } else if (!BD->getAtomicRoot()) {
        errs() << "BOLT-WARNING: no atomic BinaryData found for relocation at "
               << "address 0x" << Twine::utohexstr(RelAddr) << " in "
               << Section.getName() << "\n";
        Valid = false;
      }
    }
  }
  return Valid;
}

void BinaryContext::updateObjectNesting(BinaryDataMapType::iterator GAI) {
  const uint64_t Address = GAI->second->getAddress();
  const uint64_t Size = GAI->second->getSize();

  auto fixParents = [&](BinaryDataMapType::iterator Itr,
                        BinaryData *NewParent) {
    BinaryData *OldParent = Itr->second->Parent;
    Itr->second->Parent = NewParent;
    ++Itr;
    while (Itr != BinaryDataMap.end() && OldParent &&
           Itr->second->Parent == OldParent) {
      Itr->second->Parent = NewParent;
      ++Itr;
    }
  };

  // Check if the previous symbol contains the newly added symbol.
  if (GAI != BinaryDataMap.begin()) {
    BinaryData *Prev = std::prev(GAI)->second;
    while (Prev) {
      if (Prev->getSection() == GAI->second->getSection() &&
          Prev->containsRange(Address, Size)) {
        fixParents(GAI, Prev);
      } else {
        fixParents(GAI, nullptr);
      }
      Prev = Prev->Parent;
    }
  }

  // Check if the newly added symbol contains any subsequent symbols.
  if (Size != 0) {
    BinaryData *BD = GAI->second->Parent ? GAI->second->Parent : GAI->second;
    auto Itr = std::next(GAI);
    while (
        Itr != BinaryDataMap.end() &&
        BD->containsRange(Itr->second->getAddress(), Itr->second->getSize())) {
      Itr->second->Parent = BD;
      ++Itr;
    }
  }
}

iterator_range<BinaryContext::binary_data_iterator>
BinaryContext::getSubBinaryData(BinaryData *BD) {
  auto Start = std::next(BinaryDataMap.find(BD->getAddress()));
  auto End = Start;
  while (End != BinaryDataMap.end() && BD->isAncestorOf(End->second))
    ++End;
  return make_range(Start, End);
}

std::pair<const MCSymbol *, uint64_t>
BinaryContext::handleAddressRef(uint64_t Address, BinaryFunction &BF,
                                bool IsPCRel) {
  if (isAArch64()) {
    // Check if this is an access to a constant island and create bookkeeping
    // to keep track of it and emit it later as part of this function.
    if (MCSymbol *IslandSym = BF.getOrCreateIslandAccess(Address))
      return std::make_pair(IslandSym, 0);

    // Detect custom code written in assembly that refers to arbitrary
    // constant islands from other functions. Write this reference so we
    // can pull this constant island and emit it as part of this function
    // too.
    auto IslandIter = AddressToConstantIslandMap.lower_bound(Address);

    if (IslandIter != AddressToConstantIslandMap.begin() &&
        (IslandIter == AddressToConstantIslandMap.end() ||
         IslandIter->first > Address))
      --IslandIter;

    if (IslandIter != AddressToConstantIslandMap.end()) {
      // Fall-back to referencing the original constant island in the presence
      // of dynamic relocs, as we currently do not support cloning them.
      // Notice: we might fail to link because of this, if the original constant
      // island we are referring would be emitted too far away.
      if (IslandIter->second->hasDynamicRelocationAtIsland()) {
        MCSymbol *IslandSym =
            IslandIter->second->getOrCreateIslandAccess(Address);
        if (IslandSym)
          return std::make_pair(IslandSym, 0);
      } else if (MCSymbol *IslandSym =
                     IslandIter->second->getOrCreateProxyIslandAccess(Address,
                                                                      BF)) {
        BF.createIslandDependency(IslandSym, IslandIter->second);
        return std::make_pair(IslandSym, 0);
      }
    }
  }

  // Note that the address does not necessarily have to reside inside
  // a section, it could be an absolute address too.
  ErrorOr<BinarySection &> Section = getSectionForAddress(Address);
  if (Section && Section->isText()) {
    if (BF.containsAddress(Address, /*UseMaxSize=*/isAArch64())) {
      if (Address != BF.getAddress()) {
        // The address could potentially escape. Mark it as another entry
        // point into the function.
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: potentially escaped address 0x"
                 << Twine::utohexstr(Address) << " in function " << BF << '\n';
        }
        BF.HasInternalLabelReference = true;
        return std::make_pair(
            BF.addEntryPointAtOffset(Address - BF.getAddress()), 0);
      }
    } else {
      addInterproceduralReference(&BF, Address);
    }
  }

  // With relocations, catch jump table references outside of the basic block
  // containing the indirect jump.
  if (HasRelocations) {
    const MemoryContentsType MemType = analyzeMemoryAt(Address, BF);
    if (MemType == MemoryContentsType::POSSIBLE_PIC_JUMP_TABLE && IsPCRel) {
      const MCSymbol *Symbol =
          getOrCreateJumpTable(BF, Address, JumpTable::JTT_PIC);

      return std::make_pair(Symbol, 0);
    }
  }

  if (BinaryData *BD = getBinaryDataContainingAddress(Address))
    return std::make_pair(BD->getSymbol(), Address - BD->getAddress());

  // TODO: use DWARF info to get size/alignment here?
  MCSymbol *TargetSymbol = getOrCreateGlobalSymbol(Address, "DATAat");
  LLVM_DEBUG(dbgs() << "Created symbol " << TargetSymbol->getName() << '\n');
  return std::make_pair(TargetSymbol, 0);
}

MemoryContentsType BinaryContext::analyzeMemoryAt(uint64_t Address,
                                                  BinaryFunction &BF) {
  if (!isX86())
    return MemoryContentsType::UNKNOWN;

  ErrorOr<BinarySection &> Section = getSectionForAddress(Address);
  if (!Section) {
    // No section - possibly an absolute address. Since we don't allow
    // internal function addresses to escape the function scope - we
    // consider it a tail call.
    if (opts::Verbosity > 1) {
      errs() << "BOLT-WARNING: no section for address 0x"
             << Twine::utohexstr(Address) << " referenced from function " << BF
             << '\n';
    }
    return MemoryContentsType::UNKNOWN;
  }

  if (Section->isVirtual()) {
    // The contents are filled at runtime.
    return MemoryContentsType::UNKNOWN;
  }

  // No support for jump tables in code yet.
  if (Section->isText())
    return MemoryContentsType::UNKNOWN;

  // Start with checking for PIC jump table. We expect non-PIC jump tables
  // to have high 32 bits set to 0.
  if (analyzeJumpTable(Address, JumpTable::JTT_PIC, BF))
    return MemoryContentsType::POSSIBLE_PIC_JUMP_TABLE;

  if (analyzeJumpTable(Address, JumpTable::JTT_NORMAL, BF))
    return MemoryContentsType::POSSIBLE_JUMP_TABLE;

  return MemoryContentsType::UNKNOWN;
}

bool BinaryContext::analyzeJumpTable(const uint64_t Address,
                                     const JumpTable::JumpTableType Type,
                                     const BinaryFunction &BF,
                                     const uint64_t NextJTAddress,
                                     JumpTable::AddressesType *EntriesAsAddress,
                                     bool *HasEntryInFragment) const {
  // Is one of the targets __builtin_unreachable?
  bool HasUnreachable = false;

  // Does one of the entries match function start address?
  bool HasStartAsEntry = false;

  // Number of targets other than __builtin_unreachable.
  uint64_t NumRealEntries = 0;

  auto addEntryAddress = [&](uint64_t EntryAddress) {
    if (EntriesAsAddress)
      EntriesAsAddress->emplace_back(EntryAddress);
  };

  ErrorOr<const BinarySection &> Section = getSectionForAddress(Address);
  if (!Section)
    return false;

  // The upper bound is defined by containing object, section limits, and
  // the next jump table in memory.
  uint64_t UpperBound = Section->getEndAddress();
  const BinaryData *JumpTableBD = getBinaryDataAtAddress(Address);
  if (JumpTableBD && JumpTableBD->getSize()) {
    assert(JumpTableBD->getEndAddress() <= UpperBound &&
           "data object cannot cross a section boundary");
    UpperBound = JumpTableBD->getEndAddress();
  }
  if (NextJTAddress)
    UpperBound = std::min(NextJTAddress, UpperBound);

  LLVM_DEBUG({
    using JTT = JumpTable::JumpTableType;
    dbgs() << formatv("BOLT-DEBUG: analyzeJumpTable @{0:x} in {1}, JTT={2}\n",
                      Address, BF.getPrintName(),
                      Type == JTT::JTT_PIC ? "PIC" : "Normal");
  });
  const uint64_t EntrySize = getJumpTableEntrySize(Type);
  for (uint64_t EntryAddress = Address; EntryAddress <= UpperBound - EntrySize;
       EntryAddress += EntrySize) {
    LLVM_DEBUG(dbgs() << "  * Checking 0x" << Twine::utohexstr(EntryAddress)
                      << " -> ");
    // Check if there's a proper relocation against the jump table entry.
    if (HasRelocations) {
      if (Type == JumpTable::JTT_PIC &&
          !DataPCRelocations.count(EntryAddress)) {
        LLVM_DEBUG(
            dbgs() << "FAIL: JTT_PIC table, no relocation for this address\n");
        break;
      }
      if (Type == JumpTable::JTT_NORMAL && !getRelocationAt(EntryAddress)) {
        LLVM_DEBUG(
            dbgs()
            << "FAIL: JTT_NORMAL table, no relocation for this address\n");
        break;
      }
    }

    const uint64_t Value =
        (Type == JumpTable::JTT_PIC)
            ? Address + *getSignedValueAtAddress(EntryAddress, EntrySize)
            : *getPointerAtAddress(EntryAddress);

    // __builtin_unreachable() case.
    if (Value == BF.getAddress() + BF.getSize()) {
      addEntryAddress(Value);
      HasUnreachable = true;
      LLVM_DEBUG(dbgs() << formatv("OK: {0:x} __builtin_unreachable\n", Value));
      continue;
    }

    // Function start is another special case. It is allowed in the jump table,
    // but we need at least one another regular entry to distinguish the table
    // from, e.g. a function pointer array.
    if (Value == BF.getAddress()) {
      HasStartAsEntry = true;
      addEntryAddress(Value);
      continue;
    }

    // Function or one of its fragments.
    const BinaryFunction *TargetBF = getBinaryFunctionContainingAddress(Value);
    const bool DoesBelongToFunction =
        BF.containsAddress(Value) ||
        (TargetBF && TargetBF->isParentOrChildOf(BF));
    if (!DoesBelongToFunction) {
      LLVM_DEBUG({
        if (!BF.containsAddress(Value)) {
          dbgs() << "FAIL: function doesn't contain this address\n";
          if (TargetBF) {
            dbgs() << "  ! function containing this address: "
                   << TargetBF->getPrintName() << '\n';
            if (TargetBF->isFragment()) {
              dbgs() << "  ! is a fragment";
              for (BinaryFunction *Parent : TargetBF->ParentFragments)
                dbgs() << ", parent: " << Parent->getPrintName();
              dbgs() << '\n';
            }
          }
        }
      });
      break;
    }

    // Check there's an instruction at this offset.
    if (TargetBF->getState() == BinaryFunction::State::Disassembled &&
        !TargetBF->getInstructionAtOffset(Value - TargetBF->getAddress())) {
      LLVM_DEBUG(dbgs() << formatv("FAIL: no instruction at {0:x}\n", Value));
      break;
    }

    ++NumRealEntries;
    LLVM_DEBUG(dbgs() << formatv("OK: {0:x} real entry\n", Value));

    if (TargetBF != &BF && HasEntryInFragment)
      *HasEntryInFragment = true;
    addEntryAddress(Value);
  }

  // It's a jump table if the number of real entries is more than 1, or there's
  // one real entry and one or more special targets. If there are only multiple
  // special targets, then it's not a jump table.
  return NumRealEntries + (HasUnreachable || HasStartAsEntry) >= 2;
}

void BinaryContext::populateJumpTables() {
  LLVM_DEBUG(dbgs() << "DataPCRelocations: " << DataPCRelocations.size()
                    << '\n');
  for (auto JTI = JumpTables.begin(), JTE = JumpTables.end(); JTI != JTE;
       ++JTI) {
    JumpTable *JT = JTI->second;

    bool NonSimpleParent = false;
    for (BinaryFunction *BF : JT->Parents)
      NonSimpleParent |= !BF->isSimple();
    if (NonSimpleParent)
      continue;

    uint64_t NextJTAddress = 0;
    auto NextJTI = std::next(JTI);
    if (NextJTI != JTE)
      NextJTAddress = NextJTI->second->getAddress();

    const bool Success =
        analyzeJumpTable(JT->getAddress(), JT->Type, *(JT->Parents[0]),
                         NextJTAddress, &JT->EntriesAsAddress, &JT->IsSplit);
    if (!Success) {
      LLVM_DEBUG({
        dbgs() << "failed to analyze ";
        JT->print(dbgs());
        if (NextJTI != JTE) {
          dbgs() << "next ";
          NextJTI->second->print(dbgs());
        }
      });
      llvm_unreachable("jump table heuristic failure");
    }
    for (BinaryFunction *Frag : JT->Parents) {
      if (JT->IsSplit)
        Frag->setHasIndirectTargetToSplitFragment(true);
      for (uint64_t EntryAddress : JT->EntriesAsAddress)
        // if target is builtin_unreachable
        if (EntryAddress == Frag->getAddress() + Frag->getSize()) {
          Frag->IgnoredBranches.emplace_back(EntryAddress - Frag->getAddress(),
                                             Frag->getSize());
        } else if (EntryAddress >= Frag->getAddress() &&
                   EntryAddress < Frag->getAddress() + Frag->getSize()) {
          Frag->registerReferencedOffset(EntryAddress - Frag->getAddress());
        }
    }

    // In strict mode, erase PC-relative relocation record. Later we check that
    // all such records are erased and thus have been accounted for.
    if (opts::StrictMode && JT->Type == JumpTable::JTT_PIC) {
      for (uint64_t Address = JT->getAddress();
           Address < JT->getAddress() + JT->getSize();
           Address += JT->EntrySize) {
        DataPCRelocations.erase(DataPCRelocations.find(Address));
      }
    }

    // Mark to skip the function and all its fragments.
    for (BinaryFunction *Frag : JT->Parents)
      if (Frag->hasIndirectTargetToSplitFragment())
        addFragmentsToSkip(Frag);
  }

  if (opts::StrictMode && DataPCRelocations.size()) {
    LLVM_DEBUG({
      dbgs() << DataPCRelocations.size()
             << " unclaimed PC-relative relocations left in data:\n";
      for (uint64_t Reloc : DataPCRelocations)
        dbgs() << Twine::utohexstr(Reloc) << '\n';
    });
    assert(0 && "unclaimed PC-relative relocations left in data\n");
  }
  clearList(DataPCRelocations);
}

void BinaryContext::skipMarkedFragments() {
  std::vector<BinaryFunction *> FragmentQueue;
  // Copy the functions to FragmentQueue.
  FragmentQueue.assign(FragmentsToSkip.begin(), FragmentsToSkip.end());
  auto addToWorklist = [&](BinaryFunction *Function) -> void {
    if (FragmentsToSkip.count(Function))
      return;
    FragmentQueue.push_back(Function);
    addFragmentsToSkip(Function);
  };
  // Functions containing split jump tables need to be skipped with all
  // fragments (transitively).
  for (size_t I = 0; I != FragmentQueue.size(); I++) {
    BinaryFunction *BF = FragmentQueue[I];
    assert(FragmentsToSkip.count(BF) &&
           "internal error in traversing function fragments");
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: Ignoring " << BF->getPrintName() << '\n';
    BF->setSimple(false);
    BF->setHasIndirectTargetToSplitFragment(true);

    llvm::for_each(BF->Fragments, addToWorklist);
    llvm::for_each(BF->ParentFragments, addToWorklist);
  }
  if (!FragmentsToSkip.empty())
    errs() << "BOLT-WARNING: skipped " << FragmentsToSkip.size() << " function"
           << (FragmentsToSkip.size() == 1 ? "" : "s")
           << " due to cold fragments\n";
}

MCSymbol *BinaryContext::getOrCreateGlobalSymbol(uint64_t Address, Twine Prefix,
                                                 uint64_t Size,
                                                 uint16_t Alignment,
                                                 unsigned Flags) {
  auto Itr = BinaryDataMap.find(Address);
  if (Itr != BinaryDataMap.end()) {
    assert(Itr->second->getSize() == Size || !Size);
    return Itr->second->getSymbol();
  }

  std::string Name = (Prefix + "0x" + Twine::utohexstr(Address)).str();
  assert(!GlobalSymbols.count(Name) && "created name is not unique");
  return registerNameAtAddress(Name, Address, Size, Alignment, Flags);
}

MCSymbol *BinaryContext::getOrCreateUndefinedGlobalSymbol(StringRef Name) {
  return Ctx->getOrCreateSymbol(Name);
}

BinaryFunction *BinaryContext::createBinaryFunction(
    const std::string &Name, BinarySection &Section, uint64_t Address,
    uint64_t Size, uint64_t SymbolSize, uint16_t Alignment) {
  auto Result = BinaryFunctions.emplace(
      Address, BinaryFunction(Name, Section, Address, Size, *this));
  assert(Result.second == true && "unexpected duplicate function");
  BinaryFunction *BF = &Result.first->second;
  registerNameAtAddress(Name, Address, SymbolSize ? SymbolSize : Size,
                        Alignment);
  setSymbolToFunctionMap(BF->getSymbol(), BF);
  return BF;
}

const MCSymbol *
BinaryContext::getOrCreateJumpTable(BinaryFunction &Function, uint64_t Address,
                                    JumpTable::JumpTableType Type) {
  // Two fragments of same function access same jump table
  if (JumpTable *JT = getJumpTableContainingAddress(Address)) {
    assert(JT->Type == Type && "jump table types have to match");
    assert(Address == JT->getAddress() && "unexpected non-empty jump table");

    // Prevent associating a jump table to a specific fragment twice.
    // This simple check arises from the assumption: no more than 2 fragments.
    if (JT->Parents.size() == 1 && JT->Parents[0] != &Function) {
      assert(JT->Parents[0]->isParentOrChildOf(Function) &&
             "cannot re-use jump table of a different function");
      // Duplicate the entry for the parent function for easy access
      JT->Parents.push_back(&Function);
      if (opts::Verbosity > 2) {
        outs() << "BOLT-INFO: Multiple fragments access same jump table: "
               << JT->Parents[0]->getPrintName() << "; "
               << Function.getPrintName() << "\n";
        JT->print(outs());
      }
      Function.JumpTables.emplace(Address, JT);
      JT->Parents[0]->setHasIndirectTargetToSplitFragment(true);
      JT->Parents[1]->setHasIndirectTargetToSplitFragment(true);
    }

    bool IsJumpTableParent = false;
    (void)IsJumpTableParent;
    for (BinaryFunction *Frag : JT->Parents)
      if (Frag == &Function)
        IsJumpTableParent = true;
    assert(IsJumpTableParent &&
           "cannot re-use jump table of a different function");
    return JT->getFirstLabel();
  }

  // Re-use the existing symbol if possible.
  MCSymbol *JTLabel = nullptr;
  if (BinaryData *Object = getBinaryDataAtAddress(Address)) {
    if (!isInternalSymbolName(Object->getSymbol()->getName()))
      JTLabel = Object->getSymbol();
  }

  const uint64_t EntrySize = getJumpTableEntrySize(Type);
  if (!JTLabel) {
    const std::string JumpTableName = generateJumpTableName(Function, Address);
    JTLabel = registerNameAtAddress(JumpTableName, Address, 0, EntrySize);
  }

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: creating jump table " << JTLabel->getName()
                    << " in function " << Function << '\n');

  JumpTable *JT = new JumpTable(*JTLabel, Address, EntrySize, Type,
                                JumpTable::LabelMapType{{0, JTLabel}},
                                *getSectionForAddress(Address));
  JT->Parents.push_back(&Function);
  if (opts::Verbosity > 2)
    JT->print(outs());
  JumpTables.emplace(Address, JT);

  // Duplicate the entry for the parent function for easy access.
  Function.JumpTables.emplace(Address, JT);
  return JTLabel;
}

std::pair<uint64_t, const MCSymbol *>
BinaryContext::duplicateJumpTable(BinaryFunction &Function, JumpTable *JT,
                                  const MCSymbol *OldLabel) {
  auto L = scopeLock();
  unsigned Offset = 0;
  bool Found = false;
  for (std::pair<const unsigned, MCSymbol *> Elmt : JT->Labels) {
    if (Elmt.second != OldLabel)
      continue;
    Offset = Elmt.first;
    Found = true;
    break;
  }
  assert(Found && "Label not found");
  (void)Found;
  MCSymbol *NewLabel = Ctx->createNamedTempSymbol("duplicatedJT");
  JumpTable *NewJT =
      new JumpTable(*NewLabel, JT->getAddress(), JT->EntrySize, JT->Type,
                    JumpTable::LabelMapType{{Offset, NewLabel}},
                    *getSectionForAddress(JT->getAddress()));
  NewJT->Parents = JT->Parents;
  NewJT->Entries = JT->Entries;
  NewJT->Counts = JT->Counts;
  uint64_t JumpTableID = ++DuplicatedJumpTables;
  // Invert it to differentiate from regular jump tables whose IDs are their
  // addresses in the input binary memory space
  JumpTableID = ~JumpTableID;
  JumpTables.emplace(JumpTableID, NewJT);
  Function.JumpTables.emplace(JumpTableID, NewJT);
  return std::make_pair(JumpTableID, NewLabel);
}

std::string BinaryContext::generateJumpTableName(const BinaryFunction &BF,
                                                 uint64_t Address) {
  size_t Id;
  uint64_t Offset = 0;
  if (const JumpTable *JT = BF.getJumpTableContainingAddress(Address)) {
    Offset = Address - JT->getAddress();
    auto Itr = JT->Labels.find(Offset);
    if (Itr != JT->Labels.end())
      return std::string(Itr->second->getName());
    Id = JumpTableIds.at(JT->getAddress());
  } else {
    Id = JumpTableIds[Address] = BF.JumpTables.size();
  }
  return ("JUMP_TABLE/" + BF.getOneName().str() + "." + std::to_string(Id) +
          (Offset ? ("." + std::to_string(Offset)) : ""));
}

bool BinaryContext::hasValidCodePadding(const BinaryFunction &BF) {
  // FIXME: aarch64 support is missing.
  if (!isX86())
    return true;

  if (BF.getSize() == BF.getMaxSize())
    return true;

  ErrorOr<ArrayRef<unsigned char>> FunctionData = BF.getData();
  assert(FunctionData && "cannot get function as data");

  uint64_t Offset = BF.getSize();
  MCInst Instr;
  uint64_t InstrSize = 0;
  uint64_t InstrAddress = BF.getAddress() + Offset;
  using std::placeholders::_1;

  // Skip instructions that satisfy the predicate condition.
  auto skipInstructions = [&](std::function<bool(const MCInst &)> Predicate) {
    const uint64_t StartOffset = Offset;
    for (; Offset < BF.getMaxSize();
         Offset += InstrSize, InstrAddress += InstrSize) {
      if (!DisAsm->getInstruction(Instr, InstrSize, FunctionData->slice(Offset),
                                  InstrAddress, nulls()))
        break;
      if (!Predicate(Instr))
        break;
    }

    return Offset - StartOffset;
  };

  // Skip a sequence of zero bytes.
  auto skipZeros = [&]() {
    const uint64_t StartOffset = Offset;
    for (; Offset < BF.getMaxSize(); ++Offset)
      if ((*FunctionData)[Offset] != 0)
        break;

    return Offset - StartOffset;
  };

  // Accept the whole padding area filled with breakpoints.
  auto isBreakpoint = std::bind(&MCPlusBuilder::isBreakpoint, MIB.get(), _1);
  if (skipInstructions(isBreakpoint) && Offset == BF.getMaxSize())
    return true;

  auto isNoop = std::bind(&MCPlusBuilder::isNoop, MIB.get(), _1);

  // Some functions have a jump to the next function or to the padding area
  // inserted after the body.
  auto isSkipJump = [&](const MCInst &Instr) {
    uint64_t TargetAddress = 0;
    if (MIB->isUnconditionalBranch(Instr) &&
        MIB->evaluateBranch(Instr, InstrAddress, InstrSize, TargetAddress)) {
      if (TargetAddress >= InstrAddress + InstrSize &&
          TargetAddress <= BF.getAddress() + BF.getMaxSize()) {
        return true;
      }
    }
    return false;
  };

  // Skip over nops, jumps, and zero padding. Allow interleaving (this happens).
  while (skipInstructions(isNoop) || skipInstructions(isSkipJump) ||
         skipZeros())
    ;

  if (Offset == BF.getMaxSize())
    return true;

  if (opts::Verbosity >= 1) {
    errs() << "BOLT-WARNING: bad padding at address 0x"
           << Twine::utohexstr(BF.getAddress() + BF.getSize())
           << " starting at offset " << (Offset - BF.getSize())
           << " in function " << BF << '\n'
           << FunctionData->slice(BF.getSize(), BF.getMaxSize() - BF.getSize())
           << '\n';
  }

  return false;
}

void BinaryContext::adjustCodePadding() {
  for (auto &BFI : BinaryFunctions) {
    BinaryFunction &BF = BFI.second;
    if (!shouldEmit(BF))
      continue;

    if (!hasValidCodePadding(BF)) {
      if (HasRelocations) {
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: function " << BF
                 << " has invalid padding. Ignoring the function.\n";
        }
        BF.setIgnored();
      } else {
        BF.setMaxSize(BF.getSize());
      }
    }
  }
}

MCSymbol *BinaryContext::registerNameAtAddress(StringRef Name, uint64_t Address,
                                               uint64_t Size,
                                               uint16_t Alignment,
                                               unsigned Flags) {
  // Register the name with MCContext.
  MCSymbol *Symbol = Ctx->getOrCreateSymbol(Name);

  auto GAI = BinaryDataMap.find(Address);
  BinaryData *BD;
  if (GAI == BinaryDataMap.end()) {
    ErrorOr<BinarySection &> SectionOrErr = getSectionForAddress(Address);
    BinarySection &Section =
        SectionOrErr ? SectionOrErr.get() : absoluteSection();
    BD = new BinaryData(*Symbol, Address, Size, Alignment ? Alignment : 1,
                        Section, Flags);
    GAI = BinaryDataMap.emplace(Address, BD).first;
    GlobalSymbols[Name] = BD;
    updateObjectNesting(GAI);
  } else {
    BD = GAI->second;
    if (!BD->hasName(Name)) {
      GlobalSymbols[Name] = BD;
      BD->Symbols.push_back(Symbol);
    }
  }

  return Symbol;
}

const BinaryData *
BinaryContext::getBinaryDataContainingAddressImpl(uint64_t Address) const {
  auto NI = BinaryDataMap.lower_bound(Address);
  auto End = BinaryDataMap.end();
  if ((NI != End && Address == NI->first) ||
      ((NI != BinaryDataMap.begin()) && (NI-- != BinaryDataMap.begin()))) {
    if (NI->second->containsAddress(Address))
      return NI->second;

    // If this is a sub-symbol, see if a parent data contains the address.
    const BinaryData *BD = NI->second->getParent();
    while (BD) {
      if (BD->containsAddress(Address))
        return BD;
      BD = BD->getParent();
    }
  }
  return nullptr;
}

BinaryData *BinaryContext::getGOTSymbol() {
  // First tries to find a global symbol with that name
  BinaryData *GOTSymBD = getBinaryDataByName("_GLOBAL_OFFSET_TABLE_");
  if (GOTSymBD)
    return GOTSymBD;

  // This symbol might be hidden from run-time link, so fetch the local
  // definition if available.
  GOTSymBD = getBinaryDataByName("_GLOBAL_OFFSET_TABLE_/1");
  if (!GOTSymBD)
    return nullptr;

  // If the local symbol is not unique, fail
  unsigned Index = 2;
  SmallString<30> Storage;
  while (const BinaryData *BD =
             getBinaryDataByName(Twine("_GLOBAL_OFFSET_TABLE_/")
                                     .concat(Twine(Index++))
                                     .toStringRef(Storage)))
    if (BD->getAddress() != GOTSymBD->getAddress())
      return nullptr;

  return GOTSymBD;
}

bool BinaryContext::setBinaryDataSize(uint64_t Address, uint64_t Size) {
  auto NI = BinaryDataMap.find(Address);
  assert(NI != BinaryDataMap.end());
  if (NI == BinaryDataMap.end())
    return false;
  // TODO: it's possible that a jump table starts at the same address
  // as a larger blob of private data.  When we set the size of the
  // jump table, it might be smaller than the total blob size.  In this
  // case we just leave the original size since (currently) it won't really
  // affect anything.
  assert((!NI->second->Size || NI->second->Size == Size ||
          (NI->second->isJumpTable() && NI->second->Size > Size)) &&
         "can't change the size of a symbol that has already had its "
         "size set");
  if (!NI->second->Size) {
    NI->second->Size = Size;
    updateObjectNesting(NI);
    return true;
  }
  return false;
}

void BinaryContext::generateSymbolHashes() {
  auto isPadding = [](const BinaryData &BD) {
    StringRef Contents = BD.getSection().getContents();
    StringRef SymData = Contents.substr(BD.getOffset(), BD.getSize());
    return (BD.getName().startswith("HOLEat") ||
            SymData.find_first_not_of(0) == StringRef::npos);
  };

  uint64_t NumCollisions = 0;
  for (auto &Entry : BinaryDataMap) {
    BinaryData &BD = *Entry.second;
    StringRef Name = BD.getName();

    if (!isInternalSymbolName(Name))
      continue;

    // First check if a non-anonymous alias exists and move it to the front.
    if (BD.getSymbols().size() > 1) {
      auto Itr = llvm::find_if(BD.getSymbols(), [&](const MCSymbol *Symbol) {
        return !isInternalSymbolName(Symbol->getName());
      });
      if (Itr != BD.getSymbols().end()) {
        size_t Idx = std::distance(BD.getSymbols().begin(), Itr);
        std::swap(BD.getSymbols()[0], BD.getSymbols()[Idx]);
        continue;
      }
    }

    // We have to skip 0 size symbols since they will all collide.
    if (BD.getSize() == 0) {
      continue;
    }

    const uint64_t Hash = BD.getSection().hash(BD);
    const size_t Idx = Name.find("0x");
    std::string NewName =
        (Twine(Name.substr(0, Idx)) + "_" + Twine::utohexstr(Hash)).str();
    if (getBinaryDataByName(NewName)) {
      // Ignore collisions for symbols that appear to be padding
      // (i.e. all zeros or a "hole")
      if (!isPadding(BD)) {
        if (opts::Verbosity) {
          errs() << "BOLT-WARNING: collision detected when hashing " << BD
                 << " with new name (" << NewName << "), skipping.\n";
        }
        ++NumCollisions;
      }
      continue;
    }
    BD.Symbols.insert(BD.Symbols.begin(), Ctx->getOrCreateSymbol(NewName));
    GlobalSymbols[NewName] = &BD;
  }
  if (NumCollisions) {
    errs() << "BOLT-WARNING: " << NumCollisions
           << " collisions detected while hashing binary objects";
    if (!opts::Verbosity)
      errs() << ". Use -v=1 to see the list.";
    errs() << '\n';
  }
}

bool BinaryContext::registerFragment(BinaryFunction &TargetFunction,
                                     BinaryFunction &Function) const {
  assert(TargetFunction.isFragment() && "TargetFunction must be a fragment");
  if (TargetFunction.isChildOf(Function))
    return true;
  TargetFunction.addParentFragment(Function);
  Function.addFragment(TargetFunction);
  if (!HasRelocations) {
    TargetFunction.setSimple(false);
    Function.setSimple(false);
  }
  if (opts::Verbosity >= 1) {
    outs() << "BOLT-INFO: marking " << TargetFunction << " as a fragment of "
           << Function << '\n';
  }
  return true;
}

void BinaryContext::addAdrpAddRelocAArch64(BinaryFunction &BF,
                                           MCInst &LoadLowBits,
                                           MCInst &LoadHiBits,
                                           uint64_t Target) {
  const MCSymbol *TargetSymbol;
  uint64_t Addend = 0;
  std::tie(TargetSymbol, Addend) = handleAddressRef(Target, BF,
                                                    /*IsPCRel*/ true);
  int64_t Val;
  MIB->replaceImmWithSymbolRef(LoadHiBits, TargetSymbol, Addend, Ctx.get(), Val,
                               ELF::R_AARCH64_ADR_PREL_PG_HI21);
  MIB->replaceImmWithSymbolRef(LoadLowBits, TargetSymbol, Addend, Ctx.get(),
                               Val, ELF::R_AARCH64_ADD_ABS_LO12_NC);
}

bool BinaryContext::handleAArch64Veneer(uint64_t Address, bool MatchOnly) {
  BinaryFunction *TargetFunction = getBinaryFunctionContainingAddress(Address);
  if (TargetFunction)
    return false;

  ErrorOr<BinarySection &> Section = getSectionForAddress(Address);
  assert(Section && "cannot get section for referenced address");
  if (!Section->isText())
    return false;

  bool Ret = false;
  StringRef SectionContents = Section->getContents();
  uint64_t Offset = Address - Section->getAddress();
  const uint64_t MaxSize = SectionContents.size() - Offset;
  const uint8_t *Bytes =
      reinterpret_cast<const uint8_t *>(SectionContents.data());
  ArrayRef<uint8_t> Data(Bytes + Offset, MaxSize);

  auto matchVeneer = [&](BinaryFunction::InstrMapType &Instructions,
                         MCInst &Instruction, uint64_t Offset,
                         uint64_t AbsoluteInstrAddr,
                         uint64_t TotalSize) -> bool {
    MCInst *TargetHiBits, *TargetLowBits;
    uint64_t TargetAddress, Count;
    Count = MIB->matchLinkerVeneer(Instructions.begin(), Instructions.end(),
                                   AbsoluteInstrAddr, Instruction, TargetHiBits,
                                   TargetLowBits, TargetAddress);
    if (!Count)
      return false;

    if (MatchOnly)
      return true;

    // NOTE The target symbol was created during disassemble's
    // handleExternalReference
    const MCSymbol *VeneerSymbol = getOrCreateGlobalSymbol(Address, "FUNCat");
    BinaryFunction *Veneer = createBinaryFunction(VeneerSymbol->getName().str(),
                                                  *Section, Address, TotalSize);
    addAdrpAddRelocAArch64(*Veneer, *TargetLowBits, *TargetHiBits,
                           TargetAddress);
    MIB->addAnnotation(Instruction, "AArch64Veneer", true);
    Veneer->addInstruction(Offset, std::move(Instruction));
    --Count;
    for (auto It = Instructions.rbegin(); Count != 0; ++It, --Count) {
      MIB->addAnnotation(It->second, "AArch64Veneer", true);
      Veneer->addInstruction(It->first, std::move(It->second));
    }

    Veneer->getOrCreateLocalLabel(Address);
    Veneer->setMaxSize(TotalSize);
    Veneer->updateState(BinaryFunction::State::Disassembled);
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: handling veneer function at 0x" << Address
                      << "\n");
    return true;
  };

  uint64_t Size = 0, TotalSize = 0;
  BinaryFunction::InstrMapType VeneerInstructions;
  for (Offset = 0; Offset < MaxSize; Offset += Size) {
    MCInst Instruction;
    const uint64_t AbsoluteInstrAddr = Address + Offset;
    if (!SymbolicDisAsm->getInstruction(Instruction, Size, Data.slice(Offset),
                                        AbsoluteInstrAddr, nulls()))
      break;

    TotalSize += Size;
    if (MIB->isBranch(Instruction)) {
      Ret = matchVeneer(VeneerInstructions, Instruction, Offset,
                        AbsoluteInstrAddr, TotalSize);
      break;
    }

    VeneerInstructions.emplace(Offset, std::move(Instruction));
  }

  return Ret;
}

void BinaryContext::processInterproceduralReferences() {
  for (const std::pair<BinaryFunction *, uint64_t> &It :
       InterproceduralReferences) {
    BinaryFunction &Function = *It.first;
    uint64_t Address = It.second;
    if (!Address || Function.isIgnored())
      continue;

    BinaryFunction *TargetFunction =
        getBinaryFunctionContainingAddress(Address);
    if (&Function == TargetFunction)
      continue;

    if (TargetFunction) {
      if (TargetFunction->isFragment() &&
          !TargetFunction->isChildOf(Function)) {
        errs() << "BOLT-WARNING: interprocedural reference between unrelated "
                  "fragments: "
               << Function.getPrintName() << " and "
               << TargetFunction->getPrintName() << '\n';
      }
      if (uint64_t Offset = Address - TargetFunction->getAddress())
        TargetFunction->addEntryPointAtOffset(Offset);

      continue;
    }

    // Check if address falls in function padding space - this could be
    // unmarked data in code. In this case adjust the padding space size.
    ErrorOr<BinarySection &> Section = getSectionForAddress(Address);
    assert(Section && "cannot get section for referenced address");

    if (!Section->isText())
      continue;

    // PLT requires special handling and could be ignored in this context.
    StringRef SectionName = Section->getName();
    if (SectionName == ".plt" || SectionName == ".plt.got")
      continue;

    // Check if it is aarch64 veneer written at Address
    if (isAArch64() && handleAArch64Veneer(Address))
      continue;

    if (opts::processAllFunctions()) {
      errs() << "BOLT-ERROR: cannot process binaries with unmarked "
             << "object in code at address 0x" << Twine::utohexstr(Address)
             << " belonging to section " << SectionName << " in current mode\n";
      exit(1);
    }

    TargetFunction = getBinaryFunctionContainingAddress(Address,
                                                        /*CheckPastEnd=*/false,
                                                        /*UseMaxSize=*/true);
    // We are not going to overwrite non-simple functions, but for simple
    // ones - adjust the padding size.
    if (TargetFunction && TargetFunction->isSimple()) {
      errs() << "BOLT-WARNING: function " << *TargetFunction
             << " has an object detected in a padding region at address 0x"
             << Twine::utohexstr(Address) << '\n';
      TargetFunction->setMaxSize(TargetFunction->getSize());
    }
  }

  InterproceduralReferences.clear();
}

void BinaryContext::postProcessSymbolTable() {
  fixBinaryDataHoles();
  bool Valid = true;
  for (auto &Entry : BinaryDataMap) {
    BinaryData *BD = Entry.second;
    if ((BD->getName().startswith("SYMBOLat") ||
         BD->getName().startswith("DATAat")) &&
        !BD->getParent() && !BD->getSize() && !BD->isAbsolute() &&
        BD->getSection()) {
      errs() << "BOLT-WARNING: zero-sized top level symbol: " << *BD << "\n";
      Valid = false;
    }
  }
  assert(Valid);
  (void)Valid;
  generateSymbolHashes();
}

void BinaryContext::foldFunction(BinaryFunction &ChildBF,
                                 BinaryFunction &ParentBF) {
  assert(!ChildBF.isMultiEntry() && !ParentBF.isMultiEntry() &&
         "cannot merge functions with multiple entry points");

  std::unique_lock<llvm::sys::RWMutex> WriteCtxLock(CtxMutex, std::defer_lock);
  std::unique_lock<llvm::sys::RWMutex> WriteSymbolMapLock(
      SymbolToFunctionMapMutex, std::defer_lock);

  const StringRef ChildName = ChildBF.getOneName();

  // Move symbols over and update bookkeeping info.
  for (MCSymbol *Symbol : ChildBF.getSymbols()) {
    ParentBF.getSymbols().push_back(Symbol);
    WriteSymbolMapLock.lock();
    SymbolToFunctionMap[Symbol] = &ParentBF;
    WriteSymbolMapLock.unlock();
    // NB: there's no need to update BinaryDataMap and GlobalSymbols.
  }
  ChildBF.getSymbols().clear();

  // Move other names the child function is known under.
  llvm::move(ChildBF.Aliases, std::back_inserter(ParentBF.Aliases));
  ChildBF.Aliases.clear();

  if (HasRelocations) {
    // Merge execution counts of ChildBF into those of ParentBF.
    // Without relocations, we cannot reliably merge profiles as both functions
    // continue to exist and either one can be executed.
    ChildBF.mergeProfileDataInto(ParentBF);

    std::shared_lock<llvm::sys::RWMutex> ReadBfsLock(BinaryFunctionsMutex,
                                                     std::defer_lock);
    std::unique_lock<llvm::sys::RWMutex> WriteBfsLock(BinaryFunctionsMutex,
                                                      std::defer_lock);
    // Remove ChildBF from the global set of functions in relocs mode.
    ReadBfsLock.lock();
    auto FI = BinaryFunctions.find(ChildBF.getAddress());
    ReadBfsLock.unlock();

    assert(FI != BinaryFunctions.end() && "function not found");
    assert(&ChildBF == &FI->second && "function mismatch");

    WriteBfsLock.lock();
    ChildBF.clearDisasmState();
    FI = BinaryFunctions.erase(FI);
    WriteBfsLock.unlock();

  } else {
    // In non-relocation mode we keep the function, but rename it.
    std::string NewName = "__ICF_" + ChildName.str();

    WriteCtxLock.lock();
    ChildBF.getSymbols().push_back(Ctx->getOrCreateSymbol(NewName));
    WriteCtxLock.unlock();

    ChildBF.setFolded(&ParentBF);
  }

  ParentBF.setHasFunctionsFoldedInto();
}

void BinaryContext::fixBinaryDataHoles() {
  assert(validateObjectNesting() && "object nesting inconsitency detected");

  for (BinarySection &Section : allocatableSections()) {
    std::vector<std::pair<uint64_t, uint64_t>> Holes;

    auto isNotHole = [&Section](const binary_data_iterator &Itr) {
      BinaryData *BD = Itr->second;
      bool isHole = (!BD->getParent() && !BD->getSize() && BD->isObject() &&
                     (BD->getName().startswith("SYMBOLat0x") ||
                      BD->getName().startswith("DATAat0x") ||
                      BD->getName().startswith("ANONYMOUS")));
      return !isHole && BD->getSection() == Section && !BD->getParent();
    };

    auto BDStart = BinaryDataMap.begin();
    auto BDEnd = BinaryDataMap.end();
    auto Itr = FilteredBinaryDataIterator(isNotHole, BDStart, BDEnd);
    auto End = FilteredBinaryDataIterator(isNotHole, BDEnd, BDEnd);

    uint64_t EndAddress = Section.getAddress();

    while (Itr != End) {
      if (Itr->second->getAddress() > EndAddress) {
        uint64_t Gap = Itr->second->getAddress() - EndAddress;
        Holes.emplace_back(EndAddress, Gap);
      }
      EndAddress = Itr->second->getEndAddress();
      ++Itr;
    }

    if (EndAddress < Section.getEndAddress())
      Holes.emplace_back(EndAddress, Section.getEndAddress() - EndAddress);

    // If there is already a symbol at the start of the hole, grow that symbol
    // to cover the rest.  Otherwise, create a new symbol to cover the hole.
    for (std::pair<uint64_t, uint64_t> &Hole : Holes) {
      BinaryData *BD = getBinaryDataAtAddress(Hole.first);
      if (BD) {
        // BD->getSection() can be != Section if there are sections that
        // overlap.  In this case it is probably safe to just skip the holes
        // since the overlapping section will not(?) have any symbols in it.
        if (BD->getSection() == Section)
          setBinaryDataSize(Hole.first, Hole.second);
      } else {
        getOrCreateGlobalSymbol(Hole.first, "HOLEat", Hole.second, 1);
      }
    }
  }

  assert(validateObjectNesting() && "object nesting inconsitency detected");
  assert(validateHoles() && "top level hole detected in object map");
}

void BinaryContext::printGlobalSymbols(raw_ostream &OS) const {
  const BinarySection *CurrentSection = nullptr;
  bool FirstSection = true;

  for (auto &Entry : BinaryDataMap) {
    const BinaryData *BD = Entry.second;
    const BinarySection &Section = BD->getSection();
    if (FirstSection || Section != *CurrentSection) {
      uint64_t Address, Size;
      StringRef Name = Section.getName();
      if (Section) {
        Address = Section.getAddress();
        Size = Section.getSize();
      } else {
        Address = BD->getAddress();
        Size = BD->getSize();
      }
      OS << "BOLT-INFO: Section " << Name << ", "
         << "0x" + Twine::utohexstr(Address) << ":"
         << "0x" + Twine::utohexstr(Address + Size) << "/" << Size << "\n";
      CurrentSection = &Section;
      FirstSection = false;
    }

    OS << "BOLT-INFO: ";
    const BinaryData *P = BD->getParent();
    while (P) {
      OS << "  ";
      P = P->getParent();
    }
    OS << *BD << "\n";
  }
}

Expected<unsigned> BinaryContext::getDwarfFile(
    StringRef Directory, StringRef FileName, unsigned FileNumber,
    std::optional<MD5::MD5Result> Checksum, std::optional<StringRef> Source,
    unsigned CUID, unsigned DWARFVersion) {
  DwarfLineTable &Table = DwarfLineTablesCUMap[CUID];
  return Table.tryGetFile(Directory, FileName, Checksum, Source, DWARFVersion,
                          FileNumber);
}

unsigned BinaryContext::addDebugFilenameToUnit(const uint32_t DestCUID,
                                               const uint32_t SrcCUID,
                                               unsigned FileIndex) {
  DWARFCompileUnit *SrcUnit = DwCtx->getCompileUnitForOffset(SrcCUID);
  const DWARFDebugLine::LineTable *LineTable =
      DwCtx->getLineTableForUnit(SrcUnit);
  const std::vector<DWARFDebugLine::FileNameEntry> &FileNames =
      LineTable->Prologue.FileNames;
  // Dir indexes start at 1, as DWARF file numbers, and a dir index 0
  // means empty dir.
  assert(FileIndex > 0 && FileIndex <= FileNames.size() &&
         "FileIndex out of range for the compilation unit.");
  StringRef Dir = "";
  if (FileNames[FileIndex - 1].DirIdx != 0) {
    if (std::optional<const char *> DirName = dwarf::toString(
            LineTable->Prologue
                .IncludeDirectories[FileNames[FileIndex - 1].DirIdx - 1])) {
      Dir = *DirName;
    }
  }
  StringRef FileName = "";
  if (std::optional<const char *> FName =
          dwarf::toString(FileNames[FileIndex - 1].Name))
    FileName = *FName;
  assert(FileName != "");
  DWARFCompileUnit *DstUnit = DwCtx->getCompileUnitForOffset(DestCUID);
  return cantFail(getDwarfFile(Dir, FileName, 0, std::nullopt, std::nullopt,
                               DestCUID, DstUnit->getVersion()));
}

std::vector<BinaryFunction *> BinaryContext::getSortedFunctions() {
  std::vector<BinaryFunction *> SortedFunctions(BinaryFunctions.size());
  llvm::transform(llvm::make_second_range(BinaryFunctions),
                  SortedFunctions.begin(),
                  [](BinaryFunction &BF) { return &BF; });

  llvm::stable_sort(SortedFunctions,
                    [](const BinaryFunction *A, const BinaryFunction *B) {
                      if (A->hasValidIndex() && B->hasValidIndex()) {
                        return A->getIndex() < B->getIndex();
                      }
                      return A->hasValidIndex();
                    });
  return SortedFunctions;
}

std::vector<BinaryFunction *> BinaryContext::getAllBinaryFunctions() {
  std::vector<BinaryFunction *> AllFunctions;
  AllFunctions.reserve(BinaryFunctions.size() + InjectedBinaryFunctions.size());
  llvm::transform(llvm::make_second_range(BinaryFunctions),
                  std::back_inserter(AllFunctions),
                  [](BinaryFunction &BF) { return &BF; });
  llvm::copy(InjectedBinaryFunctions, std::back_inserter(AllFunctions));

  return AllFunctions;
}

std::optional<DWARFUnit *> BinaryContext::getDWOCU(uint64_t DWOId) {
  auto Iter = DWOCUs.find(DWOId);
  if (Iter == DWOCUs.end())
    return std::nullopt;

  return Iter->second;
}

DWARFContext *BinaryContext::getDWOContext() const {
  if (DWOCUs.empty())
    return nullptr;
  return &DWOCUs.begin()->second->getContext();
}

/// Handles DWO sections that can either be in .o, .dwo or .dwp files.
void BinaryContext::preprocessDWODebugInfo() {
  for (const std::unique_ptr<DWARFUnit> &CU : DwCtx->compile_units()) {
    DWARFUnit *const DwarfUnit = CU.get();
    if (std::optional<uint64_t> DWOId = DwarfUnit->getDWOId()) {
      DWARFUnit *DWOCU = DwarfUnit->getNonSkeletonUnitDIE(false).getDwarfUnit();
      if (!DWOCU->isDWOUnit()) {
        std::string DWOName = dwarf::toString(
            DwarfUnit->getUnitDIE().find(
                {dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}),
            "");
        outs() << "BOLT-WARNING: Debug Fission: DWO debug information for "
               << DWOName
               << " was not retrieved and won't be updated. Please check "
                  "relative path.\n";
        continue;
      }
      DWOCUs[*DWOId] = DWOCU;
    }
  }
  if (!DWOCUs.empty())
    outs() << "BOLT-INFO: processing split DWARF\n";
}

void BinaryContext::preprocessDebugInfo() {
  struct CURange {
    uint64_t LowPC;
    uint64_t HighPC;
    DWARFUnit *Unit;

    bool operator<(const CURange &Other) const { return LowPC < Other.LowPC; }
  };

  // Building a map of address ranges to CUs similar to .debug_aranges and use
  // it to assign CU to functions.
  std::vector<CURange> AllRanges;
  AllRanges.reserve(DwCtx->getNumCompileUnits());
  for (const std::unique_ptr<DWARFUnit> &CU : DwCtx->compile_units()) {
    Expected<DWARFAddressRangesVector> RangesOrError =
        CU->getUnitDIE().getAddressRanges();
    if (!RangesOrError) {
      consumeError(RangesOrError.takeError());
      continue;
    }
    for (DWARFAddressRange &Range : *RangesOrError) {
      // Parts of the debug info could be invalidated due to corresponding code
      // being removed from the binary by the linker. Hence we check if the
      // address is a valid one.
      if (containsAddress(Range.LowPC))
        AllRanges.emplace_back(CURange{Range.LowPC, Range.HighPC, CU.get()});
    }

    ContainsDwarf5 |= CU->getVersion() >= 5;
    ContainsDwarfLegacy |= CU->getVersion() < 5;
  }

  llvm::sort(AllRanges);
  for (auto &KV : BinaryFunctions) {
    const uint64_t FunctionAddress = KV.first;
    BinaryFunction &Function = KV.second;

    auto It = llvm::partition_point(
        AllRanges, [=](CURange R) { return R.HighPC <= FunctionAddress; });
    if (It != AllRanges.end() && It->LowPC <= FunctionAddress)
      Function.setDWARFUnit(It->Unit);
  }

  // Discover units with debug info that needs to be updated.
  for (const auto &KV : BinaryFunctions) {
    const BinaryFunction &BF = KV.second;
    if (shouldEmit(BF) && BF.getDWARFUnit())
      ProcessedCUs.insert(BF.getDWARFUnit());
  }

  // Clear debug info for functions from units that we are not going to process.
  for (auto &KV : BinaryFunctions) {
    BinaryFunction &BF = KV.second;
    if (BF.getDWARFUnit() && !ProcessedCUs.count(BF.getDWARFUnit()))
      BF.setDWARFUnit(nullptr);
  }

  if (opts::Verbosity >= 1) {
    outs() << "BOLT-INFO: " << ProcessedCUs.size() << " out of "
           << DwCtx->getNumCompileUnits() << " CUs will be updated\n";
  }

  preprocessDWODebugInfo();

  // Populate MCContext with DWARF files from all units.
  StringRef GlobalPrefix = AsmInfo->getPrivateGlobalPrefix();
  for (const std::unique_ptr<DWARFUnit> &CU : DwCtx->compile_units()) {
    const uint64_t CUID = CU->getOffset();
    DwarfLineTable &BinaryLineTable = getDwarfLineTable(CUID);
    BinaryLineTable.setLabel(Ctx->getOrCreateSymbol(
        GlobalPrefix + "line_table_start" + Twine(CUID)));

    if (!ProcessedCUs.count(CU.get()))
      continue;

    const DWARFDebugLine::LineTable *LineTable =
        DwCtx->getLineTableForUnit(CU.get());
    const std::vector<DWARFDebugLine::FileNameEntry> &FileNames =
        LineTable->Prologue.FileNames;

    uint16_t DwarfVersion = LineTable->Prologue.getVersion();
    if (DwarfVersion >= 5) {
      std::optional<MD5::MD5Result> Checksum;
      if (LineTable->Prologue.ContentTypes.HasMD5)
        Checksum = LineTable->Prologue.FileNames[0].Checksum;
      std::optional<const char *> Name =
          dwarf::toString(CU->getUnitDIE().find(dwarf::DW_AT_name), nullptr);
      if (std::optional<uint64_t> DWOID = CU->getDWOId()) {
        auto Iter = DWOCUs.find(*DWOID);
        assert(Iter != DWOCUs.end() && "DWO CU was not found.");
        Name = dwarf::toString(
            Iter->second->getUnitDIE().find(dwarf::DW_AT_name), nullptr);
      }
      BinaryLineTable.setRootFile(CU->getCompilationDir(), *Name, Checksum,
                                  std::nullopt);
    }

    BinaryLineTable.setDwarfVersion(DwarfVersion);

    // Assign a unique label to every line table, one per CU.
    // Make sure empty debug line tables are registered too.
    if (FileNames.empty()) {
      cantFail(getDwarfFile("", "<unknown>", 0, std::nullopt, std::nullopt,
                            CUID, DwarfVersion));
      continue;
    }
    const uint32_t Offset = DwarfVersion < 5 ? 1 : 0;
    for (size_t I = 0, Size = FileNames.size(); I != Size; ++I) {
      // Dir indexes start at 1, as DWARF file numbers, and a dir index 0
      // means empty dir.
      StringRef Dir = "";
      if (FileNames[I].DirIdx != 0 || DwarfVersion >= 5)
        if (std::optional<const char *> DirName = dwarf::toString(
                LineTable->Prologue
                    .IncludeDirectories[FileNames[I].DirIdx - Offset]))
          Dir = *DirName;
      StringRef FileName = "";
      if (std::optional<const char *> FName =
              dwarf::toString(FileNames[I].Name))
        FileName = *FName;
      assert(FileName != "");
      std::optional<MD5::MD5Result> Checksum;
      if (DwarfVersion >= 5 && LineTable->Prologue.ContentTypes.HasMD5)
        Checksum = LineTable->Prologue.FileNames[I].Checksum;
      cantFail(getDwarfFile(Dir, FileName, 0, Checksum, std::nullopt, CUID,
                            DwarfVersion));
    }
  }
}

bool BinaryContext::shouldEmit(const BinaryFunction &Function) const {
  if (Function.isPseudo())
    return false;

  if (opts::processAllFunctions())
    return true;

  if (Function.isIgnored())
    return false;

  // In relocation mode we will emit non-simple functions with CFG.
  // If the function does not have a CFG it should be marked as ignored.
  return HasRelocations || Function.isSimple();
}

void BinaryContext::printCFI(raw_ostream &OS, const MCCFIInstruction &Inst) {
  uint32_t Operation = Inst.getOperation();
  switch (Operation) {
  case MCCFIInstruction::OpSameValue:
    OS << "OpSameValue Reg" << Inst.getRegister();
    break;
  case MCCFIInstruction::OpRememberState:
    OS << "OpRememberState";
    break;
  case MCCFIInstruction::OpRestoreState:
    OS << "OpRestoreState";
    break;
  case MCCFIInstruction::OpOffset:
    OS << "OpOffset Reg" << Inst.getRegister() << " " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpDefCfaRegister:
    OS << "OpDefCfaRegister Reg" << Inst.getRegister();
    break;
  case MCCFIInstruction::OpDefCfaOffset:
    OS << "OpDefCfaOffset " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpDefCfa:
    OS << "OpDefCfa Reg" << Inst.getRegister() << " " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpRelOffset:
    OS << "OpRelOffset Reg" << Inst.getRegister() << " " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    OS << "OfAdjustCfaOffset " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpEscape:
    OS << "OpEscape";
    break;
  case MCCFIInstruction::OpRestore:
    OS << "OpRestore Reg" << Inst.getRegister();
    break;
  case MCCFIInstruction::OpUndefined:
    OS << "OpUndefined Reg" << Inst.getRegister();
    break;
  case MCCFIInstruction::OpRegister:
    OS << "OpRegister Reg" << Inst.getRegister() << " Reg"
       << Inst.getRegister2();
    break;
  case MCCFIInstruction::OpWindowSave:
    OS << "OpWindowSave";
    break;
  case MCCFIInstruction::OpGnuArgsSize:
    OS << "OpGnuArgsSize";
    break;
  default:
    OS << "Op#" << Operation;
    break;
  }
}

MarkerSymType BinaryContext::getMarkerType(const SymbolRef &Symbol) const {
  // For aarch64 and riscv, the ABI defines mapping symbols so we identify data
  // in the code section (see IHI0056B). $x identifies a symbol starting code or
  // the end of a data chunk inside code, $d indentifies start of data.
  if ((!isAArch64() && !isRISCV()) || ELFSymbolRef(Symbol).getSize())
    return MarkerSymType::NONE;

  Expected<StringRef> NameOrError = Symbol.getName();
  Expected<object::SymbolRef::Type> TypeOrError = Symbol.getType();

  if (!TypeOrError || !NameOrError)
    return MarkerSymType::NONE;

  if (*TypeOrError != SymbolRef::ST_Unknown)
    return MarkerSymType::NONE;

  if (*NameOrError == "$x" || NameOrError->startswith("$x."))
    return MarkerSymType::CODE;

  // $x<ISA>
  if (isRISCV() && NameOrError->startswith("$x"))
    return MarkerSymType::CODE;

  if (*NameOrError == "$d" || NameOrError->startswith("$d."))
    return MarkerSymType::DATA;

  return MarkerSymType::NONE;
}

bool BinaryContext::isMarker(const SymbolRef &Symbol) const {
  return getMarkerType(Symbol) != MarkerSymType::NONE;
}

static void printDebugInfo(raw_ostream &OS, const MCInst &Instruction,
                           const BinaryFunction *Function,
                           DWARFContext *DwCtx) {
  DebugLineTableRowRef RowRef =
      DebugLineTableRowRef::fromSMLoc(Instruction.getLoc());
  if (RowRef == DebugLineTableRowRef::NULL_ROW)
    return;

  const DWARFDebugLine::LineTable *LineTable;
  if (Function && Function->getDWARFUnit() &&
      Function->getDWARFUnit()->getOffset() == RowRef.DwCompileUnitIndex) {
    LineTable = Function->getDWARFLineTable();
  } else {
    LineTable = DwCtx->getLineTableForUnit(
        DwCtx->getCompileUnitForOffset(RowRef.DwCompileUnitIndex));
  }
  assert(LineTable && "line table expected for instruction with debug info");

  const DWARFDebugLine::Row &Row = LineTable->Rows[RowRef.RowIndex - 1];
  StringRef FileName = "";
  if (std::optional<const char *> FName =
          dwarf::toString(LineTable->Prologue.FileNames[Row.File - 1].Name))
    FileName = *FName;
  OS << " # debug line " << FileName << ":" << Row.Line;
  if (Row.Column)
    OS << ":" << Row.Column;
  if (Row.Discriminator)
    OS << " discriminator:" << Row.Discriminator;
}

void BinaryContext::printInstruction(raw_ostream &OS, const MCInst &Instruction,
                                     uint64_t Offset,
                                     const BinaryFunction *Function,
                                     bool PrintMCInst, bool PrintMemData,
                                     bool PrintRelocations,
                                     StringRef Endl) const {
  if (MIB->isEHLabel(Instruction)) {
    OS << "  EH_LABEL: " << *MIB->getTargetSymbol(Instruction) << Endl;
    return;
  }
  OS << format("    %08" PRIx64 ": ", Offset);
  if (MIB->isCFI(Instruction)) {
    uint32_t Offset = Instruction.getOperand(0).getImm();
    OS << "\t!CFI\t$" << Offset << "\t; ";
    if (Function)
      printCFI(OS, *Function->getCFIFor(Instruction));
    OS << Endl;
    return;
  }
  InstPrinter->printInst(&Instruction, 0, "", *STI, OS);
  if (MIB->isCall(Instruction)) {
    if (MIB->isTailCall(Instruction))
      OS << " # TAILCALL ";
    if (MIB->isInvoke(Instruction)) {
      const std::optional<MCPlus::MCLandingPad> EHInfo =
          MIB->getEHInfo(Instruction);
      OS << " # handler: ";
      if (EHInfo->first)
        OS << *EHInfo->first;
      else
        OS << '0';
      OS << "; action: " << EHInfo->second;
      const int64_t GnuArgsSize = MIB->getGnuArgsSize(Instruction);
      if (GnuArgsSize >= 0)
        OS << "; GNU_args_size = " << GnuArgsSize;
    }
  } else if (MIB->isIndirectBranch(Instruction)) {
    if (uint64_t JTAddress = MIB->getJumpTable(Instruction)) {
      OS << " # JUMPTABLE @0x" << Twine::utohexstr(JTAddress);
    } else {
      OS << " # UNKNOWN CONTROL FLOW";
    }
  }
  if (std::optional<uint32_t> Offset = MIB->getOffset(Instruction))
    OS << " # Offset: " << *Offset;
  if (auto Label = MIB->getLabel(Instruction))
    OS << " # Label: " << **Label;

  MIB->printAnnotations(Instruction, OS);

  if (opts::PrintDebugInfo)
    printDebugInfo(OS, Instruction, Function, DwCtx.get());

  if ((opts::PrintRelocations || PrintRelocations) && Function) {
    const uint64_t Size = computeCodeSize(&Instruction, &Instruction + 1);
    Function->printRelocations(OS, Offset, Size);
  }

  OS << Endl;

  if (PrintMCInst) {
    Instruction.dump_pretty(OS, InstPrinter.get());
    OS << Endl;
  }
}

std::optional<uint64_t>
BinaryContext::getBaseAddressForMapping(uint64_t MMapAddress,
                                        uint64_t FileOffset) const {
  // Find a segment with a matching file offset.
  for (auto &KV : SegmentMapInfo) {
    const SegmentInfo &SegInfo = KV.second;
    if (alignDown(SegInfo.FileOffset, SegInfo.Alignment) == FileOffset) {
      // Use segment's aligned memory offset to calculate the base address.
      const uint64_t MemOffset = alignDown(SegInfo.Address, SegInfo.Alignment);
      return MMapAddress - MemOffset;
    }
  }

  return std::nullopt;
}

ErrorOr<BinarySection &> BinaryContext::getSectionForAddress(uint64_t Address) {
  auto SI = AddressToSection.upper_bound(Address);
  if (SI != AddressToSection.begin()) {
    --SI;
    uint64_t UpperBound = SI->first + SI->second->getSize();
    if (!SI->second->getSize())
      UpperBound += 1;
    if (UpperBound > Address)
      return *SI->second;
  }
  return std::make_error_code(std::errc::bad_address);
}

ErrorOr<StringRef>
BinaryContext::getSectionNameForAddress(uint64_t Address) const {
  if (ErrorOr<const BinarySection &> Section = getSectionForAddress(Address))
    return Section->getName();
  return std::make_error_code(std::errc::bad_address);
}

BinarySection &BinaryContext::registerSection(BinarySection *Section) {
  auto Res = Sections.insert(Section);
  (void)Res;
  assert(Res.second && "can't register the same section twice.");

  // Only register allocatable sections in the AddressToSection map.
  if (Section->isAllocatable() && Section->getAddress())
    AddressToSection.insert(std::make_pair(Section->getAddress(), Section));
  NameToSection.insert(
      std::make_pair(std::string(Section->getName()), Section));
  if (Section->hasSectionRef())
    SectionRefToBinarySection.insert(
        std::make_pair(Section->getSectionRef(), Section));

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: registering " << *Section << "\n");
  return *Section;
}

BinarySection &BinaryContext::registerSection(SectionRef Section) {
  return registerSection(new BinarySection(*this, Section));
}

BinarySection &
BinaryContext::registerSection(const Twine &SectionName,
                               const BinarySection &OriginalSection) {
  return registerSection(
      new BinarySection(*this, SectionName, OriginalSection));
}

BinarySection &
BinaryContext::registerOrUpdateSection(const Twine &Name, unsigned ELFType,
                                       unsigned ELFFlags, uint8_t *Data,
                                       uint64_t Size, unsigned Alignment) {
  auto NamedSections = getSectionByName(Name);
  if (NamedSections.begin() != NamedSections.end()) {
    assert(std::next(NamedSections.begin()) == NamedSections.end() &&
           "can only update unique sections");
    BinarySection *Section = NamedSections.begin()->second;

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: updating " << *Section << " -> ");
    const bool Flag = Section->isAllocatable();
    (void)Flag;
    Section->update(Data, Size, Alignment, ELFType, ELFFlags);
    LLVM_DEBUG(dbgs() << *Section << "\n");
    // FIXME: Fix section flags/attributes for MachO.
    if (isELF())
      assert(Flag == Section->isAllocatable() &&
             "can't change section allocation status");
    return *Section;
  }

  return registerSection(
      new BinarySection(*this, Name, Data, Size, Alignment, ELFType, ELFFlags));
}

void BinaryContext::deregisterSectionName(const BinarySection &Section) {
  auto NameRange = NameToSection.equal_range(Section.getName().str());
  while (NameRange.first != NameRange.second) {
    if (NameRange.first->second == &Section) {
      NameToSection.erase(NameRange.first);
      break;
    }
    ++NameRange.first;
  }
}

void BinaryContext::deregisterUnusedSections() {
  ErrorOr<BinarySection &> AbsSection = getUniqueSectionByName("<absolute>");
  for (auto SI = Sections.begin(); SI != Sections.end();) {
    BinarySection *Section = *SI;
    // We check getOutputData() instead of getOutputSize() because sometimes
    // zero-sized .text.cold sections are allocated.
    if (Section->hasSectionRef() || Section->getOutputData() ||
        (AbsSection && Section == &AbsSection.get())) {
      ++SI;
      continue;
    }

    LLVM_DEBUG(dbgs() << "LLVM-DEBUG: deregistering " << Section->getName()
                      << '\n';);
    deregisterSectionName(*Section);
    SI = Sections.erase(SI);
    delete Section;
  }
}

bool BinaryContext::deregisterSection(BinarySection &Section) {
  BinarySection *SectionPtr = &Section;
  auto Itr = Sections.find(SectionPtr);
  if (Itr != Sections.end()) {
    auto Range = AddressToSection.equal_range(SectionPtr->getAddress());
    while (Range.first != Range.second) {
      if (Range.first->second == SectionPtr) {
        AddressToSection.erase(Range.first);
        break;
      }
      ++Range.first;
    }

    deregisterSectionName(*SectionPtr);
    Sections.erase(Itr);
    delete SectionPtr;
    return true;
  }
  return false;
}

void BinaryContext::renameSection(BinarySection &Section,
                                  const Twine &NewName) {
  auto Itr = Sections.find(&Section);
  assert(Itr != Sections.end() && "Section must exist to be renamed.");
  Sections.erase(Itr);

  deregisterSectionName(Section);

  Section.Name = NewName.str();
  Section.setOutputName(Section.Name);

  NameToSection.insert(std::make_pair(Section.Name, &Section));

  // Reinsert with the new name.
  Sections.insert(&Section);
}

void BinaryContext::printSections(raw_ostream &OS) const {
  for (BinarySection *const &Section : Sections)
    OS << "BOLT-INFO: " << *Section << "\n";
}

BinarySection &BinaryContext::absoluteSection() {
  if (ErrorOr<BinarySection &> Section = getUniqueSectionByName("<absolute>"))
    return *Section;
  return registerOrUpdateSection("<absolute>", ELF::SHT_NULL, 0u);
}

ErrorOr<uint64_t> BinaryContext::getUnsignedValueAtAddress(uint64_t Address,
                                                           size_t Size) const {
  const ErrorOr<const BinarySection &> Section = getSectionForAddress(Address);
  if (!Section)
    return std::make_error_code(std::errc::bad_address);

  if (Section->isVirtual())
    return 0;

  DataExtractor DE(Section->getContents(), AsmInfo->isLittleEndian(),
                   AsmInfo->getCodePointerSize());
  auto ValueOffset = static_cast<uint64_t>(Address - Section->getAddress());
  return DE.getUnsigned(&ValueOffset, Size);
}

ErrorOr<uint64_t> BinaryContext::getSignedValueAtAddress(uint64_t Address,
                                                         size_t Size) const {
  const ErrorOr<const BinarySection &> Section = getSectionForAddress(Address);
  if (!Section)
    return std::make_error_code(std::errc::bad_address);

  if (Section->isVirtual())
    return 0;

  DataExtractor DE(Section->getContents(), AsmInfo->isLittleEndian(),
                   AsmInfo->getCodePointerSize());
  auto ValueOffset = static_cast<uint64_t>(Address - Section->getAddress());
  return DE.getSigned(&ValueOffset, Size);
}

void BinaryContext::addRelocation(uint64_t Address, MCSymbol *Symbol,
                                  uint64_t Type, uint64_t Addend,
                                  uint64_t Value) {
  ErrorOr<BinarySection &> Section = getSectionForAddress(Address);
  assert(Section && "cannot find section for address");
  Section->addRelocation(Address - Section->getAddress(), Symbol, Type, Addend,
                         Value);
}

void BinaryContext::addDynamicRelocation(uint64_t Address, MCSymbol *Symbol,
                                         uint64_t Type, uint64_t Addend,
                                         uint64_t Value) {
  ErrorOr<BinarySection &> Section = getSectionForAddress(Address);
  assert(Section && "cannot find section for address");
  Section->addDynamicRelocation(Address - Section->getAddress(), Symbol, Type,
                                Addend, Value);
}

bool BinaryContext::removeRelocationAt(uint64_t Address) {
  ErrorOr<BinarySection &> Section = getSectionForAddress(Address);
  assert(Section && "cannot find section for address");
  return Section->removeRelocationAt(Address - Section->getAddress());
}

const Relocation *BinaryContext::getRelocationAt(uint64_t Address) const {
  ErrorOr<const BinarySection &> Section = getSectionForAddress(Address);
  if (!Section)
    return nullptr;

  return Section->getRelocationAt(Address - Section->getAddress());
}

const Relocation *
BinaryContext::getDynamicRelocationAt(uint64_t Address) const {
  ErrorOr<const BinarySection &> Section = getSectionForAddress(Address);
  if (!Section)
    return nullptr;

  return Section->getDynamicRelocationAt(Address - Section->getAddress());
}

void BinaryContext::markAmbiguousRelocations(BinaryData &BD,
                                             const uint64_t Address) {
  auto setImmovable = [&](BinaryData &BD) {
    BinaryData *Root = BD.getAtomicRoot();
    LLVM_DEBUG(if (Root->isMoveable()) {
      dbgs() << "BOLT-DEBUG: setting " << *Root << " as immovable "
             << "due to ambiguous relocation referencing 0x"
             << Twine::utohexstr(Address) << '\n';
    });
    Root->setIsMoveable(false);
  };

  if (Address == BD.getAddress()) {
    setImmovable(BD);

    // Set previous symbol as immovable
    BinaryData *Prev = getBinaryDataContainingAddress(Address - 1);
    if (Prev && Prev->getEndAddress() == BD.getAddress())
      setImmovable(*Prev);
  }

  if (Address == BD.getEndAddress()) {
    setImmovable(BD);

    // Set next symbol as immovable
    BinaryData *Next = getBinaryDataContainingAddress(BD.getEndAddress());
    if (Next && Next->getAddress() == BD.getEndAddress())
      setImmovable(*Next);
  }
}

BinaryFunction *BinaryContext::getFunctionForSymbol(const MCSymbol *Symbol,
                                                    uint64_t *EntryDesc) {
  std::shared_lock<llvm::sys::RWMutex> Lock(SymbolToFunctionMapMutex);
  auto BFI = SymbolToFunctionMap.find(Symbol);
  if (BFI == SymbolToFunctionMap.end())
    return nullptr;

  BinaryFunction *BF = BFI->second;
  if (EntryDesc)
    *EntryDesc = BF->getEntryIDForSymbol(Symbol);

  return BF;
}

void BinaryContext::exitWithBugReport(StringRef Message,
                                      const BinaryFunction &Function) const {
  errs() << "=======================================\n";
  errs() << "BOLT is unable to proceed because it couldn't properly understand "
            "this function.\n";
  errs() << "If you are running the most recent version of BOLT, you may "
            "want to "
            "report this and paste this dump.\nPlease check that there is no "
            "sensitive contents being shared in this dump.\n";
  errs() << "\nOffending function: " << Function.getPrintName() << "\n\n";
  ScopedPrinter SP(errs());
  SP.printBinaryBlock("Function contents", *Function.getData());
  errs() << "\n";
  Function.dump();
  errs() << "ERROR: " << Message;
  errs() << "\n=======================================\n";
  exit(1);
}

BinaryFunction *
BinaryContext::createInjectedBinaryFunction(const std::string &Name,
                                            bool IsSimple) {
  InjectedBinaryFunctions.push_back(new BinaryFunction(Name, *this, IsSimple));
  BinaryFunction *BF = InjectedBinaryFunctions.back();
  setSymbolToFunctionMap(BF->getSymbol(), BF);
  BF->CurrentState = BinaryFunction::State::CFG;
  return BF;
}

std::pair<size_t, size_t>
BinaryContext::calculateEmittedSize(BinaryFunction &BF, bool FixBranches) {
  // Adjust branch instruction to match the current layout.
  if (FixBranches)
    BF.fixBranches();

  // Create local MC context to isolate the effect of ephemeral code emission.
  IndependentCodeEmitter MCEInstance = createIndependentMCCodeEmitter();
  MCContext *LocalCtx = MCEInstance.LocalCtx.get();
  MCAsmBackend *MAB =
      TheTarget->createMCAsmBackend(*STI, *MRI, MCTargetOptions());

  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);

  std::unique_ptr<MCObjectWriter> OW = MAB->createObjectWriter(VecOS);
  std::unique_ptr<MCStreamer> Streamer(TheTarget->createMCObjectStreamer(
      *TheTriple, *LocalCtx, std::unique_ptr<MCAsmBackend>(MAB), std::move(OW),
      std::unique_ptr<MCCodeEmitter>(MCEInstance.MCE.release()), *STI,
      /*RelaxAll=*/false,
      /*IncrementalLinkerCompatible=*/false,
      /*DWARFMustBeAtTheEnd=*/false));

  Streamer->initSections(false, *STI);

  MCSection *Section = MCEInstance.LocalMOFI->getTextSection();
  Section->setHasInstructions(true);

  // Create symbols in the LocalCtx so that they get destroyed with it.
  MCSymbol *StartLabel = LocalCtx->createTempSymbol();
  MCSymbol *EndLabel = LocalCtx->createTempSymbol();

  Streamer->switchSection(Section);
  Streamer->emitLabel(StartLabel);
  emitFunctionBody(*Streamer, BF, BF.getLayout().getMainFragment(),
                   /*EmitCodeOnly=*/true);
  Streamer->emitLabel(EndLabel);

  using LabelRange = std::pair<const MCSymbol *, const MCSymbol *>;
  SmallVector<LabelRange> SplitLabels;
  for (FunctionFragment &FF : BF.getLayout().getSplitFragments()) {
    MCSymbol *const SplitStartLabel = LocalCtx->createTempSymbol();
    MCSymbol *const SplitEndLabel = LocalCtx->createTempSymbol();
    SplitLabels.emplace_back(SplitStartLabel, SplitEndLabel);

    MCSectionELF *const SplitSection = LocalCtx->getELFSection(
        BF.getCodeSectionName(FF.getFragmentNum()), ELF::SHT_PROGBITS,
        ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);
    SplitSection->setHasInstructions(true);
    Streamer->switchSection(SplitSection);

    Streamer->emitLabel(SplitStartLabel);
    emitFunctionBody(*Streamer, BF, FF, /*EmitCodeOnly=*/true);
    Streamer->emitLabel(SplitEndLabel);
    // To avoid calling MCObjectStreamer::flushPendingLabels() which is
    // private
    Streamer->emitBytes(StringRef(""));
    Streamer->switchSection(Section);
  }

  // To avoid calling MCObjectStreamer::flushPendingLabels() which is private or
  // MCStreamer::Finish(), which does more than we want
  Streamer->emitBytes(StringRef(""));

  MCAssembler &Assembler =
      static_cast<MCObjectStreamer *>(Streamer.get())->getAssembler();
  MCAsmLayout Layout(Assembler);
  Assembler.layout(Layout);

  const uint64_t HotSize =
      Layout.getSymbolOffset(*EndLabel) - Layout.getSymbolOffset(*StartLabel);
  const uint64_t ColdSize =
      std::accumulate(SplitLabels.begin(), SplitLabels.end(), 0ULL,
                      [&](const uint64_t Accu, const LabelRange &Labels) {
                        return Accu + Layout.getSymbolOffset(*Labels.second) -
                               Layout.getSymbolOffset(*Labels.first);
                      });

  // Clean-up the effect of the code emission.
  for (const MCSymbol &Symbol : Assembler.symbols()) {
    MCSymbol *MutableSymbol = const_cast<MCSymbol *>(&Symbol);
    MutableSymbol->setUndefined();
    MutableSymbol->setIsRegistered(false);
  }

  return std::make_pair(HotSize, ColdSize);
}

bool BinaryContext::validateInstructionEncoding(
    ArrayRef<uint8_t> InputSequence) const {
  MCInst Inst;
  uint64_t InstSize;
  DisAsm->getInstruction(Inst, InstSize, InputSequence, 0, nulls());
  assert(InstSize == InputSequence.size() &&
         "Disassembled instruction size does not match the sequence.");

  SmallString<256> Code;
  SmallVector<MCFixup, 4> Fixups;

  MCE->encodeInstruction(Inst, Code, Fixups, *STI);
  auto OutputSequence = ArrayRef<uint8_t>((uint8_t *)Code.data(), Code.size());
  if (InputSequence != OutputSequence) {
    if (opts::Verbosity > 1) {
      errs() << "BOLT-WARNING: mismatched encoding detected\n"
             << "      input: " << InputSequence << '\n'
             << "     output: " << OutputSequence << '\n';
    }
    return false;
  }

  return true;
}

uint64_t BinaryContext::getHotThreshold() const {
  static uint64_t Threshold = 0;
  if (Threshold == 0) {
    Threshold = std::max(
        (uint64_t)opts::ExecutionCountThreshold,
        NumProfiledFuncs ? SumExecutionCount / (2 * NumProfiledFuncs) : 1);
  }
  return Threshold;
}

BinaryFunction *BinaryContext::getBinaryFunctionContainingAddress(
    uint64_t Address, bool CheckPastEnd, bool UseMaxSize) {
  auto FI = BinaryFunctions.upper_bound(Address);
  if (FI == BinaryFunctions.begin())
    return nullptr;
  --FI;

  const uint64_t UsedSize =
      UseMaxSize ? FI->second.getMaxSize() : FI->second.getSize();

  if (Address >= FI->first + UsedSize + (CheckPastEnd ? 1 : 0))
    return nullptr;

  return &FI->second;
}

BinaryFunction *BinaryContext::getBinaryFunctionAtAddress(uint64_t Address) {
  // First, try to find a function starting at the given address. If the
  // function was folded, this will get us the original folded function if it
  // wasn't removed from the list, e.g. in non-relocation mode.
  auto BFI = BinaryFunctions.find(Address);
  if (BFI != BinaryFunctions.end())
    return &BFI->second;

  // We might have folded the function matching the object at the given
  // address. In such case, we look for a function matching the symbol
  // registered at the original address. The new function (the one that the
  // original was folded into) will hold the symbol.
  if (const BinaryData *BD = getBinaryDataAtAddress(Address)) {
    uint64_t EntryID = 0;
    BinaryFunction *BF = getFunctionForSymbol(BD->getSymbol(), &EntryID);
    if (BF && EntryID == 0)
      return BF;
  }
  return nullptr;
}

DebugAddressRangesVector BinaryContext::translateModuleAddressRanges(
    const DWARFAddressRangesVector &InputRanges) const {
  DebugAddressRangesVector OutputRanges;

  for (const DWARFAddressRange Range : InputRanges) {
    auto BFI = BinaryFunctions.lower_bound(Range.LowPC);
    while (BFI != BinaryFunctions.end()) {
      const BinaryFunction &Function = BFI->second;
      if (Function.getAddress() >= Range.HighPC)
        break;
      const DebugAddressRangesVector FunctionRanges =
          Function.getOutputAddressRanges();
      llvm::move(FunctionRanges, std::back_inserter(OutputRanges));
      std::advance(BFI, 1);
    }
  }

  return OutputRanges;
}

} // namespace bolt
} // namespace llvm
