//===- bolt/Rewrite/StrippedBinary.cpp - Stripped ELF helpers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/Object/ELFObjectFile.h"

#include <algorithm>
#include <optional>

namespace llvm {
namespace bolt {
namespace {

/// Decode the 4-byte instruction at Address. Reject unaligned addresses and
/// addresses outside the file-backed contents of an executable section.
bool decodeAt(BinaryContext &BC, uint64_t Address, MCInst &Inst) {
  auto Section = BC.getSectionForAddress(Address);
  if (!Section || !Section->isText() || Address % 4)
    return false;
  const StringRef Contents = Section->getContents();
  const uint64_t Offset = Address - Section->getAddress();
  if (Offset > Contents.size() || Contents.size() - Offset < 4)
    return false;
  const auto *Bytes = reinterpret_cast<const uint8_t *>(Contents.data());
  uint64_t Size = 0;
  return BC.DisAsm->getInstruction(
             Inst, Size, ArrayRef<uint8_t>(Bytes + Offset, 4), Address,
             nulls()) == MCDisassembler::Success &&
         Size == 4;
}

/// Return true for a direct unconditional branch, excluding indirect
/// branches and returns.
bool isDirectBranch(BinaryContext &BC, const MCInst &Inst) {
  return BC.MIB->isUnconditionalBranch(Inst) &&
         !BC.MIB->isIndirectBranch(Inst) && !BC.MIB->isReturn(Inst);
}

/// Return true when Inst can prefix glibc's optional main wrapper. The prefix
/// is NOP without branch target identification and BTI C when it is enabled.
/// LLVM represents them as HINT instructions with immediates 0 and 34.
bool isAArch64WrapperPrefix(BinaryContext &BC, const MCInst &Inst) {
  if (BC.MIB->isNoop(Inst))
    return true;
  return BC.MII->getName(Inst.getOpcode()) == "HINT" &&
         Inst.getNumOperands() != 0 && Inst.getOperand(0).isImm() &&
         Inst.getOperand(0).getImm() == 34;
}

/// Match a direct, non-tail call in stripped AArch64 entry code. For an input
/// with a dynamic section, require the target name Name or Name@PLT. For a
/// static input, accept any registered direct target; the caller verifies its
/// position in the ordered glibc startup sequence.
bool matchesAArch64EntryCall(BinaryContext &BC, const MCInst &Inst,
                             uint64_t Address, StringRef Name,
                             bool HasDynamicSection) {
  uint64_t Target = 0;
  if (!BC.MIB->isCall(Inst) || BC.MIB->isTailCall(Inst) ||
      !BC.MIB->evaluateBranch(Inst, Address, 4, Target))
    return false;
  const BinaryFunction *BF = BC.getBinaryFunctionAtAddress(Target);
  if (!BF)
    return false;
  return !HasDynamicSection || BF->hasName(Name.str()) ||
         BF->hasName(Name.str() + "@PLT");
}

/// Extent of a recognized _start sequence and the optional offset at which
/// its adjacent main wrapper can also be entered.
struct AArch64EntryMatch {
  uint64_t Size;
  std::optional<uint64_t> SecondaryEntryOffset;
};

/// Scan [Entry, End) for the AArch64 glibc startup sequence. It consists of a
/// call to __libc_start_main immediately followed by a call to abort. After
/// those calls, NOP or BTI C prefixes followed by a direct branch to a known
/// function identify an adjacent main wrapper. Report the first prefix, or
/// the branch itself when there is no prefix, as a secondary entry to _start.
std::optional<AArch64EntryMatch>
matchAArch64EntryPattern(BinaryContext &BC, uint64_t Entry, uint64_t End,
                         bool HasDynamicSection) {
  enum class MatchState { StartMain, Abort, Wrapper };
  MatchState State = MatchState::StartMain;
  uint64_t Size = 0;
  std::optional<uint64_t> WrapperPrefixAddress;

  for (uint64_t Address = Entry; Address < End && End - Address >= 4;
       Address += 4) {
    MCInst Inst;
    if (!decodeAt(BC, Address, Inst))
      break;

    if (State == MatchState::StartMain) {
      if (matchesAArch64EntryCall(BC, Inst, Address, "__libc_start_main",
                                  HasDynamicSection)) {
        State = MatchState::Abort;
        continue;
      }
      if (BC.MIB->isCall(Inst) || BC.MIB->isBranch(Inst) ||
          BC.MIB->isReturn(Inst))
        break;
      continue;
    }

    if (State == MatchState::Abort) {
      if (!matchesAArch64EntryCall(BC, Inst, Address, "abort",
                                   HasDynamicSection))
        break;
      Size = Address + 4 - Entry;
      State = MatchState::Wrapper;
      continue;
    }

    if (isAArch64WrapperPrefix(BC, Inst)) {
      if (!WrapperPrefixAddress)
        WrapperPrefixAddress = Address;
      continue;
    }
    if (!isDirectBranch(BC, Inst))
      return AArch64EntryMatch{Size, std::nullopt};

    uint64_t Target = 0;
    if (!BC.MIB->evaluateBranch(Inst, Address, 4, Target) ||
        !BC.getBinaryFunctionAtAddress(Target))
      return AArch64EntryMatch{Size, std::nullopt};
    const uint64_t WrapperAddress = WrapperPrefixAddress.value_or(Address);
    return AArch64EntryMatch{Address + 4 - Entry, WrapperAddress - Entry};
  }

  if (State == MatchState::Wrapper)
    return AArch64EntryMatch{Size, std::nullopt};
  return std::nullopt;
}

/// Recover AArch64 _start from the ELF entry address. Limit the scan to the
/// existing entry function, the next known function, or file-backed executable
/// contents. If _start is missing, create it only after matching the glibc
/// startup sequence. If it already exists, keep its bounds and use a match only
/// to add the main-wrapper secondary entry.
Error discoverAArch64Entry(BinaryContext &BC, uint64_t Entry) {
  assert(BC.isAArch64() && "AArch64 entry discovery requires AArch64 input");
  if (!Entry)
    return Error::success(); // Shared objects commonly have no process entry.
  BinaryFunction *Start = BC.getBinaryFunctionAtAddress(Entry);
  auto Section = BC.getSectionForAddress(Entry);
  if (!Section || !Section->isText() || Entry % 4)
    return createFatalBOLTError("invalid stripped AArch64 ELF entry point");
  if (!Start && BC.getBinaryFunctionContainingAddress(Entry))
    return createFatalBOLTError("ELF entry overlaps a discovered function");

  uint64_t End = Section->getAddress() + Section->getContents().size();
  if (Entry >= End)
    return createFatalBOLTError(
        "stripped AArch64 ELF entry has no file-backed instruction");
  if (Start) {
    // FDE discovery already supplied the entry function's exact boundary.
    End = std::min(End, Entry + Start->getSize());
  } else {
    // The next FDE-backed function bounds an unregistered _start candidate. If
    // none exists, cap the scan within the file-backed executable contents.
    auto Next = BC.getBinaryFunctions().upper_bound(Entry);
    if (Next != BC.getBinaryFunctions().end())
      End = std::min(End, Next->first);
    else
      End = Entry + std::min<uint64_t>(End - Entry, 4096);
  }

  const bool HasDynamicSection = bool(BC.getUniqueSectionByName(".dynamic"));
  const std::optional<AArch64EntryMatch> Match =
      matchAArch64EntryPattern(BC, Entry, End, HasDynamicSection);
  if (Match) {
    // Register a new function only after the bounded pattern matches. An
    // existing FDE-backed _start keeps its boundary and gains only the
    // secondary entry identified by the pattern.
    if (!Start) {
      Start = BC.createBinaryFunction("_start", *Section, Entry, Match->Size);
      Start->setMaxSize(Match->Size);
      BC.outs() << "BOLT-INFO: recovered stripped entry at 0x"
                << Twine::utohexstr(Entry) << '\n';
    }
    if (Match->SecondaryEntryOffset)
      Start->addEntryPointAtOffset(*Match->SecondaryEntryOffset);
    return Error::success();
  }
  // An existing FDE-backed entry already has a function boundary. An
  // unregistered entry must match the startup pattern before it is added.
  if (Start)
    return Error::success();
  return createFatalBOLTError(
      "cannot recover stripped entry: expected a bounded "
      "glibc __libc_start_main/abort sequence");
}

/// Find linker-generated Cortex-A53 erratum 843419 veneers reached from known
/// functions. The source ADRP must occupy one of the last two instruction slots
/// in a 4 KiB page: its address has low 12 bits 0xFF8 or 0xFFC. From that ADRP,
/// look for a nearby direct branch to an unregistered two-instruction helper
/// that performs a load or store using the ADRP destination register and then
/// branches back to the instruction after the original branch. Register each
/// helper before normal function disassembly, which must not modify the
/// function map.
///
/// See the Arm Cortex-A53 MPCore Software Developers Errata Notice, erratum
/// 843419.
void discoverErratumVeneers(BinaryContext &BC) {
  SmallVector<BinaryFunction *, 16> Functions;
  for (auto &BFI : BC.getBinaryFunctions())
    Functions.push_back(&BFI.second);
  for (BinaryFunction *BF : Functions) {
    for (uint64_t Offset = 0; Offset < BF->getSize(); Offset += 4) {
      const uint64_t Address = BF->getAddress() + Offset;
      if ((Address & 4095) != 4088 && (Address & 4095) != 4092)
        continue;
      MCInst ADRP;
      if (!decodeAt(BC, Address, ADRP) || !BC.MIB->isADRP(ADRP))
        continue;
      const unsigned Reg = ADRP.getOperand(0).getReg();
      for (unsigned Distance = 4;
           Distance <= 16 && Offset + Distance < BF->getSize(); Distance += 4) {
        const uint64_t BranchAddress = Address + Distance;
        MCInst Branch;
        if (!decodeAt(BC, BranchAddress, Branch))
          break;
        if (!isDirectBranch(BC, Branch)) {
          if (BC.MIB->isBranch(Branch) || BC.MIB->isCall(Branch) ||
              BC.MIB->isReturn(Branch) ||
              (BC.MIB->hasDefOfPhysReg(Branch, Reg) &&
               !BC.MIB->hasUseOfPhysReg(Branch, Reg)))
            break;
          continue;
        }
        uint64_t Target = 0, ReturnAddress = 0;
        if (!BC.MIB->evaluateBranch(Branch, BranchAddress, 4, Target) ||
            BC.getBinaryFunctionContainingAddress(Target) ||
            BC.getBinaryFunctionContainingAddress(Target + 7))
          break;
        auto Section = BC.getSectionForAddress(Target);
        MCInst Memory, Return;
        if (!Section || !decodeAt(BC, Target, Memory) ||
            !decodeAt(BC, Target + 4, Return) ||
            (!BC.MIB->mayLoad(Memory) && !BC.MIB->mayStore(Memory)) ||
            !BC.MIB->hasUseOfPhysReg(Memory, Reg) ||
            !isDirectBranch(BC, Return) ||
            !BC.MIB->evaluateBranch(Return, Target + 4, 4, ReturnAddress) ||
            ReturnAddress != BranchAddress + 4 ||
            ReturnAddress >= BF->getAddress() + BF->getSize())
          break;
        BinaryFunction *Veneer = BC.createBinaryFunction(
            "__BOLT_e843419_" + Twine::utohexstr(Target).str(), *Section,
            Target, 8);
        Veneer->setMaxSize(8);
        break;
      }
    }
  }
}

} // namespace

/// Recover functions that FDE discovery can miss in a stripped ELF file.
/// Currently this discovers AArch64 _start and erratum 843419 veneers.
Error RewriteInstance::discoverStrippedFunctions() {
  const auto *ELF = dyn_cast<object::ELF64LEObjectFile>(InputFile);
  if (!ELF)
    return createFatalBOLTError(
        "stripped recovery requires little-endian ELF64");
  if (BC->isAArch64()) {
    if (Error E =
            discoverAArch64Entry(*BC, ELF->getELFFile().getHeader().e_entry))
      return E;
    discoverErratumVeneers(*BC);
  }
  return Error::success();
}

} // namespace bolt
} // namespace llvm
