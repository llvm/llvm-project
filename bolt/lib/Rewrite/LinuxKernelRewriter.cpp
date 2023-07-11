//===- bolt/Rewrite/LinuxKernelRewriter.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for updating Linux Kernel metadata.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Rewrite/MetadataRewriter.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"

using namespace llvm;
using namespace bolt;

namespace {
class LinuxKernelRewriter final : public MetadataRewriter {

  /// Linux Kernel special sections point to a specific instruction in many
  /// cases. Unlike SDTMarkerInfo, these markers can come from different
  /// sections.
  struct LKInstructionMarkerInfo {
    uint64_t SectionOffset;
    int32_t PCRelativeOffset;
    bool IsPCRelative;
    StringRef SectionName;
  };

  /// Map linux kernel program locations/instructions to their pointers in
  /// special linux kernel sections
  std::unordered_map<uint64_t, std::vector<LKInstructionMarkerInfo>> LKMarkers;

  /// Insert an LKMarker for a given code pointer \p PC from a non-code section
  /// \p SectionName.
  void insertLKMarker(uint64_t PC, uint64_t SectionOffset,
                      int32_t PCRelativeOffset, bool IsPCRelative,
                      StringRef SectionName);

  /// Process linux kernel special sections and their relocations.
  void processLKSections();

  /// Process special linux kernel section, __ex_table.
  void processLKExTable();

  /// Process special linux kernel section, .pci_fixup.
  void processLKPCIFixup();

  /// Process __ksymtab and __ksymtab_gpl.
  void processLKKSymtab(bool IsGPL = false);

  /// Process special linux kernel section, __bug_table.
  void processLKBugTable();

  /// Process special linux kernel section, .smp_locks.
  void processLKSMPLocks();

  /// Update LKMarkers' locations for the output binary.
  void updateLKMarkers();

  /// Mark instructions referenced by kernel metadata.
  Error markInstructions();

public:
  LinuxKernelRewriter(BinaryContext &BC)
      : MetadataRewriter("linux-kernel-rewriter", BC) {}

  Error preCFGInitializer() override {
    if (opts::LinuxKernelMode) {
      processLKSections();
      if (Error E = markInstructions())
        return E;
    }

    return Error::success();
  }

  Error postEmitFinalizer() override {
    updateLKMarkers();
    return Error::success();
  }
};

Error LinuxKernelRewriter::markInstructions() {
  for (const uint64_t PC : llvm::make_first_range(LKMarkers)) {
    BinaryFunction *BF = BC.getBinaryFunctionContainingAddress(PC);

    if (!BF || !BC.shouldEmit(*BF))
      continue;

    const uint64_t Offset = PC - BF->getAddress();
    MCInst *Inst = BF->getInstructionAtOffset(Offset);
    if (!Inst)
      return createStringError(errc::executable_format_error,
                               "no instruction matches kernel marker offset");

    BC.MIB->setOffset(*Inst, static_cast<uint32_t>(Offset));

    BF->setHasSDTMarker(true);
  }

  return Error::success();
}

void LinuxKernelRewriter::insertLKMarker(uint64_t PC, uint64_t SectionOffset,
                                         int32_t PCRelativeOffset,
                                         bool IsPCRelative,
                                         StringRef SectionName) {
  LKMarkers[PC].emplace_back(LKInstructionMarkerInfo{
      SectionOffset, PCRelativeOffset, IsPCRelative, SectionName});
}

void LinuxKernelRewriter::processLKSections() {
  assert(opts::LinuxKernelMode &&
         "process Linux Kernel special sections and their relocations only in "
         "linux kernel mode.\n");

  processLKExTable();
  processLKPCIFixup();
  processLKKSymtab();
  processLKKSymtab(true);
  processLKBugTable();
  processLKSMPLocks();
}

/// Process __ex_table section of Linux Kernel.
/// This section contains information regarding kernel level exception
/// handling (https://www.kernel.org/doc/html/latest/x86/exception-tables.html).
/// More documentation is in arch/x86/include/asm/extable.h.
///
/// The section is the list of the following structures:
///
///   struct exception_table_entry {
///     int insn;
///     int fixup;
///     int handler;
///   };
///
void LinuxKernelRewriter::processLKExTable() {
  ErrorOr<BinarySection &> SectionOrError =
      BC.getUniqueSectionByName("__ex_table");
  if (!SectionOrError)
    return;

  const uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 12) == 0 &&
         "The size of the __ex_table section should be a multiple of 12");
  for (uint64_t I = 0; I < SectionSize; I += 4) {
    const uint64_t EntryAddress = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC.getSignedValueAtAddress(EntryAddress, 4);
    assert(Offset && "failed reading PC-relative offset for __ex_table");
    int32_t SignedOffset = *Offset;
    const uint64_t RefAddress = EntryAddress + SignedOffset;

    BinaryFunction *ContainingBF =
        BC.getBinaryFunctionContainingAddress(RefAddress);
    if (!ContainingBF)
      continue;

    MCSymbol *ReferencedSymbol = ContainingBF->getSymbol();
    const uint64_t FunctionOffset = RefAddress - ContainingBF->getAddress();
    switch (I % 12) {
    default:
      llvm_unreachable("bad alignment of __ex_table");
      break;
    case 0:
      // insn
      insertLKMarker(RefAddress, I, SignedOffset, true, "__ex_table");
      break;
    case 4:
      // fixup
      if (FunctionOffset)
        ReferencedSymbol = ContainingBF->addEntryPointAtOffset(FunctionOffset);
      BC.addRelocation(EntryAddress, ReferencedSymbol, Relocation::getPC32(), 0,
                       *Offset);
      break;
    case 8:
      // handler
      assert(!FunctionOffset &&
             "__ex_table handler entry should point to function start");
      BC.addRelocation(EntryAddress, ReferencedSymbol, Relocation::getPC32(), 0,
                       *Offset);
      break;
    }
  }
}

/// Process .pci_fixup section of Linux Kernel.
/// This section contains a list of entries for different PCI devices and their
/// corresponding hook handler (code pointer where the fixup
/// code resides, usually on x86_64 it is an entry PC relative 32 bit offset).
/// Documentation is in include/linux/pci.h.
void LinuxKernelRewriter::processLKPCIFixup() {
  ErrorOr<BinarySection &> SectionOrError =
      BC.getUniqueSectionByName(".pci_fixup");
  assert(SectionOrError &&
         ".pci_fixup section not found in Linux Kernel binary");
  const uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 16) == 0 && ".pci_fixup size is not a multiple of 16");

  for (uint64_t I = 12; I + 4 <= SectionSize; I += 16) {
    const uint64_t PC = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC.getSignedValueAtAddress(PC, 4);
    assert(Offset && "cannot read value from .pci_fixup");
    const int32_t SignedOffset = *Offset;
    const uint64_t HookupAddress = PC + SignedOffset;
    BinaryFunction *HookupFunction =
        BC.getBinaryFunctionAtAddress(HookupAddress);
    assert(HookupFunction && "expected function for entry in .pci_fixup");
    BC.addRelocation(PC, HookupFunction->getSymbol(), Relocation::getPC32(), 0,
                     *Offset);
  }
}

/// Process __ksymtab[_gpl] sections of Linux Kernel.
/// This section lists all the vmlinux symbols that kernel modules can access.
///
/// All the entries are 4 bytes each and hence we can read them by one by one
/// and ignore the ones that are not pointing to the .text section. All pointers
/// are PC relative offsets. Always, points to the beginning of the function.
void LinuxKernelRewriter::processLKKSymtab(bool IsGPL) {
  StringRef SectionName = "__ksymtab";
  if (IsGPL)
    SectionName = "__ksymtab_gpl";
  ErrorOr<BinarySection &> SectionOrError =
      BC.getUniqueSectionByName(SectionName);
  assert(SectionOrError &&
         "__ksymtab[_gpl] section not found in Linux Kernel binary");
  const uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 4) == 0 &&
         "The size of the __ksymtab[_gpl] section should be a multiple of 4");

  for (uint64_t I = 0; I < SectionSize; I += 4) {
    const uint64_t EntryAddress = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC.getSignedValueAtAddress(EntryAddress, 4);
    assert(Offset && "Reading valid PC-relative offset for a ksymtab entry");
    const int32_t SignedOffset = *Offset;
    const uint64_t RefAddress = EntryAddress + SignedOffset;
    BinaryFunction *BF = BC.getBinaryFunctionAtAddress(RefAddress);
    if (!BF)
      continue;

    BC.addRelocation(EntryAddress, BF->getSymbol(), Relocation::getPC32(), 0,
                     *Offset);
  }
}

/// Process __bug_table section.
/// This section contains information useful for kernel debugging.
/// Each entry in the section is a struct bug_entry that contains a pointer to
/// the ud2 instruction corresponding to the bug, corresponding file name (both
/// pointers use PC relative offset addressing), line number, and flags.
/// The definition of the struct bug_entry can be found in
/// `include/asm-generic/bug.h`
void LinuxKernelRewriter::processLKBugTable() {
  ErrorOr<BinarySection &> SectionOrError =
      BC.getUniqueSectionByName("__bug_table");
  if (!SectionOrError)
    return;

  const uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 12) == 0 &&
         "The size of the __bug_table section should be a multiple of 12");
  for (uint64_t I = 0; I < SectionSize; I += 12) {
    const uint64_t EntryAddress = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC.getSignedValueAtAddress(EntryAddress, 4);
    assert(Offset &&
           "Reading valid PC-relative offset for a __bug_table entry");
    const int32_t SignedOffset = *Offset;
    const uint64_t RefAddress = EntryAddress + SignedOffset;
    assert(BC.getBinaryFunctionContainingAddress(RefAddress) &&
           "__bug_table entries should point to a function");

    insertLKMarker(RefAddress, I, SignedOffset, true, "__bug_table");
  }
}

/// .smp_locks section contains PC-relative references to instructions with LOCK
/// prefix. The prefix can be converted to NOP at boot time on non-SMP systems.
void LinuxKernelRewriter::processLKSMPLocks() {
  ErrorOr<BinarySection &> SectionOrError =
      BC.getUniqueSectionByName(".smp_locks");
  if (!SectionOrError)
    return;

  uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 4) == 0 &&
         "The size of the .smp_locks section should be a multiple of 4");

  for (uint64_t I = 0; I < SectionSize; I += 4) {
    const uint64_t EntryAddress = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC.getSignedValueAtAddress(EntryAddress, 4);
    assert(Offset && "Reading valid PC-relative offset for a .smp_locks entry");
    int32_t SignedOffset = *Offset;
    uint64_t RefAddress = EntryAddress + SignedOffset;

    BinaryFunction *ContainingBF =
        BC.getBinaryFunctionContainingAddress(RefAddress);
    if (!ContainingBF)
      continue;

    insertLKMarker(RefAddress, I, SignedOffset, true, ".smp_locks");
  }
}

void LinuxKernelRewriter::updateLKMarkers() {
  if (LKMarkers.size() == 0)
    return;

  std::unordered_map<std::string, uint64_t> PatchCounts;
  for (std::pair<const uint64_t, std::vector<LKInstructionMarkerInfo>>
           &LKMarkerInfoKV : LKMarkers) {
    const uint64_t OriginalAddress = LKMarkerInfoKV.first;
    const BinaryFunction *BF =
        BC.getBinaryFunctionContainingAddress(OriginalAddress, false, true);
    if (!BF)
      continue;

    uint64_t NewAddress = BF->translateInputToOutputAddress(OriginalAddress);
    if (NewAddress == 0)
      continue;

    // Apply base address.
    if (OriginalAddress >= 0xffffffff00000000 && NewAddress < 0xffffffff)
      NewAddress = NewAddress + 0xffffffff00000000;

    if (OriginalAddress == NewAddress)
      continue;

    for (LKInstructionMarkerInfo &LKMarkerInfo : LKMarkerInfoKV.second) {
      StringRef SectionName = LKMarkerInfo.SectionName;
      SimpleBinaryPatcher *LKPatcher;
      ErrorOr<BinarySection &> BSec = BC.getUniqueSectionByName(SectionName);
      assert(BSec && "missing section info for kernel section");
      if (!BSec->getPatcher())
        BSec->registerPatcher(std::make_unique<SimpleBinaryPatcher>());
      LKPatcher = static_cast<SimpleBinaryPatcher *>(BSec->getPatcher());
      PatchCounts[std::string(SectionName)]++;
      if (LKMarkerInfo.IsPCRelative)
        LKPatcher->addLE32Patch(LKMarkerInfo.SectionOffset,
                                NewAddress - OriginalAddress +
                                    LKMarkerInfo.PCRelativeOffset);
      else
        LKPatcher->addLE64Patch(LKMarkerInfo.SectionOffset, NewAddress);
    }
  }
  outs() << "BOLT-INFO: patching linux kernel sections. Total patches per "
            "section are as follows:\n";
  for (const std::pair<const std::string, uint64_t> &KV : PatchCounts)
    outs() << "  Section: " << KV.first << ", patch-counts: " << KV.second
           << '\n';
}
} // namespace

std::unique_ptr<MetadataRewriter>
llvm::bolt::createLinuxKernelRewriter(BinaryContext &BC) {
  return std::make_unique<LinuxKernelRewriter>(BC);
}
