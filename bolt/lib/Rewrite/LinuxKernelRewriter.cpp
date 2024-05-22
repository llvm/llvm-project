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

namespace opts {

static cl::opt<bool>
    PrintORC("print-orc",
             cl::desc("print ORC unwind information for instructions"),
             cl::init(true), cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool>
    DumpORC("dump-orc", cl::desc("dump raw ORC unwind information (sorted)"),
            cl::init(false), cl::Hidden, cl::cat(BoltCategory));

} // namespace opts

/// Linux Kernel supports stack unwinding using ORC (oops rewind capability).
/// ORC state at every IP can be described by the following data structure.
struct ORCState {
  int16_t SPOffset;
  int16_t BPOffset;
  int16_t Info;

  bool operator==(const ORCState &Other) const {
    return SPOffset == Other.SPOffset && BPOffset == Other.BPOffset &&
           Info == Other.Info;
  }

  bool operator!=(const ORCState &Other) const { return !(*this == Other); }
};

/// Basic printer for ORC entry. It does not provide the same level of
/// information as objtool (for now).
inline raw_ostream &operator<<(raw_ostream &OS, const ORCState &E) {
  if (opts::PrintORC)
    OS << format("{sp: %d, bp: %d, info: 0x%x}", E.SPOffset, E.BPOffset,
                 E.Info);
  return OS;
}

namespace {

/// Section terminator ORC entry.
static ORCState NullORC = {0, 0, 0};

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

  /// Linux ORC sections.
  ErrorOr<BinarySection &> ORCUnwindSection = std::errc::bad_address;
  ErrorOr<BinarySection &> ORCUnwindIPSection = std::errc::bad_address;

  /// Size of entries in ORC sections.
  static constexpr size_t ORC_UNWIND_ENTRY_SIZE = 6;
  static constexpr size_t ORC_UNWIND_IP_ENTRY_SIZE = 4;

  struct ORCListEntry {
    uint64_t IP;        /// Instruction address.
    BinaryFunction *BF; /// Binary function corresponding to the entry.
    ORCState ORC;       /// Stack unwind info in ORC format.

    bool operator<(const ORCListEntry &Other) const {
      if (IP < Other.IP)
        return 1;
      if (IP > Other.IP)
        return 0;
      return ORC == NullORC;
    }
  };

  using ORCListType = std::vector<ORCListEntry>;
  ORCListType ORCEntries;

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

  /// Read ORC unwind information and annotate instructions.
  Error readORCTables();

  /// Update ORC for functions once CFG is constructed.
  Error processORCPostCFG();

  /// Update ORC data in the binary.
  Error rewriteORCTables();

  /// Mark instructions referenced by kernel metadata.
  Error markInstructions();

public:
  LinuxKernelRewriter(BinaryContext &BC)
      : MetadataRewriter("linux-kernel-rewriter", BC) {}

  Error preCFGInitializer() override {
    processLKSections();
    if (Error E = markInstructions())
      return E;

    if (Error E = readORCTables())
      return E;

    return Error::success();
  }

  Error postCFGInitializer() override {
    if (Error E = processORCPostCFG())
      return E;

    return Error::success();
  }

  Error postEmitFinalizer() override {
    updateLKMarkers();

    if (Error E = rewriteORCTables())
      return E;

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

Error LinuxKernelRewriter::readORCTables() {
  // NOTE: we should ignore relocations for orc tables as the tables are sorted
  // post-link time and relocations are not updated.
  ORCUnwindSection = BC.getUniqueSectionByName(".orc_unwind");
  ORCUnwindIPSection = BC.getUniqueSectionByName(".orc_unwind_ip");

  if (!ORCUnwindSection && !ORCUnwindIPSection)
    return Error::success();

  if (!ORCUnwindSection || !ORCUnwindIPSection)
    return createStringError(errc::executable_format_error,
                             "missing ORC section");

  const uint64_t NumEntries =
      ORCUnwindIPSection->getSize() / ORC_UNWIND_IP_ENTRY_SIZE;
  if (ORCUnwindSection->getSize() != NumEntries * ORC_UNWIND_ENTRY_SIZE ||
      ORCUnwindIPSection->getSize() != NumEntries * ORC_UNWIND_IP_ENTRY_SIZE)
    return createStringError(errc::executable_format_error,
                             "ORC entries number mismatch detected");

  const uint64_t IPSectionAddress = ORCUnwindIPSection->getAddress();
  DataExtractor OrcDE = DataExtractor(ORCUnwindSection->getContents(),
                                      BC.AsmInfo->isLittleEndian(),
                                      BC.AsmInfo->getCodePointerSize());
  DataExtractor IPDE = DataExtractor(ORCUnwindIPSection->getContents(),
                                     BC.AsmInfo->isLittleEndian(),
                                     BC.AsmInfo->getCodePointerSize());
  DataExtractor::Cursor ORCCursor(0);
  DataExtractor::Cursor IPCursor(0);
  uint64_t PrevIP = 0;
  for (uint32_t Index = 0; Index < NumEntries; ++Index) {
    const uint64_t IP =
        IPSectionAddress + IPCursor.tell() + (int32_t)IPDE.getU32(IPCursor);

    // Consume the status of the cursor.
    if (!IPCursor)
      return createStringError(errc::executable_format_error,
                               "out of bounds while reading ORC IP table");

    if (IP < PrevIP && opts::Verbosity)
      errs() << "BOLT-WARNING: out of order IP 0x" << Twine::utohexstr(IP)
             << " detected while reading ORC\n";

    PrevIP = IP;

    // Store all entries, includes those we are not going to update as the
    // tables need to be sorted globally before being written out.
    ORCEntries.push_back(ORCListEntry());
    ORCListEntry &Entry = ORCEntries.back();

    Entry.IP = IP;
    Entry.ORC.SPOffset = (int16_t)OrcDE.getU16(ORCCursor);
    Entry.ORC.BPOffset = (int16_t)OrcDE.getU16(ORCCursor);
    Entry.ORC.Info = (int16_t)OrcDE.getU16(ORCCursor);

    // Consume the status of the cursor.
    if (!ORCCursor)
      return createStringError(errc::executable_format_error,
                               "out of bounds while reading ORC");

    BinaryFunction *&BF = Entry.BF;
    BF = BC.getBinaryFunctionContainingAddress(IP, /*CheckPastEnd*/ true);

    // If the entry immediately pointing past the end of the function is not
    // the terminator entry, then it does not belong to this function.
    if (BF && BF->getAddress() + BF->getSize() == IP && Entry.ORC != NullORC)
      BF = 0;

    // If terminator entry points to the start of the function, then it belongs
    // to a different function that contains the previous IP.
    if (BF && BF->getAddress() == IP && Entry.ORC == NullORC)
      BF = BC.getBinaryFunctionContainingAddress(IP - 1);

    if (!BF) {
      if (opts::Verbosity)
        errs() << "BOLT-WARNING: no binary function found matching ORC 0x"
               << Twine::utohexstr(IP) << ": " << Entry.ORC << '\n';
      continue;
    }

    if (Entry.ORC == NullORC)
      continue;

    BF->setHasORC(true);

    if (!BF->hasInstructions())
      continue;

    MCInst *Inst = BF->getInstructionAtOffset(IP - BF->getAddress());
    if (!Inst)
      return createStringError(
          errc::executable_format_error,
          "no instruction at address 0x%" PRIx64 " in .orc_unwind_ip", IP);

    // Some addresses will have two entries associated with them. The first
    // one being a "weak" section terminator. Since we ignore the terminator,
    // we should only assign one entry per instruction.
    if (BC.MIB->hasAnnotation(*Inst, "ORC"))
      return createStringError(
          errc::executable_format_error,
          "duplicate non-terminal ORC IP 0x%" PRIx64 " in .orc_unwind_ip", IP);

    BC.MIB->addAnnotation(*Inst, "ORC", Entry.ORC);
  }

  // Older kernels could contain unsorted tables in the file as the tables  were
  // sorted during boot time.
  llvm::sort(ORCEntries);

  if (opts::DumpORC) {
    outs() << "BOLT-INFO: ORC unwind information:\n";
    for (const ORCListEntry &E : ORCEntries) {
      outs() << "0x" << Twine::utohexstr(E.IP) << ": " << E.ORC;
      if (E.BF)
        outs() << ": " << *E.BF;
      outs() << '\n';
    }
  }

  return Error::success();
}

Error LinuxKernelRewriter::processORCPostCFG() {
  // Propagate ORC to the rest of the function. We can annotate every
  // instruction in every function, but to minimize the overhead, we annotate
  // the first instruction in every basic block to reflect the state at the
  // entry. This way, the ORC state can be calculated based on annotations
  // regardless of the basic block layout. Note that if we insert/delete
  // instructions, we must take care to attach ORC info to the new/deleted ones.
  for (BinaryFunction &BF : llvm::make_second_range(BC.getBinaryFunctions())) {

    std::optional<ORCState> CurrentState;
    for (BinaryBasicBlock &BB : BF) {
      for (MCInst &Inst : BB) {
        ErrorOr<ORCState> State =
            BC.MIB->tryGetAnnotationAs<ORCState>(Inst, "ORC");

        if (State) {
          CurrentState = *State;
          continue;
        }

        // In case there was no ORC entry that matched the function start
        // address, we need to propagate ORC state from the previous entry.
        if (!CurrentState) {
          auto It =
              llvm::partition_point(ORCEntries, [&](const ORCListEntry &E) {
                return E.IP < BF.getAddress();
              });
          if (It != ORCEntries.begin())
            It = std::prev(It);

          if (It->ORC == NullORC && BF.hasORC())
            errs() << "BOLT-WARNING: ORC unwind info excludes prologue for "
                   << BF << '\n';

          CurrentState = It->ORC;
          if (It->ORC != NullORC)
            BF.setHasORC(true);
        }

        // While printing ORC, attach info to every instruction for convenience.
        if (opts::PrintORC || &Inst == &BB.front())
          BC.MIB->addAnnotation(Inst, "ORC", *CurrentState);
      }
    }
  }

  return Error::success();
}

Error LinuxKernelRewriter::rewriteORCTables() {
  // TODO:
  return Error::success();
}
} // namespace

std::unique_ptr<MetadataRewriter>
llvm::bolt::createLinuxKernelRewriter(BinaryContext &BC) {
  return std::make_unique<LinuxKernelRewriter>(BC);
}
