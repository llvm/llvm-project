//===- bolt/Core/BinaryContext.h - Low-level context ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Context for processing binary executable/library files.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_BINARY_CONTEXT_H
#define BOLT_CORE_BINARY_CONTEXT_H

#include "bolt/Core/BinaryData.h"
#include "bolt/Core/BinarySection.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Core/JumpTable.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/RuntimeLibs/RuntimeLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/iterator.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <map>
#include <set>
#include <shared_mutex>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace llvm {
class MCDisassembler;
class MCInstPrinter;

using namespace object;

namespace bolt {

class BinaryFunction;
class ExecutableFileMemoryManager;

/// Information on loadable part of the file.
struct SegmentInfo {
  uint64_t Address;           /// Address of the segment in memory.
  uint64_t Size;              /// Size of the segment in memory.
  uint64_t FileOffset;        /// Offset in the file.
  uint64_t FileSize;          /// Size in file.
  uint64_t Alignment;         /// Alignment of the segment.

  void print(raw_ostream &OS) const {
    OS << "SegmentInfo { Address: 0x"
       << Twine::utohexstr(Address) << ", Size: 0x"
       << Twine::utohexstr(Size) << ", FileOffset: 0x"
       << Twine::utohexstr(FileOffset) << ", FileSize: 0x"
       << Twine::utohexstr(FileSize) << ", Alignment: 0x"
       << Twine::utohexstr(Alignment) << "}";
  };
};

inline raw_ostream &operator<<(raw_ostream &OS, const SegmentInfo &SegInfo) {
  SegInfo.print(OS);
  return OS;
}

enum class MemoryContentsType : char {
  UNKNOWN = 0,             /// Unknown contents.
  POSSIBLE_JUMP_TABLE,     /// Possibly a non-PIC jump table.
  POSSIBLE_PIC_JUMP_TABLE, /// Possibly a PIC jump table.
};

/// Helper function to truncate a \p Value to given size in \p Bytes.
inline int64_t truncateToSize(int64_t Value, unsigned Bytes) {
  return Value & ((uint64_t)(int64_t)-1 >> (64 - Bytes * 8));
}

/// Filter iterator.
template <typename ItrType,
          typename PredType = std::function<bool(const ItrType &)>>
class FilterIterator
    : public std::iterator<std::bidirectional_iterator_tag,
                           typename std::iterator_traits<ItrType>::value_type> {
  using Iterator = FilterIterator;
  using T = typename std::iterator_traits<ItrType>::reference;
  using PointerT = typename std::iterator_traits<ItrType>::pointer;

  PredType Pred;
  ItrType Itr, End;

  void prev() {
    while (!Pred(--Itr))
      ;
  }
  void next() {
    ++Itr;
    nextMatching();
  }
  void nextMatching() {
    while (Itr != End && !Pred(Itr))
      ++Itr;
  }

public:
  Iterator &operator++() { next(); return *this; }
  Iterator &operator--() { prev(); return *this; }
  Iterator operator++(int) { auto Tmp(Itr); next(); return Tmp; }
  Iterator operator--(int) { auto Tmp(Itr); prev(); return Tmp; }
  bool operator==(const Iterator &Other) const { return Itr == Other.Itr; }
  bool operator!=(const Iterator &Other) const { return !operator==(Other); }
  T operator*() { return *Itr; }
  PointerT operator->() { return &operator*(); }
  FilterIterator(PredType Pred, ItrType Itr, ItrType End)
      : Pred(Pred), Itr(Itr), End(End) {
    nextMatching();
  }
};

class BinaryContext {
  BinaryContext() = delete;

  /// Name of the binary file the context originated from.
  std::string Filename;

  /// Unique build ID if available for the binary.
  Optional<std::string> FileBuildID;

  /// Set of all sections.
  struct CompareSections {
    bool operator()(const BinarySection *A, const BinarySection *B) const {
      return *A < *B;
    }
  };
  using SectionSetType = std::set<BinarySection *, CompareSections>;
  SectionSetType Sections;

  using SectionIterator = pointee_iterator<SectionSetType::iterator>;
  using SectionConstIterator = pointee_iterator<SectionSetType::const_iterator>;

  using FilteredSectionIterator = FilterIterator<SectionIterator>;
  using FilteredSectionConstIterator = FilterIterator<SectionConstIterator>;

  /// Map virtual address to a section.  It is possible to have more than one
  /// section mapped to the same address, e.g. non-allocatable sections.
  using AddressToSectionMapType = std::multimap<uint64_t, BinarySection *>;
  AddressToSectionMapType AddressToSection;

  /// multimap of section name to BinarySection object.  Some binaries
  /// have multiple sections with the same name.
  using NameToSectionMapType = std::multimap<std::string, BinarySection *>;
  NameToSectionMapType NameToSection;

  /// Low level section registration.
  BinarySection &registerSection(BinarySection *Section);

  /// Store all functions in the binary, sorted by original address.
  std::map<uint64_t, BinaryFunction> BinaryFunctions;

  /// A mutex that is used to control parallel accesses to BinaryFunctions
  mutable std::shared_timed_mutex BinaryFunctionsMutex;

  /// Functions injected by BOLT
  std::vector<BinaryFunction *> InjectedBinaryFunctions;

  /// Jump tables for all functions mapped by address.
  std::map<uint64_t, JumpTable *> JumpTables;

  /// Locations of PC-relative relocations in data objects.
  std::unordered_set<uint64_t> DataPCRelocations;

  /// Used in duplicateJumpTable() to uniquely identify a JT clone
  /// Start our IDs with a high number so getJumpTableContainingAddress checks
  /// with size won't overflow
  uint32_t DuplicatedJumpTables{0x10000000};

  /// Function fragments to skip.
  std::vector<BinaryFunction *> FragmentsToSkip;

  /// The runtime library.
  std::unique_ptr<RuntimeLibrary> RtLibrary;

  /// DWP Context.
  std::shared_ptr<DWARFContext> DWPContext;

  /// A map of DWO Ids to CUs.
  using DWOIdToCUMapType = std::unordered_map<uint64_t, DWARFUnit *>;
  DWOIdToCUMapType DWOCUs;

  /// Preprocess DWO debug information.
  void preprocessDWODebugInfo();

  /// DWARF line info for CUs.
  std::map<unsigned, DwarfLineTable> DwarfLineTablesCUMap;

public:
  static Expected<std::unique_ptr<BinaryContext>>
  createBinaryContext(const ObjectFile *File, bool IsPIC,
                      std::unique_ptr<DWARFContext> DwCtx);

  /// Superset of compiler units that will contain overwritten code that needs
  /// new debug info. In a few cases, functions may end up not being
  /// overwritten, but it is okay to re-generate debug info for them.
  std::set<const DWARFUnit *> ProcessedCUs;

  // Setup MCPlus target builder
  void initializeTarget(std::unique_ptr<MCPlusBuilder> TargetBuilder) {
    MIB = std::move(TargetBuilder);
  }

  /// Given DWOId returns CU if it exists in DWOCUs.
  Optional<DWARFUnit *> getDWOCU(uint64_t DWOId);

  /// Returns DWOContext if it exists.
  DWARFContext *getDWOContext();

  /// Get Number of DWOCUs in a map.
  uint32_t getNumDWOCUs() { return DWOCUs.size(); }

  const std::map<unsigned, DwarfLineTable> &getDwarfLineTables() const {
    return DwarfLineTablesCUMap;
  }

  DwarfLineTable &getDwarfLineTable(unsigned CUID) {
    return DwarfLineTablesCUMap[CUID];
  }

  Expected<unsigned> getDwarfFile(StringRef Directory, StringRef FileName,
                                  unsigned FileNumber,
                                  Optional<MD5::MD5Result> Checksum,
                                  Optional<StringRef> Source, unsigned CUID);

  /// [start memory address] -> [segment info] mapping.
  std::map<uint64_t, SegmentInfo> SegmentMapInfo;

  /// Symbols that are expected to be undefined in MCContext during emission.
  std::unordered_set<MCSymbol *> UndefinedSymbols;

  /// [name] -> [BinaryData*] map used for global symbol resolution.
  using SymbolMapType = StringMap<BinaryData *>;
  SymbolMapType GlobalSymbols;

  /// [address] -> [BinaryData], ...
  /// Addresses never change.
  /// Note: it is important that clients do not hold on to instances of
  /// BinaryData* while the map is still being modified during BinaryFunction
  /// disassembly.  This is because of the possibility that a regular
  /// BinaryData is later discovered to be a JumpTable.
  using BinaryDataMapType = std::map<uint64_t, BinaryData *>;
  using binary_data_iterator = BinaryDataMapType::iterator;
  using binary_data_const_iterator = BinaryDataMapType::const_iterator;
  BinaryDataMapType BinaryDataMap;

  using FilteredBinaryDataConstIterator =
      FilterIterator<binary_data_const_iterator>;
  using FilteredBinaryDataIterator = FilterIterator<binary_data_iterator>;

  /// Memory manager for sections and segments. Used to communicate with ORC
  /// among other things.
  std::shared_ptr<ExecutableFileMemoryManager> EFMM;

  StringRef getFilename() const { return Filename; }
  void setFilename(StringRef Name) { Filename = std::string(Name); }

  Optional<StringRef> getFileBuildID() const {
    if (FileBuildID)
      return StringRef(*FileBuildID);

    return NoneType();
  }
  void setFileBuildID(StringRef ID) { FileBuildID = std::string(ID); }

  bool hasSymbolsWithFileName() const { return HasSymbolsWithFileName; }
  void setHasSymbolsWithFileName(bool Value) { HasSymbolsWithFileName = true; }

  /// Return true if relocations against symbol with a given name
  /// must be created.
  bool forceSymbolRelocations(StringRef SymbolName) const;

  uint64_t getNumUnusedProfiledObjects() const {
    return NumUnusedProfiledObjects;
  }
  void setNumUnusedProfiledObjects(uint64_t N) { NumUnusedProfiledObjects = N; }

  RuntimeLibrary *getRuntimeLibrary() { return RtLibrary.get(); }
  void setRuntimeLibrary(std::unique_ptr<RuntimeLibrary> Lib) {
    assert(!RtLibrary && "Cannot set runtime library twice.");
    RtLibrary = std::move(Lib);
  }

  /// Return BinaryFunction containing a given \p Address or nullptr if
  /// no registered function contains the \p Address.
  ///
  /// In a binary a function has somewhat vague  boundaries. E.g. a function can
  /// refer to the first byte past the end of the function, and it will still be
  /// referring to this function, not the function following it in the address
  /// space. Thus we have the following flags that allow to lookup for
  /// a function where a caller has more context for the search.
  ///
  /// If \p CheckPastEnd is true and the \p Address falls on a byte
  /// immediately following the last byte of some function and there's no other
  /// function that starts there, then return the function as the one containing
  /// the \p Address. This is useful when we need to locate functions for
  /// references pointing immediately past a function body.
  ///
  /// If \p UseMaxSize is true, then include the space between this function
  /// body and the next object in address ranges that we check.
  BinaryFunction *getBinaryFunctionContainingAddress(uint64_t Address,
                                                     bool CheckPastEnd = false,
                                                     bool UseMaxSize = false);

  /// Return a BinaryFunction that starts at a given \p Address.
  BinaryFunction *getBinaryFunctionAtAddress(uint64_t Address);

  const BinaryFunction *getBinaryFunctionAtAddress(uint64_t Address) const {
    return const_cast<BinaryContext *>(this)->getBinaryFunctionAtAddress(
        Address);
  }

  /// Return size of an entry for the given jump table \p Type.
  uint64_t getJumpTableEntrySize(JumpTable::JumpTableType Type) const {
    return Type == JumpTable::JTT_PIC ? 4 : AsmInfo->getCodePointerSize();
  }

  /// Return JumpTable containing a given \p Address.
  JumpTable *getJumpTableContainingAddress(uint64_t Address) {
    auto JTI = JumpTables.upper_bound(Address);
    if (JTI == JumpTables.begin())
      return nullptr;
    --JTI;
    if (JTI->first + JTI->second->getSize() > Address)
      return JTI->second;
    if (JTI->second->getSize() == 0 && JTI->first == Address)
      return JTI->second;
    return nullptr;
  }

  unsigned getDWARFEncodingSize(unsigned Encoding) {
    switch (Encoding & 0x0f) {
    default:
      llvm_unreachable("unknown encoding");
    case dwarf::DW_EH_PE_absptr:
    case dwarf::DW_EH_PE_signed:
      return AsmInfo->getCodePointerSize();
    case dwarf::DW_EH_PE_udata2:
    case dwarf::DW_EH_PE_sdata2:
      return 2;
    case dwarf::DW_EH_PE_udata4:
    case dwarf::DW_EH_PE_sdata4:
      return 4;
    case dwarf::DW_EH_PE_udata8:
    case dwarf::DW_EH_PE_sdata8:
      return 8;
    }
  }

  /// [MCSymbol] -> [BinaryFunction]
  ///
  /// As we fold identical functions, multiple symbols can point
  /// to the same BinaryFunction.
  std::unordered_map<const MCSymbol *, BinaryFunction *> SymbolToFunctionMap;

  /// A mutex that is used to control parallel accesses to SymbolToFunctionMap
  mutable std::shared_timed_mutex SymbolToFunctionMapMutex;

  /// Look up the symbol entry that contains the given \p Address (based on
  /// the start address and size for each symbol).  Returns a pointer to
  /// the BinaryData for that symbol.  If no data is found, nullptr is returned.
  const BinaryData *getBinaryDataContainingAddressImpl(uint64_t Address) const;

  /// Update the Parent fields in BinaryDatas after adding a new entry into
  /// \p BinaryDataMap.
  void updateObjectNesting(BinaryDataMapType::iterator GAI);

  /// Validate that if object address ranges overlap that the object with
  /// the larger range is a parent of the object with the smaller range.
  bool validateObjectNesting() const;

  /// Validate that there are no top level "holes" in each section
  /// and that all relocations with a section are mapped to a valid
  /// top level BinaryData.
  bool validateHoles() const;

  /// Produce output address ranges based on input ranges for some module.
  DebugAddressRangesVector translateModuleAddressRanges(
      const DWARFAddressRangesVector &InputRanges) const;

  /// Get a bogus "absolute" section that will be associated with all
  /// absolute BinaryDatas.
  BinarySection &absoluteSection();

  /// Process "holes" in between known BinaryData objects.  For now,
  /// symbols are padded with the space before the next BinaryData object.
  void fixBinaryDataHoles();

  /// Generate names based on data hashes for unknown symbols.
  void generateSymbolHashes();

  /// Construct BinaryFunction object and add it to internal maps.
  BinaryFunction *createBinaryFunction(const std::string &Name,
                                       BinarySection &Section, uint64_t Address,
                                       uint64_t Size, uint64_t SymbolSize = 0,
                                       uint16_t Alignment = 0);

  /// Return all functions for this rewrite instance.
  std::map<uint64_t, BinaryFunction> &getBinaryFunctions() {
    return BinaryFunctions;
  }

  /// Return all functions for this rewrite instance.
  const std::map<uint64_t, BinaryFunction> &getBinaryFunctions() const {
    return BinaryFunctions;
  }

  /// Create BOLT-injected function
  BinaryFunction *createInjectedBinaryFunction(const std::string &Name,
                                               bool IsSimple = true);

  std::vector<BinaryFunction *> &getInjectedBinaryFunctions() {
    return InjectedBinaryFunctions;
  }

  /// Return vector with all functions, i.e. include functions from the input
  /// binary and functions created by BOLT.
  std::vector<BinaryFunction *> getAllBinaryFunctions();

  /// Construct a jump table for \p Function at \p Address or return an existing
  /// one at that location.
  ///
  /// May create an embedded jump table and return its label as the second
  /// element of the pair.
  const MCSymbol *getOrCreateJumpTable(BinaryFunction &Function,
                                       uint64_t Address,
                                       JumpTable::JumpTableType Type);

  /// Analyze a possible jump table of type \p Type at a given \p Address.
  /// \p BF is a function referencing the jump table.
  /// Return true if the jump table was detected at \p Address, and false
  /// otherwise.
  ///
  /// If \p NextJTAddress is different from zero, it is used as an upper
  /// bound for jump table memory layout.
  ///
  /// Optionally, populate \p Offsets with jump table entries. The entries
  /// could be partially populated if the jump table detection fails.
  bool analyzeJumpTable(const uint64_t Address,
                        const JumpTable::JumpTableType Type, BinaryFunction &BF,
                        const uint64_t NextJTAddress = 0,
                        JumpTable::OffsetsType *Offsets = nullptr);

  /// After jump table locations are established, this function will populate
  /// their OffsetEntries based on memory contents.
  void populateJumpTables();

  /// Returns a jump table ID and label pointing to the duplicated jump table.
  /// Ordinarily, jump tables are identified by their address in the input
  /// binary. We return an ID with the high bit set to differentiate it from
  /// regular addresses, avoiding conflicts with standard jump tables.
  std::pair<uint64_t, const MCSymbol *>
  duplicateJumpTable(BinaryFunction &Function, JumpTable *JT,
                     const MCSymbol *OldLabel);

  /// Generate a unique name for jump table at a given \p Address belonging
  /// to function \p BF.
  std::string generateJumpTableName(const BinaryFunction &BF, uint64_t Address);

  /// Return true if the array of bytes represents a valid code padding.
  bool hasValidCodePadding(const BinaryFunction &BF);

  /// Verify padding area between functions, and adjust max function size
  /// accordingly.
  void adjustCodePadding();

  /// Regular page size.
  static constexpr unsigned RegularPageSize = 0x1000;

  /// Huge page size to use.
  static constexpr unsigned HugePageSize = 0x200000;

  /// Map address to a constant island owner (constant data in code section)
  std::map<uint64_t, BinaryFunction *> AddressToConstantIslandMap;

  /// A map from jump table address to insertion order.  Used for generating
  /// jump table names.
  std::map<uint64_t, size_t> JumpTableIds;

  std::unique_ptr<MCContext> Ctx;

  /// A mutex that is used to control parallel accesses to Ctx
  mutable std::shared_timed_mutex CtxMutex;
  std::unique_lock<std::shared_timed_mutex> scopeLock() const {
    return std::unique_lock<std::shared_timed_mutex>(CtxMutex);
  }

  std::unique_ptr<DWARFContext> DwCtx;

  std::unique_ptr<Triple> TheTriple;

  const Target *TheTarget;

  std::string TripleName;

  std::unique_ptr<MCCodeEmitter> MCE;

  std::unique_ptr<MCObjectFileInfo> MOFI;

  std::unique_ptr<const MCAsmInfo> AsmInfo;

  std::unique_ptr<const MCInstrInfo> MII;

  std::unique_ptr<const MCSubtargetInfo> STI;

  std::unique_ptr<MCInstPrinter> InstPrinter;

  std::unique_ptr<const MCInstrAnalysis> MIA;

  std::unique_ptr<MCPlusBuilder> MIB;

  std::unique_ptr<const MCRegisterInfo> MRI;

  std::unique_ptr<MCDisassembler> DisAsm;

  std::unique_ptr<MCAsmBackend> MAB;

  /// Indicates if relocations are available for usage.
  bool HasRelocations{false};

  /// Is the binary always loaded at a fixed address. Shared objects and
  /// position-independent executables (PIEs) are examples of binaries that
  /// will have HasFixedLoadAddress set to false.
  bool HasFixedLoadAddress{true};

  /// True if the binary has no dynamic dependencies, i.e., if it was statically
  /// linked.
  bool IsStaticExecutable{false};

  /// Set to true if the binary contains PT_INTERP header.
  bool HasInterpHeader{false};

  /// Indicates if any of local symbols used for functions or data objects
  /// have an origin file name available.
  bool HasSymbolsWithFileName{false};

  /// Sum of execution count of all functions
  uint64_t SumExecutionCount{0};

  /// Number of functions with profile information
  uint64_t NumProfiledFuncs{0};

  /// Number of objects in profile whose profile was ignored.
  uint64_t NumUnusedProfiledObjects{0};

  /// Total hotness score according to profiling data for this binary.
  uint64_t TotalScore{0};

  /// Binary-wide stats for macro-fusion.
  uint64_t MissedMacroFusionPairs{0};
  uint64_t MissedMacroFusionExecCount{0};

  // Address of the first allocated segment.
  uint64_t FirstAllocAddress{std::numeric_limits<uint64_t>::max()};

  /// Track next available address for new allocatable sections. RewriteInstance
  /// sets this prior to running BOLT passes, so layout passes are aware of the
  /// final addresses functions will have.
  uint64_t LayoutStartAddress{0};

  /// Old .text info.
  uint64_t OldTextSectionAddress{0};
  uint64_t OldTextSectionOffset{0};
  uint64_t OldTextSectionSize{0};

  /// Address of the code/function that is executed before any other code in
  /// the binary.
  Optional<uint64_t> StartFunctionAddress;

  /// Address of the code/function that is going to be executed right before
  /// the execution of the binary is completed.
  Optional<uint64_t> FiniFunctionAddress;

  /// Page alignment used for code layout.
  uint64_t PageAlign{HugePageSize};

  /// True if the binary requires immediate relocation processing.
  bool RequiresZNow{false};

  /// List of functions that always trap.
  std::vector<const BinaryFunction *> TrappedFunctions;

  /// Map SDT locations to SDT markers info
  std::unordered_map<uint64_t, SDTMarkerInfo> SDTMarkers;

  /// Map linux kernel program locations/instructions to their pointers in
  /// special linux kernel sections
  std::unordered_map<uint64_t, std::vector<LKInstructionMarkerInfo>> LKMarkers;

  /// PseudoProbe decoder
  MCPseudoProbeDecoder ProbeDecoder;

  /// DWARF encoding. Available encoding types defined in BinaryFormat/Dwarf.h
  /// enum Constants, e.g. DW_EH_PE_omit.
  unsigned TTypeEncoding = dwarf::DW_EH_PE_omit;
  unsigned LSDAEncoding = dwarf::DW_EH_PE_omit;

  BinaryContext(std::unique_ptr<MCContext> Ctx,
                std::unique_ptr<DWARFContext> DwCtx,
                std::unique_ptr<Triple> TheTriple, const Target *TheTarget,
                std::string TripleName, std::unique_ptr<MCCodeEmitter> MCE,
                std::unique_ptr<MCObjectFileInfo> MOFI,
                std::unique_ptr<const MCAsmInfo> AsmInfo,
                std::unique_ptr<const MCInstrInfo> MII,
                std::unique_ptr<const MCSubtargetInfo> STI,
                std::unique_ptr<MCInstPrinter> InstPrinter,
                std::unique_ptr<const MCInstrAnalysis> MIA,
                std::unique_ptr<MCPlusBuilder> MIB,
                std::unique_ptr<const MCRegisterInfo> MRI,
                std::unique_ptr<MCDisassembler> DisAsm);

  ~BinaryContext();

  std::unique_ptr<MCObjectWriter> createObjectWriter(raw_pwrite_stream &OS);

  bool isELF() const { return TheTriple->isOSBinFormatELF(); }

  bool isMachO() const { return TheTriple->isOSBinFormatMachO(); }

  bool isAArch64() const {
    return TheTriple->getArch() == llvm::Triple::aarch64;
  }

  bool isX86() const {
    return TheTriple->getArch() == llvm::Triple::x86 ||
           TheTriple->getArch() == llvm::Triple::x86_64;
  }

  /// Iterate over all BinaryData.
  iterator_range<binary_data_const_iterator> getBinaryData() const {
    return make_range(BinaryDataMap.begin(), BinaryDataMap.end());
  }

  /// Iterate over all BinaryData.
  iterator_range<binary_data_iterator> getBinaryData() {
    return make_range(BinaryDataMap.begin(), BinaryDataMap.end());
  }

  /// Iterate over all BinaryData associated with the given \p Section.
  iterator_range<FilteredBinaryDataConstIterator>
  getBinaryDataForSection(const BinarySection &Section) const {
    auto Begin = BinaryDataMap.lower_bound(Section.getAddress());
    if (Begin != BinaryDataMap.begin())
      --Begin;
    auto End = BinaryDataMap.upper_bound(Section.getEndAddress());
    auto pred = [&Section](const binary_data_const_iterator &Itr) -> bool {
      return Itr->second->getSection() == Section;
    };
    return make_range(FilteredBinaryDataConstIterator(pred, Begin, End),
                      FilteredBinaryDataConstIterator(pred, End, End));
  }

  /// Iterate over all BinaryData associated with the given \p Section.
  iterator_range<FilteredBinaryDataIterator>
  getBinaryDataForSection(BinarySection &Section) {
    auto Begin = BinaryDataMap.lower_bound(Section.getAddress());
    if (Begin != BinaryDataMap.begin())
      --Begin;
    auto End = BinaryDataMap.upper_bound(Section.getEndAddress());
    auto pred = [&Section](const binary_data_iterator &Itr) -> bool {
      return Itr->second->getSection() == Section;
    };
    return make_range(FilteredBinaryDataIterator(pred, Begin, End),
                      FilteredBinaryDataIterator(pred, End, End));
  }

  /// Iterate over all the sub-symbols of /p BD (if any).
  iterator_range<binary_data_iterator> getSubBinaryData(BinaryData *BD);

  /// Clear the global symbol address -> name(s) map.
  void clearBinaryData() {
    GlobalSymbols.clear();
    for (auto &Entry : BinaryDataMap)
      delete Entry.second;
    BinaryDataMap.clear();
  }

  /// Process \p Address reference from code in function \BF.
  /// \p IsPCRel indicates if the reference is PC-relative.
  /// Return <Symbol, Addend> pair corresponding to the \p Address.
  std::pair<const MCSymbol *, uint64_t>
  handleAddressRef(uint64_t Address, BinaryFunction &BF, bool IsPCRel);

  /// Analyze memory contents at the given \p Address and return the type of
  /// memory contents (such as a possible jump table).
  MemoryContentsType analyzeMemoryAt(uint64_t Address, BinaryFunction &BF);

  /// Return a value of the global \p Symbol or an error if the value
  /// was not set.
  ErrorOr<uint64_t> getSymbolValue(const MCSymbol &Symbol) const {
    const BinaryData *BD = getBinaryDataByName(Symbol.getName());
    if (!BD)
      return std::make_error_code(std::errc::bad_address);
    return BD->getAddress();
  }

  /// Return a global symbol registered at a given \p Address and \p Size.
  /// If no symbol exists, create one with unique name using \p Prefix.
  /// If there are multiple symbols registered at the \p Address, then
  /// return the first one.
  MCSymbol *getOrCreateGlobalSymbol(uint64_t Address, Twine Prefix,
                                    uint64_t Size = 0, uint16_t Alignment = 0,
                                    unsigned Flags = 0);

  /// Create a global symbol without registering an address.
  MCSymbol *getOrCreateUndefinedGlobalSymbol(StringRef Name);

  /// Register a symbol with \p Name at a given \p Address using \p Size,
  /// \p Alignment, and \p Flags. See llvm::SymbolRef::Flags for the definition
  /// of \p Flags.
  MCSymbol *registerNameAtAddress(StringRef Name, uint64_t Address,
                                  uint64_t Size, uint16_t Alignment,
                                  unsigned Flags = 0);

  /// Return BinaryData registered at a given \p Address or nullptr if no
  /// global symbol was registered at the location.
  const BinaryData *getBinaryDataAtAddress(uint64_t Address) const {
    auto NI = BinaryDataMap.find(Address);
    return NI != BinaryDataMap.end() ? NI->second : nullptr;
  }

  BinaryData *getBinaryDataAtAddress(uint64_t Address) {
    auto NI = BinaryDataMap.find(Address);
    return NI != BinaryDataMap.end() ? NI->second : nullptr;
  }

  /// Look up the symbol entry that contains the given \p Address (based on
  /// the start address and size for each symbol).  Returns a pointer to
  /// the BinaryData for that symbol.  If no data is found, nullptr is returned.
  const BinaryData *getBinaryDataContainingAddress(uint64_t Address) const {
    return getBinaryDataContainingAddressImpl(Address);
  }

  BinaryData *getBinaryDataContainingAddress(uint64_t Address) {
    return const_cast<BinaryData *>(
        getBinaryDataContainingAddressImpl(Address));
  }

  /// Return BinaryData for the given \p Name or nullptr if no
  /// global symbol with that name exists.
  const BinaryData *getBinaryDataByName(StringRef Name) const {
    auto Itr = GlobalSymbols.find(Name);
    return Itr != GlobalSymbols.end() ? Itr->second : nullptr;
  }

  BinaryData *getBinaryDataByName(StringRef Name) {
    auto Itr = GlobalSymbols.find(Name);
    return Itr != GlobalSymbols.end() ? Itr->second : nullptr;
  }

  /// Return true if \p SymbolName was generated internally and was not present
  /// in the input binary.
  bool isInternalSymbolName(const StringRef Name) {
    return Name.startswith("SYMBOLat") || Name.startswith("DATAat") ||
           Name.startswith("HOLEat");
  }

  MCSymbol *getHotTextStartSymbol() const {
    return Ctx->getOrCreateSymbol("__hot_start");
  }

  MCSymbol *getHotTextEndSymbol() const {
    return Ctx->getOrCreateSymbol("__hot_end");
  }

  MCSection *getTextSection() const { return MOFI->getTextSection(); }

  /// Return code section with a given name.
  MCSection *getCodeSection(StringRef SectionName) const {
    if (isELF())
      return Ctx->getELFSection(SectionName, ELF::SHT_PROGBITS,
                                ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);
    else
      return Ctx->getMachOSection("__TEXT", SectionName,
                                  MachO::S_ATTR_PURE_INSTRUCTIONS,
                                  SectionKind::getText());
  }

  /// Return data section with a given name.
  MCSection *getDataSection(StringRef SectionName) const {
    return Ctx->getELFSection(SectionName, ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
  }

  /// \name Pre-assigned Section Names
  /// @{

  const char *getMainCodeSectionName() const { return ".text"; }

  const char *getColdCodeSectionName() const { return ".text.cold"; }

  const char *getHotTextMoverSectionName() const { return ".text.mover"; }

  const char *getInjectedCodeSectionName() const { return ".text.injected"; }

  const char *getInjectedColdCodeSectionName() const {
    return ".text.injected.cold";
  }

  ErrorOr<BinarySection &> getGdbIndexSection() const {
    return getUniqueSectionByName(".gdb_index");
  }

  /// @}

  /// Register \p TargetFunction as a fragment of \p Function if checks pass:
  /// - if \p TargetFunction name matches \p Function name with a suffix:
  ///   fragment_name == parent_name.cold(.\d+)?
  /// True if the Function is registered, false if the check failed.
  bool registerFragment(BinaryFunction &TargetFunction,
                        BinaryFunction &Function) const;

  /// Resolve inter-procedural dependencies from \p Function.
  void processInterproceduralReferences(BinaryFunction &Function);

  /// Skip functions with all parent and child fragments transitively.
  void skipMarkedFragments();

  /// Perform any necessary post processing on the symbol table after
  /// function disassembly is complete.  This processing fixes top
  /// level data holes and makes sure the symbol table is valid.
  /// It also assigns all memory profiling info to the appropriate
  /// BinaryData objects.
  void postProcessSymbolTable();

  /// Set the size of the global symbol located at \p Address.  Return
  /// false if no symbol exists, true otherwise.
  bool setBinaryDataSize(uint64_t Address, uint64_t Size);

  /// Print the global symbol table.
  void printGlobalSymbols(raw_ostream &OS) const;

  /// Register information about the given \p Section so we can look up
  /// sections by address.
  BinarySection &registerSection(SectionRef Section);

  /// Register a copy of /p OriginalSection under a different name.
  BinarySection &registerSection(StringRef SectionName,
                                 const BinarySection &OriginalSection);

  /// Register or update the information for the section with the given
  /// /p Name.  If the section already exists, the information in the
  /// section will be updated with the new data.
  BinarySection &registerOrUpdateSection(StringRef Name, unsigned ELFType,
                                         unsigned ELFFlags,
                                         uint8_t *Data = nullptr,
                                         uint64_t Size = 0,
                                         unsigned Alignment = 1);

  /// Register the information for the note (non-allocatable) section
  /// with the given /p Name.  If the section already exists, the
  /// information in the section will be updated with the new data.
  BinarySection &
  registerOrUpdateNoteSection(StringRef Name, uint8_t *Data = nullptr,
                              uint64_t Size = 0, unsigned Alignment = 1,
                              bool IsReadOnly = true,
                              unsigned ELFType = ELF::SHT_PROGBITS) {
    return registerOrUpdateSection(Name, ELFType,
                                   BinarySection::getFlags(IsReadOnly), Data,
                                   Size, Alignment);
  }

  /// Remove the given /p Section from the set of all sections.  Return
  /// true if the section was removed (and deleted), otherwise false.
  bool deregisterSection(BinarySection &Section);

  /// Iterate over all registered sections.
  iterator_range<FilteredSectionIterator> sections() {
    auto notNull = [](const SectionIterator &Itr) { return (bool)*Itr; };
    return make_range(
        FilteredSectionIterator(notNull, Sections.begin(), Sections.end()),
        FilteredSectionIterator(notNull, Sections.end(), Sections.end()));
  }

  /// Iterate over all registered sections.
  iterator_range<FilteredSectionConstIterator> sections() const {
    return const_cast<BinaryContext *>(this)->sections();
  }

  /// Iterate over all registered allocatable sections.
  iterator_range<FilteredSectionIterator> allocatableSections() {
    auto isAllocatable = [](const SectionIterator &Itr) {
      return *Itr && Itr->isAllocatable();
    };
    return make_range(
        FilteredSectionIterator(isAllocatable, Sections.begin(),
                                Sections.end()),
        FilteredSectionIterator(isAllocatable, Sections.end(), Sections.end()));
  }

  /// Iterate over all registered code sections.
  iterator_range<FilteredSectionIterator> textSections() {
    auto isText = [](const SectionIterator &Itr) {
      return *Itr && Itr->isAllocatable() && Itr->isText();
    };
    return make_range(
        FilteredSectionIterator(isText, Sections.begin(), Sections.end()),
        FilteredSectionIterator(isText, Sections.end(), Sections.end()));
  }

  /// Iterate over all registered allocatable sections.
  iterator_range<FilteredSectionConstIterator> allocatableSections() const {
    return const_cast<BinaryContext *>(this)->allocatableSections();
  }

  /// Iterate over all registered non-allocatable sections.
  iterator_range<FilteredSectionIterator> nonAllocatableSections() {
    auto notAllocated = [](const SectionIterator &Itr) {
      return *Itr && !Itr->isAllocatable();
    };
    return make_range(
        FilteredSectionIterator(notAllocated, Sections.begin(), Sections.end()),
        FilteredSectionIterator(notAllocated, Sections.end(), Sections.end()));
  }

  /// Iterate over all registered non-allocatable sections.
  iterator_range<FilteredSectionConstIterator> nonAllocatableSections() const {
    return const_cast<BinaryContext *>(this)->nonAllocatableSections();
  }

  /// Iterate over all allocatable relocation sections.
  iterator_range<FilteredSectionIterator> allocatableRelaSections() {
    auto isAllocatableRela = [](const SectionIterator &Itr) {
      return *Itr && Itr->isAllocatable() && Itr->isRela();
    };
    return make_range(FilteredSectionIterator(isAllocatableRela,
                                              Sections.begin(), Sections.end()),
                      FilteredSectionIterator(isAllocatableRela, Sections.end(),
                                              Sections.end()));
  }

  /// Check if the address belongs to this binary's static allocation space.
  bool containsAddress(uint64_t Address) const {
    return Address >= FirstAllocAddress && Address < LayoutStartAddress;
  }

  /// Return section name containing the given \p Address.
  ErrorOr<StringRef> getSectionNameForAddress(uint64_t Address) const;

  /// Print all sections.
  void printSections(raw_ostream &OS) const;

  /// Return largest section containing the given \p Address.  These
  /// functions only work for allocatable sections, i.e. ones with non-zero
  /// addresses.
  ErrorOr<BinarySection &> getSectionForAddress(uint64_t Address);
  ErrorOr<const BinarySection &> getSectionForAddress(uint64_t Address) const {
    return const_cast<BinaryContext *>(this)->getSectionForAddress(Address);
  }

  /// Return section(s) associated with given \p Name.
  iterator_range<NameToSectionMapType::iterator>
  getSectionByName(StringRef Name) {
    return make_range(NameToSection.equal_range(std::string(Name)));
  }
  iterator_range<NameToSectionMapType::const_iterator>
  getSectionByName(StringRef Name) const {
    return make_range(NameToSection.equal_range(std::string(Name)));
  }

  /// Return the unique section associated with given \p Name.
  /// If there is more than one section with the same name, return an error
  /// object.
  ErrorOr<BinarySection &> getUniqueSectionByName(StringRef SectionName) const {
    auto Sections = getSectionByName(SectionName);
    if (Sections.begin() != Sections.end() &&
        std::next(Sections.begin()) == Sections.end())
      return *Sections.begin()->second;
    return std::make_error_code(std::errc::bad_address);
  }

  /// Return an unsigned value of \p Size stored at \p Address. The address has
  /// to be a valid statically allocated address for the binary.
  ErrorOr<uint64_t> getUnsignedValueAtAddress(uint64_t Address,
                                              size_t Size) const;

  /// Return a signed value of \p Size stored at \p Address. The address has
  /// to be a valid statically allocated address for the binary.
  ErrorOr<uint64_t> getSignedValueAtAddress(uint64_t Address,
                                            size_t Size) const;

  /// Special case of getUnsignedValueAtAddress() that uses a pointer size.
  ErrorOr<uint64_t> getPointerAtAddress(uint64_t Address) const {
    return getUnsignedValueAtAddress(Address, AsmInfo->getCodePointerSize());
  }

  /// Replaces all references to \p ChildBF with \p ParentBF. \p ChildBF is then
  /// removed from the list of functions \p BFs. The profile data of \p ChildBF
  /// is merged into that of \p ParentBF. This function is thread safe.
  void foldFunction(BinaryFunction &ChildBF, BinaryFunction &ParentBF);

  /// Add a Section relocation at a given \p Address.
  void addRelocation(uint64_t Address, MCSymbol *Symbol, uint64_t Type,
                     uint64_t Addend = 0, uint64_t Value = 0);

  /// Return a relocation registered at a given \p Address, or nullptr if there
  /// is no relocation at such address.
  const Relocation *getRelocationAt(uint64_t Address);

  /// Register a presence of PC-relative relocation at the given \p Address.
  void addPCRelativeDataRelocation(uint64_t Address) {
    DataPCRelocations.emplace(Address);
  }

  /// Register dynamic relocation at \p Address.
  void addDynamicRelocation(uint64_t Address, MCSymbol *Symbol, uint64_t Type,
                            uint64_t Addend, uint64_t Value = 0);

  /// Return a dynamic relocation registered at a given \p Address, or nullptr
  /// if there is no dynamic relocation at such address.
  const Relocation *getDynamicRelocationAt(uint64_t Address);

  /// Remove registered relocation at a given \p Address.
  bool removeRelocationAt(uint64_t Address);

  /// This function makes sure that symbols referenced by ambiguous relocations
  /// are marked as immovable. For now, if a section relocation points at the
  /// boundary between two symbols then those symbols are marked as immovable.
  void markAmbiguousRelocations(BinaryData &BD, const uint64_t Address);

  /// Return BinaryFunction corresponding to \p Symbol. If \p EntryDesc is not
  /// nullptr, set it to entry descriminator corresponding to \p Symbol
  /// (0 for single-entry functions). This function is thread safe.
  BinaryFunction *getFunctionForSymbol(const MCSymbol *Symbol,
                                       uint64_t *EntryDesc = nullptr);

  const BinaryFunction *
  getFunctionForSymbol(const MCSymbol *Symbol,
                       uint64_t *EntryDesc = nullptr) const {
    return const_cast<BinaryContext *>(this)->getFunctionForSymbol(Symbol,
                                                                   EntryDesc);
  }

  /// Associate the symbol \p Sym with the function \p BF for lookups with
  /// getFunctionForSymbol().
  void setSymbolToFunctionMap(const MCSymbol *Sym, BinaryFunction *BF) {
    SymbolToFunctionMap[Sym] = BF;
  }

  /// Populate some internal data structures with debug info.
  void preprocessDebugInfo();

  /// Add a filename entry from SrcCUID to DestCUID.
  unsigned addDebugFilenameToUnit(const uint32_t DestCUID,
                                  const uint32_t SrcCUID, unsigned FileIndex);

  /// Return functions in output layout order
  std::vector<BinaryFunction *> getSortedFunctions();

  /// Do the best effort to calculate the size of the function by emitting
  /// its code, and relaxing branch instructions. By default, branch
  /// instructions are updated to match the layout. Pass \p FixBranches set to
  /// false if the branches are known to be up to date with the code layout.
  ///
  /// Return the pair where the first size is for the main part, and the second
  /// size is for the cold one.
  std::pair<size_t, size_t> calculateEmittedSize(BinaryFunction &BF,
                                                 bool FixBranches = true);

  /// Calculate the size of the instruction \p Inst optionally using a
  /// user-supplied emitter for lock-free multi-thread work. MCCodeEmitter is
  /// not thread safe and each thread should operate with its own copy of it.
  uint64_t
  computeInstructionSize(const MCInst &Inst,
                         const MCCodeEmitter *Emitter = nullptr) const {
    if (auto Size = MIB->getAnnotationWithDefault<uint32_t>(Inst, "Size"))
      return Size;

    if (!Emitter)
      Emitter = this->MCE.get();
    SmallString<256> Code;
    SmallVector<MCFixup, 4> Fixups;
    raw_svector_ostream VecOS(Code);
    Emitter->encodeInstruction(Inst, VecOS, Fixups, *STI);
    return Code.size();
  }

  /// Compute the native code size for a range of instructions.
  /// Note: this can be imprecise wrt the final binary since happening prior to
  /// relaxation, as well as wrt the original binary because of opcode
  /// shortening.MCCodeEmitter is not thread safe and each thread should operate
  /// with its own copy of it.
  template <typename Itr>
  uint64_t computeCodeSize(Itr Beg, Itr End,
                           const MCCodeEmitter *Emitter = nullptr) const {
    uint64_t Size = 0;
    while (Beg != End) {
      if (!MIB->isPseudo(*Beg))
        Size += computeInstructionSize(*Beg, Emitter);
      ++Beg;
    }
    return Size;
  }

  /// Verify that assembling instruction \p Inst results in the same sequence of
  /// bytes as \p Encoding.
  bool validateEncoding(const MCInst &Instruction,
                        ArrayRef<uint8_t> Encoding) const;

  /// Return a function execution count threshold for determining whether
  /// the function is 'hot'. Consider it hot if count is above the average exec
  /// count of profiled functions.
  uint64_t getHotThreshold() const;

  /// Return true if instruction \p Inst requires an offset for further
  /// processing (e.g. assigning a profile).
  bool keepOffsetForInstruction(const MCInst &Inst) const {
    if (MIB->isCall(Inst) || MIB->isBranch(Inst) || MIB->isReturn(Inst) ||
        MIB->isPrefix(Inst) || MIB->isIndirectBranch(Inst)) {
      return true;
    }
    return false;
  }

  /// Return true if the function should be emitted to the output file.
  bool shouldEmit(const BinaryFunction &Function) const;

  /// Print the string name for a CFI operation.
  static void printCFI(raw_ostream &OS, const MCCFIInstruction &Inst);

  /// Print a single MCInst in native format.  If Function is non-null,
  /// the instruction will be annotated with CFI and possibly DWARF line table
  /// info.
  /// If printMCInst is true, the instruction is also printed in the
  /// architecture independent format.
  void printInstruction(raw_ostream &OS, const MCInst &Instruction,
                        uint64_t Offset = 0,
                        const BinaryFunction *Function = nullptr,
                        bool PrintMCInst = false, bool PrintMemData = false,
                        bool PrintRelocations = false) const;

  /// Print a range of instructions.
  template <typename Itr>
  uint64_t
  printInstructions(raw_ostream &OS, Itr Begin, Itr End, uint64_t Offset = 0,
                    const BinaryFunction *Function = nullptr,
                    bool PrintMCInst = false, bool PrintMemData = false,
                    bool PrintRelocations = false) const {
    while (Begin != End) {
      printInstruction(OS, *Begin, Offset, Function, PrintMCInst, PrintMemData,
                       PrintRelocations);
      Offset += computeCodeSize(Begin, Begin + 1);
      ++Begin;
    }
    return Offset;
  }

  void exitWithBugReport(StringRef Message,
                         const BinaryFunction &Function) const;

  struct IndependentCodeEmitter {
    std::unique_ptr<MCObjectFileInfo> LocalMOFI;
    std::unique_ptr<MCContext> LocalCtx;
    std::unique_ptr<MCCodeEmitter> MCE;
  };

  /// Encapsulates an independent MCCodeEmitter that doesn't share resources
  /// with the main one available through BinaryContext::MCE, managed by
  /// BinaryContext.
  /// This is intended to create a lock-free environment for an auxiliary thread
  /// that needs to perform work with an MCCodeEmitter that can be transient or
  /// won't be used in the main code emitter.
  IndependentCodeEmitter createIndependentMCCodeEmitter() const {
    IndependentCodeEmitter MCEInstance;
    MCEInstance.LocalCtx.reset(
        new MCContext(*TheTriple, AsmInfo.get(), MRI.get(), STI.get()));
    MCEInstance.LocalMOFI.reset(
        TheTarget->createMCObjectFileInfo(*MCEInstance.LocalCtx.get(),
                                          /*PIC=*/!HasFixedLoadAddress));
    MCEInstance.LocalCtx->setObjectFileInfo(MCEInstance.LocalMOFI.get());
    MCEInstance.MCE.reset(
        TheTarget->createMCCodeEmitter(*MII, *MCEInstance.LocalCtx));
    return MCEInstance;
  }

  /// Creating MCStreamer instance.
  std::unique_ptr<MCStreamer>
  createStreamer(llvm::raw_pwrite_stream &OS) const {
    MCCodeEmitter *MCE = TheTarget->createMCCodeEmitter(*MII, *Ctx);
    MCAsmBackend *MAB =
        TheTarget->createMCAsmBackend(*STI, *MRI, MCTargetOptions());
    std::unique_ptr<MCObjectWriter> OW = MAB->createObjectWriter(OS);
    std::unique_ptr<MCStreamer> Streamer(TheTarget->createMCObjectStreamer(
        *TheTriple, *Ctx, std::unique_ptr<MCAsmBackend>(MAB), std::move(OW),
        std::unique_ptr<MCCodeEmitter>(MCE), *STI,
        /* RelaxAll */ false,
        /* IncrementalLinkerCompatible */ false,
        /* DWARFMustBeAtTheEnd */ false));
    return Streamer;
  }
};

template <typename T, typename = std::enable_if_t<sizeof(T) == 1>>
inline raw_ostream &operator<<(raw_ostream &OS, const ArrayRef<T> &ByteArray) {
  const char *Sep = "";
  for (const auto Byte : ByteArray) {
    OS << Sep << format("%.2x", Byte);
    Sep = " ";
  }
  return OS;
}

} // namespace bolt
} // namespace llvm

#endif
