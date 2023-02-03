//===- bolt/Rewrite/RewriteInstance.h - ELF rewriter ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to control an instance of a binary rewriting process.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_REWRITE_INSTANCE_H
#define BOLT_REWRITE_REWRITE_INSTANCE_H

#include "bolt/Core/BinaryContext.h"
#include "bolt/Utils/NameResolver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include <map>
#include <set>
#include <unordered_map>

namespace llvm {

class ToolOutputFile;

namespace bolt {

class BoltAddressTranslation;
class CFIReaderWriter;
class DWARFRewriter;
class ProfileReaderBase;

/// This class encapsulates all data necessary to carry on binary reading,
/// disassembly, CFG building, BB reordering (among other binary-level
/// optimizations) and rewriting. It also has the logic to coordinate such
/// events.
class RewriteInstance {
public:
  // This constructor has complex initialization that can fail during
  // construction. Constructors can’t return errors, so clients must test \p Err
  // after the object is constructed. Use `create` method instead.
  RewriteInstance(llvm::object::ELFObjectFileBase *File, const int Argc,
                  const char *const *Argv, StringRef ToolPath, Error &Err);

  static Expected<std::unique_ptr<RewriteInstance>>
  create(llvm::object::ELFObjectFileBase *File, const int Argc,
         const char *const *Argv, StringRef ToolPath);
  ~RewriteInstance();

  /// Assign profile from \p Filename to this instance.
  Error setProfile(StringRef Filename);

  /// Run all the necessary steps to read, optimize and rewrite the binary.
  Error run();

  /// Diff this instance against another one. Non-const since we may run passes
  /// to fold identical functions.
  void compare(RewriteInstance &RI2);

  /// Return binary context.
  const BinaryContext &getBinaryContext() const { return *BC; }

  /// Return total score of all functions for this instance.
  uint64_t getTotalScore() const { return BC->TotalScore; }

  /// Return the name of the input file.
  StringRef getInputFilename() const {
    assert(InputFile && "cannot have an instance without a file");
    return InputFile->getFileName();
  }

  /// Set the build-id string if we did not fail to parse the contents of the
  /// ELF note section containing build-id information.
  void parseBuildID();

  /// The build-id is typically a stream of 20 bytes. Return these bytes in
  /// printable hexadecimal form if they are available, or std::nullopt
  /// otherwise.
  std::optional<std::string> getPrintableBuildID() const;

  /// If this instance uses a profile, return appropriate profile reader.
  const ProfileReaderBase *getProfileReader() const {
    return ProfileReader.get();
  }

private:
  using ELF64LEPhdrTy = object::ELF64LEFile::Elf_Phdr;

  /// Populate array of binary functions and other objects of interest
  /// from meta data in the file.
  void discoverFileObjects();

  /// Read info from special sections. E.g. eh_frame and .gcc_except_table
  /// for exception and stack unwinding information.
  Error readSpecialSections();

  /// Adjust supplied command-line options based on input data.
  void adjustCommandLineOptions();

  /// Process runtime relocations.
  void processDynamicRelocations();

  /// Process input relocations.
  void processRelocations();

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

  /// Read relocations from a given section.
  void readDynamicRelocations(const object::SectionRef &Section, bool IsJmpRel);

  /// Print relocation information.
  void printRelocationInfo(const RelocationRef &Rel, StringRef SymbolName,
                           uint64_t SymbolAddress, uint64_t Addend,
                           uint64_t ExtractedValue) const;

  /// Read relocations from a given section.
  void readRelocations(const object::SectionRef &Section);

  /// Handle one relocation.
  void handleRelocation(const object::SectionRef &RelocatedSection,
                        const RelocationRef &Rel);

  /// Mark functions that are not meant for processing as ignored.
  void selectFunctionsToProcess();

  /// Read information from debug sections.
  void readDebugInfo();

  /// Read profile data without having disassembled functions available.
  void preprocessProfileData();

  void processProfileDataPreCFG();

  /// Associate profile data with functions and data objects.
  void processProfileData();

  /// Disassemble each function in the binary and associate it with a
  /// BinaryFunction object, preparing all information necessary for binary
  /// optimization.
  void disassembleFunctions();

  void buildFunctionsCFG();

  void postProcessFunctions();

  void preregisterSections();

  /// Run optimizations that operate at the binary, or post-linker, level.
  void runOptimizationPasses();

  /// Write code and data into an intermediary object file, map virtual to real
  /// addresses and link the object file, resolving all relocations and
  /// performing final relaxation.
  void emitAndLink();

  /// Link additional runtime code to support instrumentation.
  void linkRuntime();

  /// Update debug and other auxiliary information in the file.
  void updateMetadata();

  /// Update SDTMarkers' locations for the output binary.
  void updateSDTMarkers();

  /// Update LKMarkers' locations for the output binary.
  void updateLKMarkers();

  /// Update address of MCDecodedPseudoProbe.
  void updatePseudoProbes();

  /// Encode MCDecodedPseudoProbe.
  void encodePseudoProbes();

  /// Return the list of code sections in the output order.
  std::vector<BinarySection *> getCodeSections();

  /// Map all sections to their final addresses.
  void mapFileSections(RuntimeDyld &RTDyld);

  /// Map code sections generated by BOLT.
  void mapCodeSections(RuntimeDyld &RTDyld);

  /// Map the rest of allocatable sections.
  void mapAllocatableSections(RuntimeDyld &RTDyld);

  /// Update output object's values based on the final \p Layout.
  void updateOutputValues(const MCAsmLayout &Layout);

  /// Rewrite back all functions (hopefully optimized) that fit in the original
  /// memory footprint for that function. If the function is now larger and does
  /// not fit in the binary, reject it and preserve the original version of the
  /// function. If we couldn't understand the function for some reason in
  /// disassembleFunctions(), also preserve the original version.
  void rewriteFile();

  /// Return address of a function in the new binary corresponding to
  /// \p OldAddress address in the original binary.
  uint64_t getNewFunctionAddress(uint64_t OldAddress);

  /// Return address of a function or moved data in the new binary
  /// corresponding to \p OldAddress address in the original binary.
  uint64_t getNewFunctionOrDataAddress(uint64_t OldAddress);

  /// Return value for the symbol \p Name in the output.
  uint64_t getNewValueForSymbol(const StringRef Name);

  /// Detect addresses and offsets available in the binary for allocating
  /// new sections.
  Error discoverStorage();

  /// Adjust function sizes and set proper maximum size values after the whole
  /// symbol table has been processed.
  void adjustFunctionBoundaries();

  /// Make .eh_frame section relocatable.
  void relocateEHFrameSection();

  /// Analyze relocation \p Rel.
  /// Return true if the relocation was successfully processed, false otherwise.
  /// The \p SymbolName, \p SymbolAddress, \p Addend and \p ExtractedValue
  /// parameters will be set on success. The \p Skip argument indicates
  /// that the relocation was analyzed, but it must not be processed.
  bool analyzeRelocation(const object::RelocationRef &Rel, uint64_t &RType,
                         std::string &SymbolName, bool &IsSectionRelocation,
                         uint64_t &SymbolAddress, int64_t &Addend,
                         uint64_t &ExtractedValue, bool &Skip) const;

  /// Rewrite non-allocatable sections with modifications.
  void rewriteNoteSections();

  /// Write .eh_frame_hdr.
  void writeEHFrameHeader();

  /// Disassemble and create function entries for PLT.
  void disassemblePLT();

  /// Auxiliary function to create .plt BinaryFunction on \p EntryAddres
  /// with the \p EntrySize size. \p TargetAddress is the .got entry
  /// associated address.
  void createPLTBinaryFunction(uint64_t TargetAddress, uint64_t EntryAddress,
                               uint64_t EntrySize);

  /// Disassemble aarch64-specific .plt \p Section auxiliary function
  void disassemblePLTSectionAArch64(BinarySection &Section);

  /// Disassemble X86-specific .plt \p Section auxiliary function. \p EntrySize
  /// is the expected .plt \p Section entry function size.
  void disassemblePLTSectionX86(BinarySection &Section, uint64_t EntrySize);

  /// ELF-specific part. TODO: refactor into new class.
#define ELF_FUNCTION(TYPE, FUNC)                                               \
  template <typename ELFT> TYPE FUNC(object::ELFObjectFile<ELFT> *Obj);        \
  TYPE FUNC() {                                                                \
    if (auto *ELF32LE = dyn_cast<object::ELF32LEObjectFile>(InputFile))        \
      return FUNC(ELF32LE);                                                    \
    if (auto *ELF64LE = dyn_cast<object::ELF64LEObjectFile>(InputFile))        \
      return FUNC(ELF64LE);                                                    \
    if (auto *ELF32BE = dyn_cast<object::ELF32BEObjectFile>(InputFile))        \
      return FUNC(ELF32BE);                                                    \
    auto *ELF64BE = cast<object::ELF64BEObjectFile>(InputFile);                \
    return FUNC(ELF64BE);                                                      \
  }

  /// Patch ELF book-keeping info.
  void patchELFPHDRTable();

  /// Create section header table.
  ELF_FUNCTION(void, patchELFSectionHeaderTable);

  /// Create the regular symbol table and patch dyn symbol tables.
  ELF_FUNCTION(void, patchELFSymTabs);

  /// Read dynamic section/segment of ELF.
  ELF_FUNCTION(Error, readELFDynamic);

  /// Patch dynamic section/segment of ELF.
  ELF_FUNCTION(void, patchELFDynamic);

  /// Patch .got
  ELF_FUNCTION(void, patchELFGOT);

  /// Patch allocatable relocation sections.
  ELF_FUNCTION(void, patchELFAllocatableRelaSections);

  /// Finalize memory image of section header string table.
  ELF_FUNCTION(void, finalizeSectionStringTable);

  /// Return a list of all sections to include in the output binary.
  /// Populate \p NewSectionIndex with a map of input to output indices.
  template <typename ELFT>
  std::vector<typename object::ELFObjectFile<ELFT>::Elf_Shdr>
  getOutputSections(object::ELFObjectFile<ELFT> *File,
                    std::vector<uint32_t> &NewSectionIndex);

  /// Return true if \p Section should be stripped from the output binary.
  template <typename ELFShdrTy>
  bool shouldStrip(const ELFShdrTy &Section, StringRef SectionName);

  /// Write ELF symbol table using \p Write and \p AddToStrTab functions
  /// based on the input file symbol table passed in \p SymTabSection.
  /// \p IsDynSym is set to true for dynamic symbol table since we
  /// are updating it in-place with minimal modifications.
  template <typename ELFT, typename WriteFuncTy, typename StrTabFuncTy>
  void updateELFSymbolTable(
      object::ELFObjectFile<ELFT> *File, bool IsDynSym,
      const typename object::ELFObjectFile<ELFT>::Elf_Shdr &SymTabSection,
      const std::vector<uint32_t> &NewSectionIndex, WriteFuncTy Write,
      StrTabFuncTy AddToStrTab);

  /// Get output index in dynamic symbol table.
  uint32_t getOutputDynamicSymbolIndex(const MCSymbol *Symbol) {
    auto It = SymbolIndex.find(Symbol);
    if (It != SymbolIndex.end())
      return It->second;
    return 0;
  }

  /// Add a notes section containing the BOLT revision and command line options.
  void addBoltInfoSection();

  /// Add a notes section containing the serialized BOLT Address Translation
  /// maps that can be used to enable sampling of the output binary for the
  /// purpose of generating BOLT profile data for the input binary.
  void addBATSection();

  /// Loop over now emitted functions to write translation maps
  void encodeBATSection();

  /// Update the ELF note section containing the binary build-id to reflect
  /// a new build-id, so tools can differentiate between the old and the
  /// rewritten binary.
  void patchBuildID();

  /// Return file offset corresponding to a given virtual address.
  uint64_t getFileOffsetFor(uint64_t Address) {
    assert(Address >= NewTextSegmentAddress &&
           "address in not in the new text segment");
    return Address - NewTextSegmentAddress + NewTextSegmentOffset;
  }

  /// Return file offset corresponding to a virtual \p Address.
  /// Return 0 if the address has no mapping in the file, including being
  /// part of .bss section.
  uint64_t getFileOffsetForAddress(uint64_t Address) const;

  /// Return true if we will overwrite contents of the section instead
  /// of appending contents to it.
  bool willOverwriteSection(StringRef SectionName);

  /// Parse .note.stapsdt section
  void parseSDTNotes();

  /// Parse .pseudo_probe_desc section and .pseudo_probe section
  /// Setup Pseudo probe decoder
  void parsePseudoProbe();

  /// Print all SDT markers
  void printSDTMarkers();

public:
  /// Standard ELF sections we overwrite.
  static constexpr const char *SectionsToOverwrite[] = {
      ".shstrtab",
      ".symtab",
      ".strtab",
  };

  /// Debug section to we overwrite while updating the debug info.
  static std::vector<std::string> DebugSectionsToOverwrite;

  /// Return true if the section holds debug information.
  static bool isDebugSection(StringRef SectionName);

  /// Return true if the section holds linux kernel symbol information.
  static bool isKSymtabSection(StringRef SectionName);

  /// Adds Debug section to overwrite.
  static void addToDebugSectionsToOverwrite(const char *Section) {
    DebugSectionsToOverwrite.emplace_back(Section);
  }

private:
  /// Get the contents of the LSDA section for this binary.
  ArrayRef<uint8_t> getLSDAData();

  /// Get the mapped address of the LSDA section for this binary.
  uint64_t getLSDAAddress();

  static const char TimerGroupName[];

  static const char TimerGroupDesc[];

  /// Alignment value used for .eh_frame_hdr.
  static constexpr uint64_t EHFrameHdrAlign = 4;

  /// Sections created by BOLT will have an internal name that starts with the
  /// following prefix. Note that the prefix is used for a section lookup
  /// internally and the section name in the output might be different.
  static StringRef getNewSecPrefix() { return ".bolt.new"; }

  /// String to be added before the original section name.
  ///
  /// When BOLT creates a new section with the same name as the one in the
  /// input file, it may need to preserve the original section. This prefix
  /// will be added to the name of the original section.
  static StringRef getOrgSecPrefix() { return ".bolt.org"; }

  /// Section name used for extra BOLT code in addition to .text.
  static StringRef getBOLTTextSectionName() { return ".bolt.text"; }

  /// Common section names.
  static StringRef getEHFrameSectionName() { return ".eh_frame"; }

  /// An instance of the input binary we are processing, externally owned.
  llvm::object::ELFObjectFileBase *InputFile;

  /// Command line args used to process binary.
  const int Argc;
  const char *const *Argv;
  StringRef ToolPath;

  std::unique_ptr<ProfileReaderBase> ProfileReader;

  std::unique_ptr<BinaryContext> BC;
  std::unique_ptr<CFIReaderWriter> CFIRdWrt;

  // Run ExecutionEngine linker with custom memory manager and symbol resolver.
  std::unique_ptr<RuntimeDyld> RTDyld;

  /// Output file where we mix original code from the input binary and
  /// optimized code for selected functions.
  std::unique_ptr<ToolOutputFile> Out;

  /// Offset in the input file where non-allocatable sections start.
  uint64_t FirstNonAllocatableOffset{0};

  /// Information about program header table.
  uint64_t PHDRTableAddress{0};
  uint64_t PHDRTableOffset{0};
  unsigned Phnum{0};

  /// New code segment info.
  uint64_t NewTextSegmentAddress{0};
  uint64_t NewTextSegmentOffset{0};
  uint64_t NewTextSegmentSize{0};

  /// Track next available address for new allocatable sections.
  uint64_t NextAvailableAddress{0};

  /// Location and size of dynamic relocations.
  std::optional<uint64_t> DynamicRelocationsAddress;
  uint64_t DynamicRelocationsSize{0};
  uint64_t DynamicRelativeRelocationsCount{0};

  /// PLT relocations are special kind of dynamic relocations stored separately.
  std::optional<uint64_t> PLTRelocationsAddress;
  uint64_t PLTRelocationsSize{0};

  /// True if relocation of specified type came from .rela.plt
  DenseMap<uint64_t, bool> IsJmpRelocation;

  /// Index of specified symbol in the dynamic symbol table. NOTE Currently it
  /// is filled and used only with the relocations-related symbols.
  std::unordered_map<const MCSymbol *, uint32_t> SymbolIndex;

  /// Store all non-zero symbols in this map for a quick address lookup.
  std::map<uint64_t, llvm::object::SymbolRef> FileSymRefs;

  std::unique_ptr<DWARFRewriter> DebugInfoRewriter;

  std::unique_ptr<BoltAddressTranslation> BAT;

  /// Number of local symbols in newly written symbol table.
  uint64_t NumLocalSymbols{0};

  /// Information on special Procedure Linkage Table sections. There are
  /// multiple variants generated by different linkers.
  struct PLTSectionInfo {
    const char *Name;
    uint64_t EntrySize{0};
  };

  /// Different types of X86-64 PLT sections.
  const PLTSectionInfo X86_64_PLTSections[4] = {
      { ".plt", 16 },
      { ".plt.got", 8 },
      { ".plt.sec", 8 },
      { nullptr, 0 }
  };

  /// AArch64 PLT sections.
  const PLTSectionInfo AArch64_PLTSections[3] = {
      {".plt"}, {".iplt"}, {nullptr}};

  /// Return PLT information for a section with \p SectionName or nullptr
  /// if the section is not PLT.
  const PLTSectionInfo *getPLTSectionInfo(StringRef SectionName) {
    const PLTSectionInfo *PLTSI = nullptr;
    switch (BC->TheTriple->getArch()) {
    default:
      break;
    case Triple::x86_64:
      PLTSI = X86_64_PLTSections;
      break;
    case Triple::aarch64:
      PLTSI = AArch64_PLTSections;
      break;
    }
    for (; PLTSI && PLTSI->Name; ++PLTSI)
      if (SectionName == PLTSI->Name)
        return PLTSI;

    return nullptr;
  }

  /// Exception handling and stack unwinding information in this binary.
  ErrorOr<BinarySection &> LSDASection{std::errc::bad_address};
  ErrorOr<BinarySection &> EHFrameSection{std::errc::bad_address};

  /// .got.plt sections.
  ///
  /// Contains jump slots (addresses) indirectly referenced by
  /// instructions in .plt section.
  ErrorOr<BinarySection &> GOTPLTSection{std::errc::bad_address};

  /// .rela.plt section.
  ///
  /// Contains relocations against .got.plt.
  ErrorOr<BinarySection &> RelaPLTSection{std::errc::bad_address};
  ErrorOr<BinarySection &> RelaDynSection{std::errc::bad_address};

  /// .note.gnu.build-id section.
  ErrorOr<BinarySection &> BuildIDSection{std::errc::bad_address};

  /// .note.stapsdt section.
  /// Contains information about statically defined tracing points
  ErrorOr<BinarySection &> SDTSection{std::errc::bad_address};

  /// .pseudo_probe_desc section.
  /// Contains information about pseudo probe description, like its related
  /// function
  ErrorOr<BinarySection &> PseudoProbeDescSection{std::errc::bad_address};

  /// .pseudo_probe section.
  /// Contains information about pseudo probe details, like its address
  ErrorOr<BinarySection &> PseudoProbeSection{std::errc::bad_address};

  /// Helper for accessing sections by name.
  BinarySection *getSection(const Twine &Name) {
    ErrorOr<BinarySection &> ErrOrSection = BC->getUniqueSectionByName(Name);
    return ErrOrSection ? &ErrOrSection.get() : nullptr;
  }

  /// A reference to the build-id bytes in the original binary
  StringRef BuildID;

  /// Keep track of functions we fail to write in the binary. We need to avoid
  /// rewriting CFI info for these functions.
  std::vector<uint64_t> FailedAddresses;

  /// Keep track of which functions didn't fit in their original space in the
  /// last emission, so that we may either decide to split or not optimize them.
  std::set<uint64_t> LargeFunctions;

  /// Section header string table.
  StringTableBuilder SHStrTab;

  /// A rewrite of strtab
  std::string NewStrTab;

  /// Number of processed to data relocations.  Used to implement the
  /// -max-relocations debugging option.
  uint64_t NumDataRelocations{0};

  /// Number of failed to process relocations.
  uint64_t NumFailedRelocations{0};

  NameResolver NR;

  friend class RewriteInstanceDiff;
};

MCPlusBuilder *createMCPlusBuilder(const Triple::ArchType Arch,
                                   const MCInstrAnalysis *Analysis,
                                   const MCInstrInfo *Info,
                                   const MCRegisterInfo *RegInfo);

} // namespace bolt
} // namespace llvm

#endif
