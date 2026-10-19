//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_CONFIG_H
#define LLD_ELF_CONFIG_H

#include "lld/Common/BPSectionOrdererBase.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorWithFlags.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CachePruning.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/TarWriter.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace lld::elf {

class InputFile;
class BinaryFile;
class BitcodeFile;
class ELFFileBase;
class SharedFile;
class InputSectionBase;
class EhInputSection;
class Defined;
class Undefined;
class Symbol;
class SymbolTable;
class BitcodeCompiler;
class OutputSection;
class LinkerScript;
class TargetInfo;
struct Ctx;
struct PhdrEntry;

class ARMExidxSyntheticSection;
class BssSection;
class BuildIdSection;
class EhFrameHeader;
class EhFrameSection;
class GdbIndexSection;
class GnuHashTableSection;
class GotPltSection;
class GotSection;
class HashTableSection;
class IgotPltSection;
class InputSection;
class IpltSection;
class MemtagAndroidNote;
class MemtagGlobalDescriptors;
class MipsGotSection;
class PPC64LongBranchTargetSection;
class PackageMetadataNote;
class PltSection;
class RelocationBaseSection;
class RelrBaseSection;
class RelroPaddingSection;
class StringTableSection;
class SymbolTableBaseSection;
class SymtabShndxSection;
class SyntheticSection;
class VersionDefinitionSection;
class VersionTableSection;

enum ELFKind : uint8_t {
  ELFNoneKind,
  ELF32LEKind,
  ELF32BEKind,
  ELF64LEKind,
  ELF64BEKind
};

// For -Bno-symbolic, -Bsymbolic-non-weak-functions, -Bsymbolic-functions,
// -Bsymbolic-non-weak, -Bsymbolic.
enum class BsymbolicKind { None, NonWeakFunctions, Functions, NonWeak, All };

// For --build-id.
enum class BuildIdKind { None, Fast, Md5, Sha1, Hexstring, Uuid };

// For --call-graph-profile-sort={none,hfsort,cdsort}.
enum class CGProfileSortKind { None, Hfsort, Cdsort };

// For --discard-{all,locals,none}.
enum class DiscardPolicy { Default, All, Locals, None };

// For --icf={none,safe,all}.
enum class ICFLevel { None, Safe, All };

// For --strip-{all,debug}.
enum class StripPolicy { None, All, Debug };

// For --unresolved-symbols.
enum class UnresolvedPolicy { ReportError, Warn, Ignore };

// For --orphan-handling.
enum class OrphanHandlingPolicy { Place, Warn, Error };

// For --sort-section and linkerscript sorting rules.
enum class SortSectionPolicy {
  Default,
  None,
  Alignment,
  Name,
  Priority,
  Reverse,
};

// For --target2
enum class Target2Policy { Abs, Rel, GotRel };

// For tracking ARM Float Argument PCS
enum class ARMVFPArgKind { Default, Base, VFP, ToolChain };

// For -z noseparate-code, -z separate-code and -z separate-loadable-segments.
enum class SeparateSegmentKind { None, Code, Loadable };

// For -z *stack
enum class GnuStackKind { None, Exec, NoExec };

// For --lto=
enum LtoKind : uint8_t {UnifiedThin, UnifiedRegular, Default};

// For -z gcs=
enum class GcsPolicy { Implicit, Never, Always };

// For -z zicfilp=
enum class ZicfilpPolicy { Implicit, Never, Unlabeled, FuncSig };

// For -z zicfiss=
enum class ZicfissPolicy { Implicit, Never, Always };

// For some options that resemble -z bti-report={none,warning,error}
enum class ReportPolicy { None, Warning, Error };

// Describes the signing schema for a file using the PAuth ABI extension.
// Two files are considered compatible when both `platform` and `version` match.
// The pair (0, 0) is reserved to indicate incompatibility with the PAuth ABI.
struct AArch64PauthAbiCoreInfo {
  uint64_t platform;
  uint64_t version;
  // Returns true if the core info is not the reserved (0, 0) value.
  bool isValid() const { return platform || version; }
  static constexpr size_t size() { return sizeof(platform) + sizeof(version); }
  bool operator==(const AArch64PauthAbiCoreInfo &other) const {
    return platform == other.platform && version == other.version;
  }
  bool operator!=(const AArch64PauthAbiCoreInfo &other) const {
    return !(*this == other);
  }
};

struct SymbolVersion {
  llvm::StringRef name;
  bool isExternCpp;
  bool hasWildcard;
};

// This struct contains symbols version definition that
// can be found in version script if it is used for link.
struct VersionDefinition {
  llvm::StringRef name;
  uint16_t id;
  SmallVector<SymbolVersion, 0> nonLocalPatterns;
  SmallVector<SymbolVersion, 0> localPatterns;
};

// Deferred file-load job: one per input, expanded by loadFiles().
struct LoadJob {
  enum Kind : uint8_t { Obj, Bitcode, Archive, Shared, Binary };
  LoadJob(llvm::MemoryBufferRef mbref, llvm::StringRef path, Kind kind,
          bool inWholeArchive, bool lazy, bool asNeeded, bool withLOption,
          uint32_t groupId);
  llvm::MemoryBufferRef mbref;
  llvm::StringRef path;
  Kind kind;
  bool getInWholeArchive() const { return tarEntries.getFlag(0); }
  void setInWholeArchive(bool Value) { tarEntries.setFlag(0, Value); }
  bool getLazy() const { return tarEntries.getFlag(1); }
  void setLazy(bool Value) { tarEntries.setFlag(1, Value); }
  bool getAsNeeded() const { return tarEntries.getFlag(2); }
  void setAsNeeded(bool Value) { tarEntries.setFlag(2, Value); }
  bool getWithLOption() const { return tarEntries.getFlag(3); }
  void setWithLOption(bool Value) { tarEntries.setFlag(3, Value); }
  uint32_t groupId;
  SmallVector<std::unique_ptr<InputFile>, 0> out;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> thinBufs;
  llvm::SmallVectorWithFlags<std::pair<std::string, llvm::StringRef>, 0, 4>
      tarEntries;
};

class LinkerDriver {
public:
  LinkerDriver(Ctx &ctx);
  LinkerDriver(LinkerDriver &) = delete;
  void linkerMain(ArrayRef<const char *> args);
  void addFile(StringRef path, bool withLOption);
  void addFile(std::unique_ptr<ELFFileBase> ef);
  void addLibrary(StringRef name);

private:
  Ctx &ctx;
  void createFiles(llvm::opt::InputArgList &args);
  void loadFiles();
  void inferMachineType();
  void waitForLTOCleanup();
  template <class ELFT> void link(llvm::opt::InputArgList &args);
  template <class ELFT> void compileBitcodeFiles(bool skipLinkedOutput);
  // True if we are in --whole-archive and --no-whole-archive.
  bool inWholeArchive = false;

  // True if we are in --start-lib and --end-lib.
  bool inLib = false;

  // True inside createFiles(): defers to loadFiles().
  bool deferLoad = false;
  SmallVector<LoadJob, 0> loadJobs;

  std::unique_ptr<BitcodeCompiler> lto;
  SmallVector<std::unique_ptr<InputFile>, 0> files, ltoObjectFiles;

public:
  // See InputFile::groupId.
  uint32_t nextGroupId;
  bool isInGroup;
  std::unique_ptr<InputFile> armCmseImpLib;
  SmallVector<std::pair<StringRef, unsigned>, 0> archiveFiles;
};

// This struct contains the global configuration for the linker.
// Most fields are direct mapping from the command line options
// and such fields have the same name as the corresponding options.
// Most fields are initialized by the ctx.driver.
struct Config {
  uint8_t osabi = 0;
  uint32_t andFeatures = 0;
  llvm::CachePruningPolicy thinLTOCachePolicy;
  llvm::SetVector<llvm::CachedHashString> dependencyFiles; // for --dependency-file
  llvm::StringMap<uint64_t> sectionStartMap;
  llvm::StringRef bfdname;
  llvm::StringRef chroot;
  llvm::StringRef dependencyFile;
  llvm::StringRef dwoDir;
  llvm::StringRef dynamicLinker;
  llvm::StringRef entry;
  llvm::StringRef emulation;
  llvm::StringRef fini;
  llvm::StringRef init;
  llvm::StringRef ltoAAPipeline;
  llvm::StringRef ltoCSProfileFile;
  llvm::StringRef ltoNewPmPasses;
  llvm::StringRef ltoObjPath;
  llvm::StringRef ltoSampleProfile;
  llvm::StringRef mapFile;
  llvm::StringRef outputFile;
  llvm::StringRef optRemarksFilename;
  std::optional<uint64_t> optRemarksHotnessThreshold = 0;
  llvm::StringRef optRemarksPasses;
  llvm::StringRef optRemarksFormat;
  llvm::StringRef optStatsFilename;
  llvm::StringRef progName;
  llvm::StringRef printArchiveStats;
  llvm::StringRef printSymbolOrder;
  llvm::StringRef soName;
  llvm::StringRef sysroot;
  llvm::StringRef thinLTOCacheDir;
  llvm::StringRef thinLTOIndexOnlyArg;
  llvm::StringRef whyExtract;
  llvm::SmallVector<llvm::GlobPattern, 0> whyLive;
  llvm::StringRef cmseInputLib;
  llvm::StringRef cmseOutputLib;
  ReportPolicy zBtiReport = ReportPolicy::None;
  llvm::StringRef zBtiReportSource;
  ReportPolicy zCetReport = ReportPolicy::None;
  ReportPolicy zPauthReport = ReportPolicy::None;
  ReportPolicy zGcsReport = ReportPolicy::None;
  llvm::StringRef zGcsReportSource;
  ReportPolicy zGcsReportDynamic = ReportPolicy::None;
  ReportPolicy zExecuteOnlyReport = ReportPolicy::None;
  ReportPolicy zZicfilpUnlabeledReport = ReportPolicy::None;
  ReportPolicy zZicfilpFuncSigReport = ReportPolicy::None;
  ReportPolicy zZicfissReport = ReportPolicy::None;
  bool ltoBBAddrMap;
  llvm::StringRef ltoBasicBlockSections;
  std::pair<llvm::StringRef, llvm::StringRef> thinLTOObjectSuffixReplace;
  llvm::StringRef thinLTOPrefixReplaceOld;
  llvm::StringRef thinLTOPrefixReplaceNew;
  llvm::StringRef thinLTOPrefixReplaceNativeObject;
  std::string rpath;
  llvm::SmallVectorWithFlags<VersionDefinition, 0, 4> versionDefinitions;
  std::optional<llvm::DenseSet<llvm::StringRef>> retainSymbols;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> auxiliaryList;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> filterList;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> passPlugins;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> searchPaths;
  llvm::SmallVector<llvm::StringRef, 0> symbolOrderingFile;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> thinLTOModulesToCompile;
  llvm::StringRef dtltoDistributor;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> dtltoDistributorArgs;
  llvm::StringRef dtltoCompiler;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> dtltoCompilerPrependArgs;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> dtltoCompilerArgs;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> undefined;
  llvm::SmallVectorWithFlags<SymbolVersion, 0, 4> dynamicList;
  llvm::SmallVector<uint8_t, 0> buildIdVector;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> mllvmOpts;
  llvm::MapVector<std::pair<const InputSectionBase *, const InputSectionBase *>,
                  uint64_t>
      callGraphProfile;
  bool cmseImplib = false;
  bool allowMultipleDefinition;
  bool fatLTOObjects;
  bool androidPackDynRelocs = false;
  bool armHasArmISA = false;
  bool armHasThumb2ISA = false;
  bool armHasBlx = false;
  bool armHasMovtMovw = false;
  bool armJ1J2BranchEncoding = false;
  bool getArmCMSESupport() const { return remapInputsWildcards.getFlag(0); }
  bool setArmCMSESupport(bool Value) {
    remapInputsWildcards.setFlag(0, Value);
    return Value;
  }
  bool asNeeded = false;
  bool getArmBe8() const { return remapInputsWildcards.getFlag(1); }
  void setArmBe8(bool Value) { remapInputsWildcards.setFlag(1, Value); }
  BsymbolicKind bsymbolic = BsymbolicKind::None;
  CGProfileSortKind callGraphProfileSort;
  llvm::StringRef irpgoProfilePath;
  bool getBpStartupFunctionSort() const {
    return remapInputsWildcards.getFlag(2);
  }
  void setBpStartupFunctionSort(bool Value) {
    remapInputsWildcards.setFlag(2, Value);
  }
  bool getBpCompressionSortStartupFunctions() const {
    return remapInputsWildcards.getFlag(3);
  }
  void setBpCompressionSortStartupFunctions(bool Value) {
    remapInputsWildcards.setFlag(3, Value);
  }
  bool getBpFunctionOrderForCompression() const {
    return packageMetadata.getFlag(0);
  }
  void setBpFunctionOrderForCompression(bool Value) {
    packageMetadata.setFlag(0, Value);
  }
  bool getBpDataOrderForCompression() const {
    return packageMetadata.getFlag(1);
  }
  void setBpDataOrderForCompression(bool Value) {
    packageMetadata.setFlag(1, Value);
  }
  llvm::SmallVectorWithFlags<BPCompressionSortSpec,
                             llvm::CalculateSmallVectorDefaultInlinedElements<
                                 BPCompressionSortSpec>::value,
                             4>
      bpCompressionSortSpecs;
  bool getBpVerboseSectionOrderer() const { return packageMetadata.getFlag(2); }
  void setBpVerboseSectionOrderer(bool Value) {
    packageMetadata.setFlag(2, Value);
  }
  bool getBranchToBranch() const { return packageMetadata.getFlag(3); }
  void setBranchToBranch(bool Value) { packageMetadata.setFlag(3, Value); }
  bool getCheckSections() const { return warnBackrefsExclude.getFlag(0); }
  void setCheckSections(bool Value) { warnBackrefsExclude.setFlag(0, Value); }
  bool getCheckDynamicRelocs() const { return warnBackrefsExclude.getFlag(1); }
  void setCheckDynamicRelocs(bool Value) {
    warnBackrefsExclude.setFlag(1, Value);
  }
  std::optional<llvm::DebugCompressionType> compressDebugSections;
  llvm::SmallVectorWithFlags<
      std::tuple<llvm::GlobPattern, llvm::DebugCompressionType, unsigned>, 0, 4>
      compressSections;
  bool getCref() const { return warnBackrefsExclude.getFlag(2); }
  void setCref(bool Value) { warnBackrefsExclude.setFlag(2, Value); }
  llvm::SmallVectorWithFlags<std::pair<llvm::GlobPattern, uint64_t>, 0, 4>
      deadRelocInNonAlloc;
  bool debugNames;
  bool demangle = true;
  bool dependentLibraries;
  bool disableVerify;
  bool ehFrameHdr;
  bool emitLLVM;
  bool emitRelocs;
  bool enableNewDtags;
  bool enableNonContiguousRegions;
  bool executeOnly;
  bool exportDynamic;
  bool fixCortexA53Errata843419;
  bool fixCortexA8;
  bool formatBinary = false;
  bool gcSections;
  bool gdbIndex;
  bool gnuHash = false;
  bool gnuUnique;
  bool ignoreDataAddressEquality;
  bool ignoreFunctionAddressEquality;
  bool ltoCSProfileGenerate;
  bool ltoPGOWarnMismatch;
  bool ltoDebugPassManager;
  bool ltoEmitAsm;
  bool ltoUniqueBasicBlockSectionNames;
  bool ltoValidateAllVtablesHaveTypeInfos;
  bool ltoWholeProgramVisibility;
  bool mergeArmExidx;
  bool mipsN32Abi = false;
  bool mmapOutputFile;
  bool nmagic;
  bool noinhibitExec;
  bool getNostdlib() const { return warnBackrefsExclude.getFlag(3); }
  void setNostdlib(bool Value) { warnBackrefsExclude.setFlag(3, Value); }
  bool getOFormatBinary() const { return shuffleSections.getFlag(0); }
  void setOFormatBinary(bool Value) { shuffleSections.setFlag(0, Value); }
  bool getOmagic() const { return shuffleSections.getFlag(1); }
  void setOmagic(bool Value) { shuffleSections.setFlag(1, Value); }
  bool getOptEB() const { return shuffleSections.getFlag(2); }
  bool setOptEB(bool Value) {
    shuffleSections.setFlag(2, Value);
    return Value;
  }
  bool getOptEL() const { return shuffleSections.getFlag(3); }
  bool setOptEL(bool Value) {
    shuffleSections.setFlag(3, Value);
    return Value;
  }
  bool getOptimizeBBJumps() const { return deadRelocInNonAlloc.getFlag(0); }
  void setOptimizeBBJumps(bool Value) { deadRelocInNonAlloc.setFlag(0, Value); }
  bool getOptRemarksWithHotness() const {
    return deadRelocInNonAlloc.getFlag(1);
  }
  void setOptRemarksWithHotness(bool Value) {
    deadRelocInNonAlloc.setFlag(1, Value);
  }
  bool getPicThunk() const { return deadRelocInNonAlloc.getFlag(2); }
  void setPicThunk(bool Value) { deadRelocInNonAlloc.setFlag(2, Value); }
  bool getPie() const { return deadRelocInNonAlloc.getFlag(3); }
  void setPie(bool Value) { deadRelocInNonAlloc.setFlag(3, Value); }
  llvm::StringRef printGcSections;
  bool getPrintIcfSections() const { return compressSections.getFlag(0); }
  void setPrintIcfSections(bool Value) { compressSections.setFlag(0, Value); }
  bool getPrintMemoryUsage() const { return compressSections.getFlag(1); }
  void setPrintMemoryUsage(bool Value) { compressSections.setFlag(1, Value); }
  std::optional<uint64_t> randomizeSectionPadding;
  bool getRejectMismatch() const { return compressSections.getFlag(2); }
  void setRejectMismatch(bool Value) { compressSections.setFlag(2, Value); }
  bool getRelax() const { return compressSections.getFlag(3); }
  void setRelax(bool Value) { compressSections.setFlag(3, Value); }
  bool getRelaxGP() const { return bpCompressionSortSpecs.getFlag(0); }
  void setRelaxGP(bool Value) { bpCompressionSortSpecs.setFlag(0, Value); }
  bool getRelocatable() const { return bpCompressionSortSpecs.getFlag(1); }
  void setRelocatable(bool Value) { bpCompressionSortSpecs.setFlag(1, Value); }
  bool getResolveGroups() const { return bpCompressionSortSpecs.getFlag(2); }
  void setResolveGroups(bool Value) {
    bpCompressionSortSpecs.setFlag(2, Value);
  }
  bool getRelrGlibc() const { return bpCompressionSortSpecs.getFlag(3); }
  void setRelrGlibc(bool Value) { bpCompressionSortSpecs.setFlag(3, Value); }
  bool relrPackDynRelocs = false;
  llvm::DenseSet<llvm::StringRef> saveTempsArgs;
  llvm::SmallVectorWithFlags<std::pair<llvm::GlobPattern, uint32_t>, 0, 4>
      shuffleSections;
  bool getSingleRoRx() const { return mllvmOpts.getFlag(0); }
  void setSingleRoRx(bool Value) { mllvmOpts.setFlag(0, Value); }
  bool getSingleXoRx() const { return mllvmOpts.getFlag(1); }
  void setSingleXoRx(bool Value) { mllvmOpts.setFlag(1, Value); }
  bool getShared() const { return mllvmOpts.getFlag(2); }
  void setShared(bool Value) { mllvmOpts.setFlag(2, Value); }
  bool getSymbolic() const { return mllvmOpts.getFlag(3); }
  void setSymbolic(bool Value) { mllvmOpts.setFlag(3, Value); }
  bool isStatic = false;
  bool getSysvHash() const { return dynamicList.getFlag(0); }
  bool setSysvHash(bool Value) {
    dynamicList.setFlag(0, Value);
    return Value;
  }
  bool getTarget1Rel() const { return dynamicList.getFlag(1); }
  void setTarget1Rel(bool Value) { dynamicList.setFlag(1, Value); }
  bool getTrace() const { return dynamicList.getFlag(2); }
  void setTrace(bool Value) { dynamicList.setFlag(2, Value); }
  bool thinLTOEmitImportsFiles;
  bool thinLTOEmitIndexFiles;
  bool getThinLTOIndexOnly() const { return dynamicList.getFlag(3); }
  void setThinLTOIndexOnly(bool Value) { dynamicList.setFlag(3, Value); }
  bool getTimeTraceEnabled() const { return undefined.getFlag(0); }
  void setTimeTraceEnabled(bool Value) { undefined.setFlag(0, Value); }
  bool getTocOptimize() const { return undefined.getFlag(1); }
  void setTocOptimize(bool Value) { undefined.setFlag(1, Value); }
  bool getPcRelOptimize() const { return undefined.getFlag(2); }
  void setPcRelOptimize(bool Value) { undefined.setFlag(2, Value); }
  bool getUndefinedVersion() const { return undefined.getFlag(3); }
  void setUndefinedVersion(bool Value) { undefined.setFlag(3, Value); }
  bool getUnique() const { return dtltoCompilerArgs.getFlag(0); }
  void setUnique(bool Value) { dtltoCompilerArgs.setFlag(0, Value); }
  bool getUseAndroidRelrTags() const { return dtltoCompilerArgs.getFlag(1); }
  void setUseAndroidRelrTags(bool Value) {
    dtltoCompilerArgs.setFlag(1, Value);
  }
  bool getWarnBackrefs() const { return dtltoCompilerArgs.getFlag(2); }
  void setWarnBackrefs(bool Value) { dtltoCompilerArgs.setFlag(2, Value); }
  llvm::SmallVectorWithFlags<llvm::GlobPattern, 0, 4> warnBackrefsExclude;
  bool warnCommon;
  bool warnMissingEntry;
  bool warnSymbolOrdering;
  bool writeAddends;
  bool zCombreloc;
  bool zCopyreloc;
  bool zDynamicUndefined;
  bool getZForceBti() const { return dtltoCompilerArgs.getFlag(3); }
  void setZForceBti(bool Value) { dtltoCompilerArgs.setFlag(3, Value); }
  bool getZForceIbt() const { return dtltoCompilerPrependArgs.getFlag(0); }
  void setZForceIbt(bool Value) { dtltoCompilerPrependArgs.setFlag(0, Value); }
  bool getZGlobal() const { return dtltoCompilerPrependArgs.getFlag(1); }
  void setZGlobal(bool Value) { dtltoCompilerPrependArgs.setFlag(1, Value); }
  bool getZHazardplt() const { return dtltoCompilerPrependArgs.getFlag(2); }
  void setZHazardplt(bool Value) { dtltoCompilerPrependArgs.setFlag(2, Value); }
  bool getZIfuncNoplt() const { return dtltoCompilerPrependArgs.getFlag(3); }
  void setZIfuncNoplt(bool Value) {
    dtltoCompilerPrependArgs.setFlag(3, Value);
  }
  bool getZInitfirst() const { return dtltoDistributorArgs.getFlag(0); }
  void setZInitfirst(bool Value) { dtltoDistributorArgs.setFlag(0, Value); }
  bool getZInterpose() const { return dtltoDistributorArgs.getFlag(1); }
  void setZInterpose(bool Value) { dtltoDistributorArgs.setFlag(1, Value); }
  bool getZKeepDataSectionPrefix() const {
    return dtltoDistributorArgs.getFlag(2);
  }
  void setZKeepDataSectionPrefix(bool Value) {
    dtltoDistributorArgs.setFlag(2, Value);
  }
  bool getZKeepTextSectionPrefix() const {
    return dtltoDistributorArgs.getFlag(3);
  }
  void setZKeepTextSectionPrefix(bool Value) {
    dtltoDistributorArgs.setFlag(3, Value);
  }
  bool getZLrodataAfterBss() const {
    return thinLTOModulesToCompile.getFlag(0);
  }
  void setZLrodataAfterBss(bool Value) {
    thinLTOModulesToCompile.setFlag(0, Value);
  }
  bool getZMarkPlt() const { return thinLTOModulesToCompile.getFlag(1); }
  void setZMarkPlt(bool Value) { thinLTOModulesToCompile.setFlag(1, Value); }
  bool getZNoBtCfi() const { return thinLTOModulesToCompile.getFlag(2); }
  void setZNoBtCfi(bool Value) { thinLTOModulesToCompile.setFlag(2, Value); }
  bool getZNodefaultlib() const { return thinLTOModulesToCompile.getFlag(3); }
  void setZNodefaultlib(bool Value) {
    thinLTOModulesToCompile.setFlag(3, Value);
  }
  bool getZNodelete() const { return searchPaths.getFlag(0); }
  void setZNodelete(bool Value) { searchPaths.setFlag(0, Value); }
  bool getZNodlopen() const { return searchPaths.getFlag(1); }
  void setZNodlopen(bool Value) { searchPaths.setFlag(1, Value); }
  bool getZNow() const { return searchPaths.getFlag(2); }
  void setZNow(bool Value) { searchPaths.setFlag(2, Value); }
  bool getZOrigin() const { return searchPaths.getFlag(3); }
  void setZOrigin(bool Value) { searchPaths.setFlag(3, Value); }
  bool getZPacPlt() const { return passPlugins.getFlag(0); }
  void setZPacPlt(bool Value) { passPlugins.setFlag(0, Value); }
  bool getZRelro() const { return passPlugins.getFlag(1); }
  bool setZRelro(bool Value) {
    passPlugins.setFlag(1, Value);
    return Value;
  }
  bool getZRodynamic() const { return passPlugins.getFlag(2); }
  void setZRodynamic(bool Value) { passPlugins.setFlag(2, Value); }
  bool getZSectionHeader() const { return passPlugins.getFlag(3); }
  void setZSectionHeader(bool Value) { passPlugins.setFlag(3, Value); }
  bool getZShstk() const { return filterList.getFlag(0); }
  void setZShstk(bool Value) { filterList.setFlag(0, Value); }
  bool getZStartStopGC() const { return filterList.getFlag(1); }
  void setZStartStopGC(bool Value) { filterList.setFlag(1, Value); }
  uint8_t zStartStopVisibility;
  bool getZText() const { return filterList.getFlag(2); }
  void setZText(bool Value) { filterList.setFlag(2, Value); }
  bool getZRetpolineplt() const { return filterList.getFlag(3); }
  void setZRetpolineplt(bool Value) { filterList.setFlag(3, Value); }
  bool getZWxneeded() const { return auxiliaryList.getFlag(0); }
  void setZWxneeded(bool Value) { auxiliaryList.setFlag(0, Value); }
  ZicfilpPolicy zZicfilp;
  ZicfissPolicy zZicfiss;
  DiscardPolicy discard;
  GnuStackKind zGnustack;
  ICFLevel icf;
  OrphanHandlingPolicy orphanHandling;
  SortSectionPolicy sortSection;
  StripPolicy strip;
  UnresolvedPolicy unresolvedSymbols;
  UnresolvedPolicy unresolvedSymbolsInShlib;
  Target2Policy target2;
  GcsPolicy zGcs;
  bool getPower10Stubs() const { return auxiliaryList.getFlag(1); }
  void setPower10Stubs(bool Value) { auxiliaryList.setFlag(1, Value); }
  ARMVFPArgKind armVFPArgs = ARMVFPArgKind::Default;
  BuildIdKind buildId = BuildIdKind::None;
  SeparateSegmentKind zSeparate;
  ELFKind ekind = ELFNoneKind;
  uint16_t emachine = llvm::ELF::EM_NONE;
  std::optional<uint64_t> imageBase;
  uint64_t commonPageSize;
  uint64_t maxPageSize;
  uint64_t mipsGotSize;
  uint64_t zStackSize;
  unsigned ltoPartitions;
  unsigned ltoo;
  llvm::CodeGenOptLevel ltoCgo;
  unsigned optimize;
  StringRef thinLTOJobs;
  unsigned timeTraceGranularity;
  int32_t splitStackAdjustSize;
  llvm::SmallVectorWithFlags<uint8_t, 0, 4> packageMetadata;

  // The following config options do not directly correspond to any
  // particular command line options.

  // True if we need to pass through relocations in input files to the
  // output file. Usually false because we consume relocations.
  bool getCopyRelocs() const { return auxiliaryList.getFlag(2); }
  void setCopyRelocs(bool Value) { auxiliaryList.setFlag(2, Value); }

  // True if the target is ELF64. False if ELF32.
  bool getIs64() const { return auxiliaryList.getFlag(3); }
  void setIs64(bool Value) { auxiliaryList.setFlag(3, Value); }

  // True if the target is little-endian. False if big-endian.
  bool getIsLE() const { return versionDefinitions.getFlag(0); }
  void setIsLE(bool Value) { versionDefinitions.setFlag(0, Value); }

  // endianness::little if isLE is true. endianness::big otherwise.
  llvm::endianness endianness;

  // True if the target is the little-endian MIPS64.
  //
  // The reason why we have this variable only for the MIPS is because
  // we use this often.  Some ELF headers for MIPS64EL are in a
  // mixed-endian (which is horrible and I'd say that's a serious spec
  // bug), and we need to know whether we are reading MIPS ELF files or
  // not in various places.
  //
  // (Note that MIPS64EL is not a typo for MIPS64LE. This is the official
  // name whatever that means. A fun hypothesis is that "EL" is short for
  // little-endian written in the little-endian order, but I don't know
  // if that's true.)
  bool getIsMips64EL() const { return versionDefinitions.getFlag(1); }
  void setIsMips64EL(bool Value) { versionDefinitions.setFlag(1, Value); }

  // Holds set of ELF header flags for the target.
  uint32_t eflags = 0;

  // The ELF spec defines two types of relocation table entries, RELA and
  // REL. RELA is a triplet of (offset, info, addend) while REL is a
  // tuple of (offset, info). Addends for REL are implicit and read from
  // the location where the relocations are applied. So, REL is more
  // compact than RELA but requires a bit of more work to process.
  //
  // (From the linker writer's view, this distinction is not necessary.
  // If the ELF had chosen whichever and sticked with it, it would have
  // been easier to write code to process relocations, but it's too late
  // to change the spec.)
  //
  // Each ABI defines its relocation type. IsRela is true if target
  // uses RELA. As far as we know, all 64-bit ABIs are using RELA. A
  // few 32-bit ABIs are using RELA too.
  bool getIsRela() const { return versionDefinitions.getFlag(2); }
  void setIsRela(bool Value) { versionDefinitions.setFlag(2, Value); }

  // True if we are creating position-independent code.
  bool getIsPic() const { return versionDefinitions.getFlag(3); }
  void setIsPic(bool Value) { versionDefinitions.setFlag(3, Value); }

  // 4 for ELF32, 8 for ELF64.
  int wordsize;

  // Mode of MTE to write to the dynamic array. Should be one of NT_MEMTAG_ASYNC
  // (for async), NT_MEMTAG_SYNC (for sync), or NT_MEMTAG_LEVEL_NONE (for none).
  // If async or sync is enabled, write the tag specifying the default MTE mode.
  int memtagMode;
  // Signal to the dynamic loader to enable heap MTE.
  bool memtagHeap;
  // Signal to the dynamic loader that this binary expects stack MTE. Generally,
  // this means to map the primary and thread stacks as PROT_MTE. Note: This is
  // not supported on Android 11 & 12.
  bool memtagStack;
  // Whether to emit the Android-specific legacy memtag note.
  bool memtagAndroidNote;

  // When using a unified pre-link LTO pipeline, specify the backend LTO mode.
  LtoKind ltoKind = LtoKind::Default;

  unsigned threadCount;

  // If an input file equals a key, remap it to the value.
  llvm::DenseMap<llvm::StringRef, llvm::StringRef> remapInputs;
  // If an input file matches a wildcard pattern, remap it to the value.
  llvm::SmallVectorWithFlags<std::pair<llvm::GlobPattern, llvm::StringRef>, 0,
                             4>
      remapInputsWildcards;
};

// Some index properties of a symbol are stored separately in this auxiliary
// struct to decrease sizeof(SymbolUnion) in the majority of cases.
struct SymbolAux {
  uint32_t gotIdx = -1;
  uint32_t pltIdx = -1;
  uint32_t tlsDescIdx = -1;
  uint32_t tlsGdIdx = -1;
};

struct DuplicateSymbol {
  const Symbol *sym;
  const InputFile *file;
  InputSectionBase *section;
  uint64_t value;
};

struct UndefinedDiag {
  Undefined *sym;
  struct Loc {
    InputSectionBase *sec;
    uint64_t offset;
  };
  llvm::SmallVectorWithFlags<Loc, 0> locs;
  bool getIsWarning() const { return locs.getFlag(0); }
  void setIsWarning(bool Value) { locs.setFlag(0, Value); }

  UndefinedDiag(Undefined *sym, SmallVector<Loc, 0> locs, bool isWarning)
      : sym(sym), locs(unsigned(isWarning), std::move(locs)) {}
};

// Linker generated sections which can be used as inputs.
struct InStruct {
  std::unique_ptr<InputSection> attributes;
  std::unique_ptr<BssSection> bss;
  std::unique_ptr<BssSection> bssRelRo;
  std::unique_ptr<BuildIdSection> buildId;
  std::unique_ptr<EhFrameHeader> ehFrameHdr;
  std::unique_ptr<EhFrameSection> ehFrame;
  std::unique_ptr<GnuHashTableSection> gnuHashTab;
  std::unique_ptr<GotPltSection> gotPlt;
  std::unique_ptr<GotSection> got;
  std::unique_ptr<HashTableSection> hashTab;
  std::unique_ptr<IgotPltSection> igotPlt;
  std::unique_ptr<IpltSection> iplt;
  std::unique_ptr<MemtagAndroidNote> memtagAndroidNote;
  std::unique_ptr<MemtagGlobalDescriptors> memtagGlobalDescriptors;
  std::unique_ptr<PackageMetadataNote> packageMetadataNote;
  std::unique_ptr<PltSection> plt;
  std::unique_ptr<RelocationBaseSection> relaDyn;
  std::unique_ptr<RelocationBaseSection> relaPlt;
  std::unique_ptr<RelrBaseSection> relrAuthDyn;
  std::unique_ptr<RelrBaseSection> relrDyn;
  std::unique_ptr<RelroPaddingSection> relroPadding;
  std::unique_ptr<StringTableSection> dynStrTab;
  std::unique_ptr<SymbolTableBaseSection> dynSymTab;
  std::unique_ptr<SyntheticSection> dynamic;
  std::unique_ptr<SyntheticSection> gnuProperty;
  std::unique_ptr<SyntheticSection> gnuStack;
  std::unique_ptr<SyntheticSection> ibtPlt;
  std::unique_ptr<SyntheticSection> verNeed;
  std::unique_ptr<VersionDefinitionSection> verDef;
  std::unique_ptr<VersionTableSection> verSym;

  std::unique_ptr<SyntheticSection> armCmseSGSection;
  std::unique_ptr<ARMExidxSyntheticSection> armExidx;
  std::unique_ptr<PPC64LongBranchTargetSection> ppc64LongBranchTarget;
  std::unique_ptr<SyntheticSection> mipsAbiFlags;
  std::unique_ptr<MipsGotSection> mipsGot;
  std::unique_ptr<SyntheticSection> mipsOptions;
  std::unique_ptr<SyntheticSection> mipsReginfo;
  std::unique_ptr<SyntheticSection> mipsRldMap;
  std::unique_ptr<SyntheticSection> ppc32Got2;
  // Non-SHF_ALLOC sections
  std::unique_ptr<SyntheticSection> debugNames;
  std::unique_ptr<GdbIndexSection> gdbIndex;
  std::unique_ptr<StringTableSection> shStrTab;
  std::unique_ptr<StringTableSection> strTab;
  std::unique_ptr<SymbolTableBaseSection> symTab;
  std::unique_ptr<SymtabShndxSection> symTabShndx;
  std::unique_ptr<SyntheticSection> hexagonAttributes;
  std::unique_ptr<SyntheticSection> riscvAttributes;
  std::unique_ptr<SyntheticSection> dynDbg;
  std::unique_ptr<SyntheticSection> dynDbgNote;
};

struct Ctx : CommonLinkerContext {
  Config arg;
  LinkerDriver driver;
  LinkerScript *script;
  std::unique_ptr<TargetInfo> target;

  // These variables are initialized by Writer and should not be used before
  // Writer is initialized.
  uint8_t *bufferStart = nullptr;
  PhdrEntry *tlsPhdr = nullptr;
  SmallVector<std::unique_ptr<PhdrEntry>, 0> phdrs;
  struct OutSections {
    std::unique_ptr<OutputSection> elfHeader;
    std::unique_ptr<OutputSection> programHeaders;
    OutputSection *preinitArray = nullptr;
    OutputSection *initArray = nullptr;
    OutputSection *finiArray = nullptr;
  };
  OutSections out;
  SmallVector<OutputSection *, 0> outputSections;

  InStruct in;

  // Some linker-generated symbols need to be created as
  // Defined symbols.
  struct ElfSym {
    // __bss_start
    Defined *bss;

    // etext and _etext
    Defined *etext1;
    Defined *etext2;

    // edata and _edata
    Defined *edata1;
    Defined *edata2;

    // end and _end
    Defined *end1;
    Defined *end2;

    // The _GLOBAL_OFFSET_TABLE_ symbol is defined by target convention to
    // be at some offset from the base of the .got section, usually 0 or
    // the end of the .got.
    Defined *globalOffsetTable;

    // _gp, _gp_disp and __gnu_local_gp symbols. Only for MIPS.
    Defined *mipsGp;
    Defined *mipsGpDisp;
    Defined *mipsLocalGp;

    // __global_pointer$ for RISC-V.
    Defined *riscvGlobalPointer;

    // __rel{,a}_iplt_{start,end} symbols.
    Defined *relaIpltStart;
    Defined *relaIpltEnd;

    // _TLS_MODULE_BASE_ on targets that support TLSDESC.
    Defined *tlsModuleBase;
  };
  ElfSym sym{};
  std::unique_ptr<SymbolTable> symtab;
  SmallVector<Symbol *, 0> synthesizedSymbols;
  // ifunc resolver symbol clones for IRELATIVE. Linker relaxation adjusts
  // these.
  SmallVector<Defined *, 0> irelativeSyms;

  SmallVector<std::unique_ptr<MemoryBuffer>> memoryBuffers;
  SmallVector<ELFFileBase *, 0> objectFiles;
  SmallVector<SharedFile *, 0> sharedFiles;
  SmallVector<BinaryFile *, 0> binaryFiles;
  SmallVector<BitcodeFile *, 0> bitcodeFiles;
  SmallVector<BitcodeFile *, 0> lazyBitcodeFiles;
  SmallVector<InputSectionBase *, 0> inputSections;
  SmallVector<EhInputSection *, 0> ehInputSections;

  SmallVector<SymbolAux, 0> symAux;
  // Duplicate symbol candidates.
  SmallVector<DuplicateSymbol, 0> duplicates;
  // Undefined diagnostics are collected in a vector and emitted once all of
  // them are known, so that some postprocessing on the list of undefined
  // symbols can happen before lld emits diagnostics.
  std::mutex relocMutex;
  SmallVector<UndefinedDiag, 0> undefErrs;
  // Symbols in a non-prevailing COMDAT group which should be changed to an
  // Undefined.
  llvm::SmallVectorWithFlags<std::pair<Symbol *, unsigned>, 0>
      nonPrevailingSyms;
  // A tuple of (reference, extractedFile, sym). Used by --why-extract=.
  llvm::SmallVectorWithFlags<
      std::tuple<std::string, const InputFile *, const Symbol &>, 0, 4>
      whyExtractRecords;
  // A mapping from a symbol to an InputFile referencing it backward. Used by
  // --warn-backrefs.
  llvm::DenseMap<const Symbol *,
                 std::pair<const InputFile *, const InputFile *>>
      backwardReferences;
  llvm::SmallSet<llvm::StringRef, 0> auxiliaryFiles;
  // If --reproduce is specified, all input files are written to this tar
  // archive.
  std::unique_ptr<llvm::TarWriter> tar;
  // InputFile for linker created symbols with no source location.
  InputFile *internalFile = nullptr;
  // Dummy Undefined for relocations without a symbol.
  Undefined *dummySym = nullptr;
  // True if symbols can be exported (isExported) or preemptible.
  bool getHasDynsym() const { return whyExtractRecords.getFlag(0); }
  void setHasDynsym(bool Value) { whyExtractRecords.setFlag(0, Value); }
  // True if there are TLS IE relocations. Set DF_STATIC_TLS if -shared.
  std::atomic<bool> hasTlsIe{false};
  // True if we need to reserve two .got entries for local-dynamic TLS model.
  std::atomic<bool> needsTlsLd{false};
  // True if all native vtable symbols have corresponding type info symbols
  // during LTO.
  bool getLtoAllVtablesHaveTypeInfos() const {
    return whyExtractRecords.getFlag(1);
  }
  void setLtoAllVtablesHaveTypeInfos(bool Value) {
    whyExtractRecords.setFlag(1, Value);
  }
  // Number of Vernaux entries (needed shared object names).
  uint32_t vernauxNum = 0;

  // Each symbol assignment and DEFINED(sym) reference is assigned an increasing
  // order. Each DEFINED(sym) evaluation checks whether the reference happens
  // before a possible `sym = expr;`.
  unsigned scriptSymOrderCounter = 1;
  llvm::DenseMap<const Symbol *, unsigned> scriptSymOrder;

  // Used to assert removeUnusedSyntheticSections-removed sections cannot become
  // needed again.
  SmallVector<SyntheticSection *, 0> removedSyntheticSections;

  // The set of TOC entries (.toc + addend) for which we should not apply
  // toc-indirect to toc-relative relaxation. const Symbol * refers to the
  // STT_SECTION symbol associated to the .toc input section.
  llvm::DenseSet<std::pair<const Symbol *, uint64_t>> ppc64noTocRelax;

  Ctx();

  llvm::raw_fd_ostream openAuxiliaryFile(llvm::StringRef, std::error_code &);

  std::optional<AArch64PauthAbiCoreInfo> aarch64PauthAbiCoreInfo;

  // True if performing the embedded unoptimized dynamic debugging relocatable
  // link.
  bool getInDynDbgLink() const { return whyExtractRecords.getFlag(2); }
  void setInDynDbgLink(bool Value) { whyExtractRecords.setFlag(2, Value); }
  // True if performing dynamic debugging style relocatable link rather than a
  // regular relocatable link.
  bool getDynDbgRelocatable() const { return whyExtractRecords.getFlag(3); }
  void setDynDbgRelocatable(bool Value) { whyExtractRecords.setFlag(3, Value); }
  // True if the link contains dynamic debugging.
  bool getHasDynDbg() const { return nonPrevailingSyms.getFlag(0); }
  void setHasDynDbg(bool Value) { nonPrevailingSyms.setFlag(0, Value); }
  // Pointer to the output of the embedded unoptimized dynamic debugging
  // relocatable link.
  std::unique_ptr<llvm::FileOutputBuffer> dynDbgOutput;
};

// The first two elements of versionDefinitions represent VER_NDX_LOCAL and
// VER_NDX_GLOBAL. This helper returns other elements.
static inline ArrayRef<VersionDefinition> namedVersionDefs(Ctx &ctx) {
  return llvm::ArrayRef(ctx.arg.versionDefinitions).slice(2);
}

struct ELFSyncStream : SyncStream {
  Ctx &ctx;
  ELFSyncStream(Ctx &ctx, DiagLevel level)
      : SyncStream(ctx.e, level), ctx(ctx) {}
};

template <typename T>
std::enable_if_t<!std::is_pointer_v<std::remove_reference_t<T>>,
                 const ELFSyncStream &>
operator<<(const ELFSyncStream &s, T &&v) {
  s.os << std::forward<T>(v);
  return s;
}

inline const ELFSyncStream &operator<<(const ELFSyncStream &s, const char *v) {
  s.os << v;
  return s;
}

inline const ELFSyncStream &operator<<(const ELFSyncStream &s, Error v) {
  s.os << llvm::toString(std::move(v));
  return s;
}

// Report a log if --verbose is specified.
ELFSyncStream Log(Ctx &ctx);

// Print a message to stdout.
ELFSyncStream Msg(Ctx &ctx);

// Report a warning. Upgraded to an error if --fatal-warnings is specified.
ELFSyncStream Warn(Ctx &ctx);

// Report an error that will suppress the output file generation. Downgraded to
// a warning if --noinhibit-exec is specified.
ELFSyncStream Err(Ctx &ctx);

// Report an error regardless of --noinhibit-exec.
ELFSyncStream ErrAlways(Ctx &ctx);

// Report a fatal error that exits immediately. This should generally be avoided
// in favor of Err.
ELFSyncStream Fatal(Ctx &ctx);

uint64_t errCount(Ctx &ctx);

ELFSyncStream InternalErr(Ctx &ctx, const uint8_t *buf);

#define CHECK2(E, S) lld::check2((E), [&] { return toStr(ctx, S); })

inline DiagLevel toDiagLevel(ReportPolicy policy) {
  if (policy == ReportPolicy::Error)
    return DiagLevel::Err;
  else if (policy == ReportPolicy::Warning)
    return DiagLevel::Warn;
  return DiagLevel::None;
}

} // namespace lld::elf

#endif
