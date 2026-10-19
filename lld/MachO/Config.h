//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_CONFIG_H
#define LLD_MACHO_CONFIG_H

#include "lld/Common/BPSectionOrdererBase.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorWithFlags.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/CachePruning.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TextAPI/Architecture.h"
#include "llvm/TextAPI/Platform.h"
#include "llvm/TextAPI/Target.h"

#include <vector>

namespace llvm {
enum class CodeGenOptLevel;
} // namespace llvm

namespace lld {
namespace macho {

class InputSection;
class Symbol;

using NamePair = std::pair<llvm::StringRef, llvm::StringRef>;
using SectionRenameMap = llvm::DenseMap<NamePair, NamePair>;
using SegmentRenameMap = llvm::DenseMap<llvm::StringRef, llvm::StringRef>;

struct PlatformInfo {
  llvm::MachO::Target target;
  llvm::VersionTuple sdk;
};

inline uint32_t encodeVersion(const llvm::VersionTuple &version) {
  return ((version.getMajor() << 020) |
          (version.getMinor().value_or(0) << 010) |
          version.getSubminor().value_or(0));
}

enum class NamespaceKind {
  twolevel,
  flat,
};

enum class UndefinedSymbolTreatment {
  unknown,
  error,
  warning,
  suppress,
  dynamic_lookup,
};

enum class ICFLevel {
  unknown,
  none,
  safe,
  safe_thunks,
  all,
};

enum class ObjCStubsMode {
  fast,
  small,
};

struct SectionAlign {
  llvm::StringRef segName;
  llvm::StringRef sectName;
  uint32_t align;
};

struct SegmentProtection {
  llvm::StringRef name;
  uint32_t maxProt;
  uint32_t initProt;
};

class SymbolPatterns {
public:
  // GlobPattern can also match literals,
  // but we prefer the O(1) lookup of DenseSet.
  llvm::SetVector<llvm::CachedHashStringRef> literals;
  std::vector<llvm::GlobPattern> globs;

  bool empty() const { return literals.empty() && globs.empty(); }
  void clear();
  void insert(llvm::StringRef symbolName);
  bool matchLiteral(llvm::StringRef symbolName) const;
  bool matchGlob(llvm::StringRef symbolName) const;
  bool match(llvm::StringRef symbolName) const;
};

enum class SymtabPresence {
  All,
  None,
  SelectivelyIncluded,
  SelectivelyExcluded,
};

struct Configuration {
  Symbol *entry = nullptr;
  bool hasReexports = false;
  bool allLoad = false;
  bool applicationExtension = false;
  bool archMultiple = false;
  bool exportDynamic = false;
  bool forceLoadObjC = false;
  bool forceLoadSwift = false; // Only applies to LC_LINKER_OPTIONs.
  bool staticLink = false;
  bool implicitDylibs = false;
  bool isPic = false;
  bool headerPadMaxInstallNames = false;
  bool markDeadStrippableDylib = false;
  bool printDylibSearch = false;
  bool printEachFile = false;
  bool printWhyLoad = false;
  bool searchDylibsFirst = false;
  bool saveTemps = false;
  bool adhocCodesign = false;
  bool emitFunctionStarts = false;
  bool emitDataInCodeInfo = false;
  bool emitEncryptionInfo = false;
  bool emitInitOffsets = false;
  bool emitChainedFixups = false;
  bool emitRelativeMethodLists = false;
  bool thinLTOEmitImportsFiles;
  bool thinLTOEmitIndexFiles;
  bool thinLTOIndexOnly;
  bool timeTraceEnabled = false;
  bool dataConst = false;
  bool dedupStrings = true;
  bool dedupSymbolStrings = true;
  bool deadStripDuplicates = false;
  bool omitDebugInfo = false;
  bool stripSwiftForceLoad = false;
  bool warnDylibInstallName = false;
  bool ignoreOptimizationHints = false;
  bool getForceExactCpuSubtypeMatch() const { return passPlugins.getFlag(0); }
  void setForceExactCpuSubtypeMatch(bool Value) {
    passPlugins.setFlag(0, Value);
  }
  uint32_t headerPad;
  uint32_t dylibCompatibilityVersion = 0;
  uint32_t dylibCurrentVersion = 0;
  uint32_t timeTraceGranularity = 500;
  unsigned optimize;
  std::string progName;

  // For `clang -arch arm64 -arch x86_64`, clang will:
  // 1. invoke the linker twice, to write one temporary output per arch
  // 2. invoke `lipo` to merge the two outputs into a single file
  // `outputFile` is the name of the temporary file the linker writes to.
  // `finalOutput `is the name of the file lipo writes to after the link.
  llvm::StringRef outputFile;
  llvm::StringRef finalOutput;

  llvm::StringRef installName;
  llvm::StringRef clientName;
  llvm::StringRef mapFile;
  llvm::StringRef ltoNewPmPasses;
  llvm::StringRef ltoObjPath;
  llvm::StringRef thinLTOJobs;
  llvm::StringRef umbrella;
  uint32_t ltoo = 2;
  llvm::CodeGenOptLevel ltoCgo;
  llvm::CachePruningPolicy thinLTOCachePolicy;
  llvm::StringRef thinLTOCacheDir;
  llvm::StringRef thinLTOIndexOnlyArg;
  std::pair<llvm::StringRef, llvm::StringRef> thinLTOObjectSuffixReplace;
  llvm::StringRef thinLTOPrefixReplaceOld;
  llvm::StringRef thinLTOPrefixReplaceNew;
  llvm::StringRef thinLTOPrefixReplaceNativeObject;
  bool deadStripDylibs = false;
  bool demangle = false;
  bool deadStrip = false;
  bool interposable = false;
  bool getErrorForArchMismatch() const { return passPlugins.getFlag(1); }
  void setErrorForArchMismatch(bool Value) { passPlugins.setFlag(1, Value); }
  bool getIgnoreAutoLink() const { return passPlugins.getFlag(2); }
  void setIgnoreAutoLink(bool Value) { passPlugins.setFlag(2, Value); }
  int readWorkers = 0;
  // ld64 allows invalid auto link options as long as the link succeeds. LLD
  // does not, but there are cases in the wild where the invalid linker options
  // exist. This allows users to ignore the specific invalid options in the case
  // they can't easily fix them.
  llvm::StringSet<> ignoreAutoLinkOptions;
  bool getStrictAutoLink() const { return passPlugins.getFlag(3); }
  void setStrictAutoLink(bool Value) { passPlugins.setFlag(3, Value); }
  PlatformInfo platformInfo;
  std::optional<PlatformInfo> secondaryPlatformInfo;
  NamespaceKind namespaceKind = NamespaceKind::twolevel;
  UndefinedSymbolTreatment undefinedSymbolTreatment =
      UndefinedSymbolTreatment::error;
  ICFLevel icfLevel = ICFLevel::none;
  bool getKeepICFStabs() const { return mllvmOpts.getFlag(0); }
  void setKeepICFStabs(bool Value) { mllvmOpts.setFlag(0, Value); }
  ObjCStubsMode objcStubsMode = ObjCStubsMode::fast;
  llvm::MachO::HeaderFileType outputType;
  std::vector<llvm::StringRef> systemLibraryRoots;
  std::vector<llvm::StringRef> librarySearchPaths;
  std::vector<llvm::StringRef> frameworkSearchPaths;
  bool getWarnDuplicateRpath() const { return !mllvmOpts.getFlag(1); }
  void setWarnDuplicateRpath(bool Value) { mllvmOpts.setFlag(1, !Value); }
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> runtimePaths;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> allowableClients;
  std::vector<std::string> astPaths;
  std::vector<Symbol *> explicitUndefineds;
  llvm::StringSet<> explicitDynamicLookups;
  // There are typically few custom sectionAlignments or segmentProtections,
  // so use a vector instead of a map.
  std::vector<SectionAlign> sectionAlignments;
  std::vector<SegmentProtection> segmentProtections;
  bool getLtoDebugPassManager() const { return mllvmOpts.getFlag(2); }
  void setLtoDebugPassManager(bool Value) { mllvmOpts.setFlag(2, Value); }
  bool getEmitLLVM() const { return mllvmOpts.getFlag(3); }
  void setEmitLLVM(bool Value) { mllvmOpts.setFlag(3, Value); }
  llvm::StringRef codegenDataGeneratePath;
  bool getCsProfileGenerate() const {
    return bpCompressionSortSpecs.getFlag(0);
  }
  void setCsProfileGenerate(bool Value) {
    bpCompressionSortSpecs.setFlag(0, Value);
  }
  llvm::StringRef csProfilePath;
  bool pgoWarnMismatch;
  bool warnThinArchiveMissingMembers;
  bool warnMissingSubsectionsViaSymbols = false;
  bool disableVerify;
  bool getSeparateCstringLiteralSections() const {
    return bpCompressionSortSpecs.getFlag(1);
  }
  void setSeparateCstringLiteralSections(bool Value) {
    bpCompressionSortSpecs.setFlag(1, Value);
  }
  bool getTailMergeStrings() const { return bpCompressionSortSpecs.getFlag(2); }
  void setTailMergeStrings(bool Value) {
    bpCompressionSortSpecs.setFlag(2, Value);
  }
  unsigned slopScale = 256;

  bool getCallGraphProfileSort() const {
    return bpCompressionSortSpecs.getFlag(3);
  }
  void setCallGraphProfileSort(bool Value) {
    bpCompressionSortSpecs.setFlag(3, Value);
  }
  llvm::StringRef printSymbolOrder;

  llvm::StringRef irpgoProfilePath;
  bool getBpStartupFunctionSort() const { return allowableClients.getFlag(0); }
  void setBpStartupFunctionSort(bool Value) {
    allowableClients.setFlag(0, Value);
  }
  bool getBpCompressionSortStartupFunctions() const {
    return allowableClients.getFlag(1);
  }
  void setBpCompressionSortStartupFunctions(bool Value) {
    allowableClients.setFlag(1, Value);
  }
  bool getBpFunctionOrderForCompression() const {
    return allowableClients.getFlag(2);
  }
  void setBpFunctionOrderForCompression(bool Value) {
    allowableClients.setFlag(2, Value);
  }
  bool getBpDataOrderForCompression() const {
    return allowableClients.getFlag(3);
  }
  void setBpDataOrderForCompression(bool Value) {
    allowableClients.setFlag(3, Value);
  }
  llvm::SmallVectorWithFlags<BPCompressionSortSpec,
                             llvm::CalculateSmallVectorDefaultInlinedElements<
                                 BPCompressionSortSpec>::value,
                             4>
      bpCompressionSortSpecs;
  bool getBpVerboseSectionOrderer() const { return runtimePaths.getFlag(0); }
  void setBpVerboseSectionOrderer(bool Value) {
    runtimePaths.setFlag(0, Value);
  }

  SectionRenameMap sectionRenameMap;
  SegmentRenameMap segmentRenameMap;

  bool getHasExplicitExports() const { return runtimePaths.getFlag(1); }
  void setHasExplicitExports(bool Value) { runtimePaths.setFlag(1, Value); }
  SymbolPatterns exportedSymbols;
  SymbolPatterns unexportedSymbols;
  SymbolPatterns whyLive;

  std::vector<std::pair<llvm::StringRef, llvm::StringRef>> aliasedSymbols;

  SymtabPresence localSymbolsPresence = SymtabPresence::All;
  SymbolPatterns localSymbolPatterns;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> mllvmOpts;
  llvm::SmallVectorWithFlags<llvm::StringRef, 0, 4> passPlugins;

  bool getZeroModTime() const { return !runtimePaths.getFlag(2); }
  bool setZeroModTime(bool Value) {
    runtimePaths.setFlag(2, !Value);
    return Value;
  }
  bool getGenerateUuid() const { return !runtimePaths.getFlag(3); }
  void setGenerateUuid(bool Value) { runtimePaths.setFlag(3, !Value); }

  llvm::StringRef osoPrefix;

  std::vector<llvm::StringRef> dyldEnvs;

  llvm::MachO::Architecture arch() const { return platformInfo.target.Arch; }

  llvm::MachO::PlatformType platform() const {
    return platformInfo.target.Platform;
  }
};

extern std::unique_ptr<Configuration> config;

} // namespace macho
} // namespace lld

#endif
