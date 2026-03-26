//===- CDependencies.cpp - Dependency Discovery C Interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the dependency discovery interface. It provides a C library for
// the functionality that clang-scan-deps provides.
//
//===----------------------------------------------------------------------===//

#include "CASUtils.h"
#include "CXDiagnosticSetDiagnosticConsumer.h"
#include "CXLoadedDiagnostic.h"
#include "CXString.h"

#include "clang-c/Dependencies.h"

#include "clang/DependencyScanning/DependencyScanningService.h"
#include "clang/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Options/Options.h"
#include "clang/Tooling/DependencyScanningTool.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Process.h"

using namespace clang;
using namespace clang::dependencies;
using namespace clang::tooling;

namespace {
struct DependencyScannerServiceOptions {
  ScanningOutputFormat ConfiguredFormat = ScanningOutputFormat::Full;
  CASOptions CASOpts;
  ScanningOptimizations OptimizeArgs = ScanningOptimizations::Default;
  std::shared_ptr<cas::ObjectStore> CAS;
  std::shared_ptr<cas::ActionCache> Cache;
  std::optional<bool> CacheNegativeStats;
  std::optional<bool> AsyncScanModules;

  ScanningOutputFormat getFormat() const;
};

struct CStringsManager {
  SmallVector<std::unique_ptr<std::vector<const char *>>> OwnedCStr;
  SmallVector<std::unique_ptr<std::vector<std::string>>> OwnedStdStr;

  /// Doesn't own the string contents.
  CXCStringArray createCStringsRef(ArrayRef<std::string> Strings) {
    OwnedCStr.push_back(std::make_unique<std::vector<const char *>>());
    std::vector<const char *> &CStrings = *OwnedCStr.back();
    CStrings.reserve(Strings.size());
    for (const auto &String : Strings)
      CStrings.push_back(String.c_str());
    return {CStrings.data(), CStrings.size()};
  }

  /// Doesn't own the string contents.
  CXCStringArray createCStringsRef(const llvm::StringSet<> &StringsUnordered) {
    std::vector<StringRef> Strings;

    for (auto SI = StringsUnordered.begin(), SE = StringsUnordered.end();
         SI != SE; ++SI)
      Strings.push_back(SI->getKey());

    llvm::sort(Strings);

    OwnedCStr.push_back(std::make_unique<std::vector<const char *>>());
    std::vector<const char *> &CStrings = *OwnedCStr.back();
    CStrings.reserve(Strings.size());
    for (const auto &String : Strings)
      CStrings.push_back(String.data());
    return {CStrings.data(), CStrings.size()};
  }

  /// Gets ownership of string contents.
  CXCStringArray createCStringsOwned(std::vector<std::string> &&Strings) {
    OwnedStdStr.push_back(
        std::make_unique<std::vector<std::string>>(std::move(Strings)));
    return createCStringsRef(*OwnedStdStr.back());
  }
};
} // end anonymous namespace

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyScannerServiceOptions,
                                   CXDependencyScannerServiceOptions)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyScanningService,
                                   CXDependencyScannerService)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyScanningWorker,
                                   CXDependencyScannerWorker)

inline ScanningOutputFormat unwrap(CXDependencyMode Format) {
  switch (Format) {
  case CXDependencyMode_Flat:
    return ScanningOutputFormat::Make;
  case CXDependencyMode_Full:
    return ScanningOutputFormat::Full;
  }
}

static CXOutputKind wrap(ModuleOutputKind MOK) {
  switch (MOK) {
  case ModuleOutputKind::ModuleFile:
    return CXOutputKind_ModuleFile;
  case ModuleOutputKind::DependencyFile:
    return CXOutputKind_Dependencies;
  case ModuleOutputKind::DependencyTargets:
    return CXOutputKind_DependenciesTarget;
  case ModuleOutputKind::DiagnosticSerializationFile:
    return CXOutputKind_SerializedDiagnostics;
  }
  llvm::report_fatal_error("unhandled ModuleOutputKind");
}

CXDependencyScannerServiceOptions
clang_experimental_DependencyScannerServiceOptions_create() {
  return wrap(new DependencyScannerServiceOptions);
}

void clang_experimental_DependencyScannerServiceOptions_dispose(
    CXDependencyScannerServiceOptions Opts) {
  delete unwrap(Opts);
}

void clang_experimental_DependencyScannerServiceOptions_setDependencyMode(
    CXDependencyScannerServiceOptions Opts, CXDependencyMode Mode) {
  unwrap(Opts)->ConfiguredFormat = unwrap(Mode);
}

void clang_experimental_DependencyScannerServiceOptions_setCASDatabases(
    CXDependencyScannerServiceOptions Opts, CXCASDatabases CDBs) {
  cas::WrappedCASDatabases &DBs = *cas::unwrap(CDBs);
  unwrap(Opts)->CASOpts = DBs.CASOpts;
  unwrap(Opts)->CAS = DBs.CAS;
  unwrap(Opts)->Cache = DBs.Cache;
}

void clang_experimental_DependencyScannerServiceOptions_setCASOptions(
    CXDependencyScannerServiceOptions Opts, CXCASOptions CASOpts) {
  unwrap(Opts)->CASOpts = *cas::unwrap(CASOpts);
}

void clang_experimental_DependencyScannerServiceOptions_setCWDOptimization(
    CXDependencyScannerServiceOptions Opts, int Value) {
  auto Mask =
      Value != 0 ? ScanningOptimizations::All : ScanningOptimizations::None;
  auto OptArgs = unwrap(Opts)->OptimizeArgs;
  unwrap(Opts)->OptimizeArgs = (OptArgs & ~ScanningOptimizations::IgnoreCWD) |
                               (ScanningOptimizations::IgnoreCWD & Mask);
}

void clang_experimental_DependencyScannerServiceOptions_setObjectStore(
    CXDependencyScannerServiceOptions Opts, CXCASObjectStore CAS) {
  unwrap(Opts)->CAS = cas::unwrap(CAS)->CAS;
  unwrap(Opts)->CASOpts.CASPath = cas::unwrap(CAS)->CASPath;
}
void clang_experimental_DependencyScannerServiceOptions_setActionCache(
    CXDependencyScannerServiceOptions Opts, CXCASActionCache Cache) {
  unwrap(Opts)->Cache = cas::unwrap(Cache)->Cache;
  unwrap(Opts)->CASOpts.CASPath = cas::unwrap(Cache)->CachePath;
}

void clang_experimental_DependencyScannerServiceOptions_setCacheNegativeStats(
    CXDependencyScannerServiceOptions Opts, bool CacheNegativeStats) {
  unwrap(Opts)->CacheNegativeStats = CacheNegativeStats;
}

ScanningOutputFormat DependencyScannerServiceOptions::getFormat() const {
  if (ConfiguredFormat != ScanningOutputFormat::Full)
    return ConfiguredFormat;

  if (!CAS || !Cache)
    return ConfiguredFormat;

  return ScanningOutputFormat::FullIncludeTree;
}

void clang_experimental_DependencyScannerServiceOptions_setAsyncScanModules(
    CXDependencyScannerServiceOptions Opts, bool AsyncScanModules) {
  unwrap(Opts)->CacheNegativeStats = AsyncScanModules;
}

CXDependencyScannerService
clang_experimental_DependencyScannerService_create_v1(
    CXDependencyScannerServiceOptions WrappedOpts) {
  std::shared_ptr<cas::ObjectStore> CAS = unwrap(WrappedOpts)->CAS;
  std::shared_ptr<cas::ActionCache> Cache = unwrap(WrappedOpts)->Cache;
  // FIXME: Pass default CASOpts now.
  DependencyScanningServiceOptions Opts;
  Opts.Format = unwrap(WrappedOpts)->getFormat();
  if (CAS && Cache)
    Opts.Compilation =
        IncludeTreeCompilation{unwrap(WrappedOpts)->CASOpts, CAS, Cache};
  Opts.MakeVFS = [CAS = CAS] {
    auto FS = llvm::vfs::createPhysicalFileSystem();
    if (CAS)
      FS = llvm::cas::createCASProvidingFileSystem(CAS, std::move(FS));
    return FS;
  };
  Opts.OptimizeArgs = unwrap(WrappedOpts)->OptimizeArgs;
  if (unwrap(WrappedOpts)->CacheNegativeStats)
    Opts.CacheNegativeStats = *unwrap(WrappedOpts)->CacheNegativeStats;
  if (unwrap(WrappedOpts)->AsyncScanModules)
    Opts.AsyncScanModules = *unwrap(WrappedOpts)->AsyncScanModules;
  return wrap(new DependencyScanningService(std::move(Opts)));
}

void clang_experimental_DependencyScannerService_dispose_v0(
    CXDependencyScannerService Service) {
  delete unwrap(Service);
}

CXDependencyScannerWorker clang_experimental_DependencyScannerWorker_create_v0(
    CXDependencyScannerService S) {
  return wrap(new DependencyScanningWorker(*unwrap(S)));
}

void clang_experimental_DependencyScannerWorker_dispose_v0(
    CXDependencyScannerWorker Worker) {
  delete unwrap(Worker);
}

namespace {
class OutputLookup {
public:
  OutputLookup(void *MLOContext, std::variant<CXModuleLookupOutputCallback *,
                                              CXModuleLookupOutputCallback_v2 *>
                                     MLO)
      : MLOContext(MLOContext), MLO(MLO) {}
  std::string lookupModuleOutput(const ModuleDeps &MD, ModuleOutputKind MOK);

private:
  llvm::DenseMap<ModuleID, std::string> PCMPaths;
  void *MLOContext;
  std::variant<CXModuleLookupOutputCallback *,
               CXModuleLookupOutputCallback_v2 *>
      MLO;
};

struct DependencyScannerWorkerScanSettings {
  int argc;
  const char *const *argv;
  const char *ModuleName;
  const char *WorkingDirectory;
  void *MLOContext;
  std::variant<CXModuleLookupOutputCallback *,
               CXModuleLookupOutputCallback_v2 *>
      MLO;
};

struct DependencyGraph {
  TranslationUnitDeps TUDeps;
  SmallString<256> SerialDiagBuf;
  CStringsManager StrMgr{};

  CXDiagnosticSet getDiagnosticSet() const {
    CXLoadDiag_Error Error;
    CXString ErrorString;
    CXDiagnosticSet DiagSet = loadCXDiagnosticsFromBuffer(
        llvm::MemoryBufferRef(SerialDiagBuf, "<diags>"), &Error, &ErrorString);
    assert(Error == CXLoadDiag_None);
    clang_disposeString(ErrorString);
    return DiagSet;
  }
};

struct DependencyGraphModule {
  const ModuleDeps *ModDeps;
  CStringsManager StrMgr{};
};

struct DependencyGraphTUCommand {
  Command *TUCmd;
  CStringsManager StrMgr{};
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyScannerWorkerScanSettings,
                                   CXDependencyScannerWorkerScanSettings)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyGraph, CXDepGraph)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyGraphModule, CXDepGraphModule)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyGraphTUCommand,
                                   CXDepGraphTUCommand)

} // end anonymous namespace

CXDependencyScannerWorkerScanSettings
clang_experimental_DependencyScannerWorkerScanSettings_create(
    int argc, const char *const *argv, const char *ModuleName,
    const char *WorkingDirectory, void *MLOContext,
    CXModuleLookupOutputCallback *MLO) {
  return wrap(new DependencyScannerWorkerScanSettings{
      argc, argv, ModuleName, WorkingDirectory, MLOContext, MLO});
}

void clang_experimental_DependencyScannerWorkerScanSettings_dispose(
    CXDependencyScannerWorkerScanSettings Settings) {
  delete unwrap(Settings);
}

enum CXErrorCode clang_experimental_DependencyScannerWorker_getDepGraph(
    CXDependencyScannerWorker W,
    CXDependencyScannerWorkerScanSettings CXSettings, CXDepGraph *Out) {
  DependencyScannerWorkerScanSettings &Settings = *unwrap(CXSettings);
  int argc = Settings.argc;
  const char *const *argv = Settings.argv;
  const char *ModuleName = Settings.ModuleName;
  const char *WorkingDirectory = Settings.WorkingDirectory;
  void *MLOContext = Settings.MLOContext;
  std::variant<CXModuleLookupOutputCallback *,
               CXModuleLookupOutputCallback_v2 *>
      MLO = Settings.MLO;

  OutputLookup OL(MLOContext, MLO);
  auto LookupOutputs = [&](const ModuleDeps &MD, ModuleOutputKind MOK) {
    return OL.lookupModuleOutput(MD, MOK);
  };

  if (!Out)
    return CXError_InvalidArguments;
  *Out = nullptr;

  DependencyGraph *DepGraph = new DependencyGraph();
  *Out = wrap(DepGraph);

  // We capture diagnostics as a serialized diagnostics buffer, so that we don't
  // need to keep a valid SourceManager in order to access diagnostic locations.
  DiagnosticOptions DiagOpts;
  auto DiagOS =
      std::make_unique<llvm::raw_svector_ostream>(DepGraph->SerialDiagBuf);
  std::unique_ptr<DiagnosticConsumer> SerialDiagConsumer =
      serialized_diags::create("<diagnostics>", DiagOpts,
                               /*MergeChildRecords=*/false, std::move(DiagOS));

  if (!W || argc < 2 || !argv)
    return CXError_InvalidArguments;

  DependencyScanningWorker *Worker = unwrap(W);

  if (Worker->getScanningFormat() != ScanningOutputFormat::Full &&
      Worker->getScanningFormat() != ScanningOutputFormat::FullIncludeTree)
    return CXError_InvalidArguments;

  std::vector<std::string> Compilation{argv, argv + argc};

  llvm::DenseSet<ModuleID> AlreadySeen;
  FullDependencyConsumer DepConsumer(AlreadySeen);
  auto Controller = DependencyScanningTool::createActionController(
      *Worker, std::move(LookupOutputs));

  bool Result = false;
  if (ModuleName) {
    // FIXME: Tool creates its own worker. Avoid that.
    DependencyScanningTool Tool(Worker->getService());
    auto MaybeCIWithCtx =
        CompilerInstanceWithContext::initializeFromCommandline(
            Tool, WorkingDirectory, Compilation, *Controller,
            *SerialDiagConsumer);

    if (!MaybeCIWithCtx)
      return CXError_Failure;
    Result = MaybeCIWithCtx->computeDependencies(StringRef(ModuleName),
                                                 DepConsumer, *Controller);
    if (!Result)
      return CXError_Failure;
  } else {
    Result = clang::tooling::computeDependencies(
        *Worker, WorkingDirectory, Compilation, DepConsumer, *Controller,
        *SerialDiagConsumer);
  }

  if (!Result)
    return CXError_Failure;

  DepGraph->TUDeps = DepConsumer.takeTranslationUnitDeps();
  return CXError_Success;
}

void clang_experimental_DepGraph_dispose(CXDepGraph Graph) {
  delete unwrap(Graph);
}

size_t clang_experimental_DepGraph_getNumModules(CXDepGraph Graph) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  return TUDeps.ModuleGraph.size();
}

CXDepGraphModule clang_experimental_DepGraph_getModule(CXDepGraph Graph,
                                                       size_t Index) {
  return clang_experimental_DepGraph_getModuleTopological(Graph, Index);
}

CXDepGraphModule
clang_experimental_DepGraph_getModuleTopological(CXDepGraph Graph,
                                                 size_t Index) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  return wrap(new DependencyGraphModule{&TUDeps.ModuleGraph[Index]});
}

void clang_experimental_DepGraphModule_dispose(CXDepGraphModule CXDepMod) {
  delete unwrap(CXDepMod);
}

const char *
clang_experimental_DepGraphModule_getName(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  return ModDeps.ID.ModuleName.c_str();
}

const char *
clang_experimental_DepGraphModule_getContextHash(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  return ModDeps.ID.ContextHash.c_str();
}

const char *
clang_experimental_DepGraphModule_getModuleMapPath(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  if (ModDeps.ClangModuleMapFile.empty())
    return nullptr;
  return ModDeps.ClangModuleMapFile.c_str();
}

CXCStringArray
clang_experimental_DepGraphModule_getFileDeps(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  std::vector<std::string> FileDeps;
  ModDeps.forEachFileDep([&](StringRef File) { FileDeps.emplace_back(File); });
  return unwrap(CXDepMod)->StrMgr.createCStringsOwned(std::move(FileDeps));
}

CXCStringArray
clang_experimental_DepGraphModule_getModuleDeps(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  std::vector<std::string> Modules;
  Modules.reserve(ModDeps.ClangModuleDeps.size());
  for (const auto &MID : ModDeps.ClangModuleDeps)
    Modules.push_back(MID.ModuleName + ":" + MID.ContextHash);
  return unwrap(CXDepMod)->StrMgr.createCStringsOwned(std::move(Modules));
}

bool clang_experimental_DepGraphModule_isInStableDirs(
    CXDepGraphModule CXDepMod) {
  return unwrap(CXDepMod)->ModDeps->IsInStableDirectories;
}

CXCStringArray
clang_experimental_DepGraphModule_getBuildArguments(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  return unwrap(CXDepMod)->StrMgr.createCStringsRef(
      ModDeps.getBuildArguments());
}

const char *clang_experimental_DepGraphModule_getFileSystemRootID(
    CXDepGraphModule CXDepMod) {
  return nullptr;
}

const char *
clang_experimental_DepGraphModule_getIncludeTreeID(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  if (ModDeps.IncludeTreeID)
    return ModDeps.IncludeTreeID->c_str();
  return nullptr;
}

const char *
clang_experimental_DepGraphModule_getCacheKey(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  if (ModDeps.ModuleCacheKey)
    return ModDeps.ModuleCacheKey->c_str();
  return nullptr;
}

int clang_experimental_DepGraphModule_isCWDIgnored(CXDepGraphModule CXDepMod) {
  return unwrap(CXDepMod)->ModDeps->IgnoreCWD;
}

size_t clang_experimental_DepGraph_getNumTUCommands(CXDepGraph Graph) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  return TUDeps.Commands.size();
}

CXDepGraphTUCommand clang_experimental_DepGraph_getTUCommand(CXDepGraph Graph,
                                                             size_t Index) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  return wrap(new DependencyGraphTUCommand{&TUDeps.Commands[Index]});
}

void clang_experimental_DepGraphTUCommand_dispose(CXDepGraphTUCommand CXCmd) {
  delete unwrap(CXCmd);
}

const char *
clang_experimental_DepGraphTUCommand_getExecutable(CXDepGraphTUCommand CXCmd) {
  Command &TUCmd = *unwrap(CXCmd)->TUCmd;
  return TUCmd.Executable.c_str();
}

CXCStringArray clang_experimental_DepGraphTUCommand_getBuildArguments(
    CXDepGraphTUCommand CXCmd) {
  Command &TUCmd = *unwrap(CXCmd)->TUCmd;
  return unwrap(CXCmd)->StrMgr.createCStringsRef(TUCmd.Arguments);
}

const char *
clang_experimental_DepGraphTUCommand_getCacheKey(CXDepGraphTUCommand CXCmd) {
  Command &TUCmd = *unwrap(CXCmd)->TUCmd;
  if (TUCmd.TUCacheKey)
    return TUCmd.TUCacheKey->c_str();
  return nullptr;
}

CXCStringArray clang_experimental_DepGraph_getTUFileDeps(CXDepGraph Graph) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  return unwrap(Graph)->StrMgr.createCStringsRef(TUDeps.FileDeps);
}

CXCStringArray clang_experimental_DepGraph_getTUModuleDeps(CXDepGraph Graph) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  std::vector<std::string> Modules;
  Modules.reserve(TUDeps.ClangModuleDeps.size());
  for (const ModuleID &MID : TUDeps.ClangModuleDeps)
    Modules.push_back(MID.ModuleName + ":" + MID.ContextHash);
  return unwrap(Graph)->StrMgr.createCStringsOwned(std::move(Modules));
}

const char *
clang_experimental_DepGraph_getTUFileSystemRootID(CXDepGraph Graph) {
  return nullptr;
}

const char *clang_experimental_DepGraph_getTUIncludeTreeID(CXDepGraph Graph) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  if (TUDeps.IncludeTreeID)
    return TUDeps.IncludeTreeID->c_str();
  return nullptr;
}

const char *clang_experimental_DepGraph_getTUContextHash(CXDepGraph Graph) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  return TUDeps.ID.ContextHash.c_str();
}

void clang_experimental_DependencyScannerWorkerScanSettings_setModuleLookupCallback(
    CXDependencyScannerWorkerScanSettings CXSettings,
    CXModuleLookupOutputCallback_v2 *MLO) {
  DependencyScannerWorkerScanSettings &Settings = *unwrap(CXSettings);
  Settings.MLO = MLO;
}

CXDiagnosticSet clang_experimental_DepGraph_getDiagnostics(CXDepGraph Graph) {
  return unwrap(Graph)->getDiagnosticSet();
}

static std::string
lookupModuleOutput(const ModuleDeps &MD, ModuleOutputKind MOK, void *MLOContext,
                   std::variant<CXModuleLookupOutputCallback *,
                                CXModuleLookupOutputCallback_v2 *>
                       MLO) {
  SmallVector<char, 256> Buffer(256);
  auto GetLengthFromOutputCallback = [&]() {
    return std::visit(llvm::makeVisitor(
                          [&](CXModuleLookupOutputCallback *) -> size_t {
                            return std::get<CXModuleLookupOutputCallback *>(
                                MLO)(MLOContext, MD.ID.ModuleName.c_str(),
                                     MD.ID.ContextHash.c_str(), wrap(MOK),
                                     Buffer.data(), Buffer.size());
                          },
                          [&](CXModuleLookupOutputCallback_v2 *) -> size_t {
                            return std::get<CXModuleLookupOutputCallback_v2 *>(
                                MLO)(MLOContext,
                                     wrap(new DependencyGraphModule{&MD}),
                                     wrap(MOK), Buffer.data(), Buffer.size());
                          }),
                      MLO);
  };

  size_t Len = GetLengthFromOutputCallback();
  if (Len > Buffer.size()) {
    Buffer.resize(Len);
    Len = GetLengthFromOutputCallback();
  }

  return std::string(Buffer.begin(), Len);
}

std::string OutputLookup::lookupModuleOutput(const ModuleDeps &MD,
                                             ModuleOutputKind MOK) {
  if (MOK != ModuleOutputKind::ModuleFile)
    return ::lookupModuleOutput(MD, MOK, MLOContext, MLO);
  // PCM paths are looked up repeatedly, so cache them.
  auto PCMPath = PCMPaths.insert({MD.ID, ""});
  if (PCMPath.second)
    PCMPath.first->second = ::lookupModuleOutput(MD, MOK, MLOContext, MLO);
  return PCMPath.first->second;
}

namespace {
typedef std::vector<DependencyScanningFilesystemSharedCache::OutOfDateEntry>
    DependencyScannerFSOutOfDateEntrySet;

typedef DependencyScanningFilesystemSharedCache::OutOfDateEntry
    DependencyScannerFSOutOfDateEntry;
} // namespace

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyScannerFSOutOfDateEntrySet,
                                   CXDepScanFSOutOfDateEntrySet)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyScannerFSOutOfDateEntry,
                                   CXDepScanFSOutOfDateEntry)

CXDepScanFSOutOfDateEntrySet
clang_experimental_DependencyScannerService_getFSCacheOutOfDateEntrySet(
    CXDependencyScannerService S) {
  DependencyScanningService &Service = *unwrap(S);

  // Note that it is critical that this FS is the same as the default virtual
  // file system we pass to the DependencyScanningWorkers.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPhysicalFileSystem();

  DependencyScannerFSOutOfDateEntrySet *OODEntrySet =
      new DependencyScannerFSOutOfDateEntrySet();
  *OODEntrySet = Service.getSharedCache().getOutOfDateEntries(*FS);

  return wrap(OODEntrySet);
}

size_t clang_experimental_DepScanFSCacheOutOfDateEntrySet_getNumOfEntries(
    CXDepScanFSOutOfDateEntrySet Entries) {
  return unwrap(Entries)->size();
}

CXDepScanFSOutOfDateEntry
clang_experimental_DepScanFSCacheOutOfDateEntrySet_getEntry(
    CXDepScanFSOutOfDateEntrySet Entries, size_t Idx) {
  DependencyScannerFSOutOfDateEntrySet *EntSet = unwrap(Entries);
  return wrap(&(*EntSet)[Idx]);
}

CXDepScanFSCacheOutOfDateKind
clang_experimental_DepScanFSCacheOutOfDateEntry_getKind(
    CXDepScanFSOutOfDateEntry Entry) {
  DependencyScannerFSOutOfDateEntry *E = unwrap(Entry);
  auto &Info = E->Info;
  return std::visit(
      llvm::makeVisitor(
          [](const DependencyScannerFSOutOfDateEntry::NegativelyCachedInfo
                 &Info) { return NegativelyCached; },
          [](const DependencyScannerFSOutOfDateEntry::SizeChangedInfo &Info) {
            return SizeChanged;
          }),
      Info);
}

CXString clang_experimental_DepScanFSCacheOutOfDateEntry_getPath(
    CXDepScanFSOutOfDateEntry Entry) {
  return cxstring::createRef(unwrap(Entry)->Path);
}

static DependencyScannerFSOutOfDateEntry::SizeChangedInfo *
getOutOfDateEntrySizeChangedInfo(DependencyScannerFSOutOfDateEntry *E) {
  auto *SizeInfo =
      std::get_if<DependencyScannerFSOutOfDateEntry::SizeChangedInfo>(&E->Info);
  assert(SizeInfo && "Wrong entry kind to get size changed info!");
  return SizeInfo;
}

uint64_t clang_experimental_DepScanFSCacheOutOfDateEntry_getCachedSize(
    CXDepScanFSOutOfDateEntry Entry) {
  DependencyScannerFSOutOfDateEntry *E = unwrap(Entry);
  return getOutOfDateEntrySizeChangedInfo(E)->CachedSize;
}

uint64_t clang_experimental_DepScanFSCacheOutOfDateEntry_getActualSize(
    CXDepScanFSOutOfDateEntry Entry) {
  DependencyScannerFSOutOfDateEntry *E = unwrap(Entry);
  return getOutOfDateEntrySizeChangedInfo(E)->ActualSize;
}

void clang_experimental_DepScanFSCacheOutOfDateEntrySet_disposeSet(
    CXDepScanFSOutOfDateEntrySet Entries) {
  delete unwrap(Entries);
}

namespace {
struct DependencyScannerReproducerOptions {
  std::vector<std::string> BuildArgs;
  std::optional<std::string> ModuleName;
  std::optional<std::string> WorkingDirectory;
  std::optional<std::string> ReproducerLocation;
  bool UseUniqueReproducerName;
  CASOptions CASOpts;
  std::shared_ptr<cas::ObjectStore> CAS;
  std::shared_ptr<cas::ActionCache> Cache;

  DependencyScannerReproducerOptions(int argc, const char *const *argv,
                                     const char *ModuleName,
                                     const char *WorkingDirectory,
                                     const char *ReproducerLocation,
                                     bool UseUniqueReproducerName)
      : UseUniqueReproducerName(UseUniqueReproducerName) {
    if (argv)
      BuildArgs.assign(argv, argv + argc);
    if (ModuleName)
      this->ModuleName = ModuleName;
    if (WorkingDirectory)
      this->WorkingDirectory = WorkingDirectory;
    if (ReproducerLocation)
      this->ReproducerLocation = ReproducerLocation;
  }
};

// Helper class to capture a returnable error code and to return a formatted
// message in a provided CXString pointer.
class MessageEmitter {
  const CXErrorCode ErrorCode;
  CXString *OutputString;
  std::string Buffer;
  llvm::raw_string_ostream Stream;

public:
  MessageEmitter(CXErrorCode Code, CXString *Output)
      : ErrorCode(Code), OutputString(Output), Stream(Buffer) {}
  ~MessageEmitter() {
    if (OutputString)
      *OutputString = clang::cxstring::createDup(Buffer.c_str());
  }

  operator CXErrorCode() const { return ErrorCode; }

  template <typename T> MessageEmitter &operator<<(const T &t) {
    Stream << t;
    return *this;
  }
};
} // end anonymous namespace

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DependencyScannerReproducerOptions,
                                   CXDependencyScannerReproducerOptions)

CXDependencyScannerReproducerOptions
clang_experimental_DependencyScannerReproducerOptions_create(
    int argc, const char *const *argv, const char *ModuleName,
    const char *WorkingDirectory, const char *ReproducerLocation,
    bool UseUniqueReproducerName) {
  return wrap(new DependencyScannerReproducerOptions{
      argc, argv, ModuleName, WorkingDirectory, ReproducerLocation,
      UseUniqueReproducerName});
}

void clang_experimental_DependencyScannerReproducerOptions_setCASOptions(
    CXDependencyScannerReproducerOptions CXOptions, CXCASDatabases CDBs,
    CXCASOptions CASOpts) {
  DependencyScannerReproducerOptions &Opts = *unwrap(CXOptions);
  if (CDBs) {
    cas::WrappedCASDatabases &DBs = *cas::unwrap(CDBs);
    Opts.CASOpts = DBs.CASOpts;
    Opts.CAS = DBs.CAS;
    Opts.Cache = DBs.Cache;
  }
  if (CASOpts)
    Opts.CASOpts = *cas::unwrap(CASOpts);
}

void clang_experimental_DependencyScannerReproducerOptions_dispose(
    CXDependencyScannerReproducerOptions Options) {
  delete unwrap(Options);
}

enum CXErrorCode clang_experimental_DependencyScanner_generateReproducer(
    CXDependencyScannerReproducerOptions CXOptions, CXString *MessageOut) {
  auto Report = [MessageOut](CXErrorCode ErrorCode) -> MessageEmitter {
    return MessageEmitter(ErrorCode, MessageOut);
  };
  auto ReportFailure = [&Report]() -> MessageEmitter {
    return Report(CXError_Failure);
  };

  DependencyScannerReproducerOptions &Opts = *unwrap(CXOptions);
  if (Opts.BuildArgs.size() < 2)
    return Report(CXError_InvalidArguments) << "missing compilation command";
  if (!Opts.WorkingDirectory)
    return Report(CXError_InvalidArguments) << "missing working directory";
  if (!Opts.UseUniqueReproducerName && !Opts.ReproducerLocation)
    return Report(CXError_InvalidArguments)
           << "non-unique reproducer is allowed only in a custom location";

  std::shared_ptr<llvm::cas::ObjectStore> UpstreamCAS = Opts.CAS;
  bool IsReproducerCASBased{UpstreamCAS};
  DependencyScanningServiceOptions ServiceOpts;
  if (IsReproducerCASBased) {
    ServiceOpts.MakeVFS = [UpstreamCAS] {
      return llvm::cas::createCASProvidingFileSystem(
          UpstreamCAS, llvm::vfs::createPhysicalFileSystem());
    };
    ServiceOpts.Format = ScanningOutputFormat::FullIncludeTree;
    ServiceOpts.Compilation =
        IncludeTreeCompilation{Opts.CASOpts, Opts.CAS, Opts.Cache};
  }
  DependencyScanningService DepsService(std::move(ServiceOpts));
  DependencyScanningTool DepsTool(DepsService);

  llvm::SmallString<128> ReproScriptPath;
  int ScriptFD;
  if (Opts.ReproducerLocation) {
    if (auto EC = llvm::sys::fs::create_directories(*Opts.ReproducerLocation))
      return ReportFailure() << "failed to create a reproducer location '"
                             << *Opts.ReproducerLocation << "'\n"
                             << EC.message();
    SmallString<128> Path(*Opts.ReproducerLocation);
    llvm::sys::path::append(Path, "reproducer");
    if (Opts.UseUniqueReproducerName) {
      if (auto EC = llvm::sys::fs::createUniqueFile(Path + "-%%%%%%.sh",
                                                    ScriptFD, ReproScriptPath))
        return ReportFailure() << "failed to create a reproducer script file\n"
                              << EC.message();
    } else {
      llvm::sys::path::replace_extension(Path, "sh");
      if (auto EC = llvm::sys::fs::openFileForReadWrite(Path, ScriptFD,
                                         llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::OF_None)) {
        return ReportFailure() << "failed to create a reproducer script file\n"
                               << EC.message();
      }
      ReproScriptPath = Path;
    }
  } else {
    if (auto EC = llvm::sys::fs::createTemporaryFile(
            "reproducer", "sh", ScriptFD, ReproScriptPath)) {
      return ReportFailure() << "failed to create a reproducer script file\n"
                             << EC.message();
    }
  }
  SmallString<128> FileCachePath = ReproScriptPath;
  llvm::sys::path::replace_extension(FileCachePath, ".cache");

  std::string FileCacheName = llvm::sys::path::filename(FileCachePath).str();
  auto LookupOutput = [&FileCacheName](const ModuleDeps &MD,
                                       ModuleOutputKind MOK) -> std::string {
    if (MOK == ModuleOutputKind::DependencyTargets)
      return "reproducerdependencytargets";
    std::string CommonPrefix = FileCacheName + "/explicitly-built-modules/" +
                               MD.ID.ModuleName + "-" + MD.ID.ContextHash;
    switch (MOK) {
    case ModuleOutputKind::ModuleFile:
      return CommonPrefix + ".pcm";
    case ModuleOutputKind::DependencyFile:
      return CommonPrefix + ".d";
    case ModuleOutputKind::DiagnosticSerializationFile:
      return CommonPrefix + ".dia";
    default:
      return "";
    }
  };

  std::string S;
  llvm::raw_string_ostream OS(S);
  DiagnosticOptions DiagOpts;
  TextDiagnosticPrinter DiagConsumer(OS, DiagOpts);
  llvm::DenseSet<ModuleID> AlreadySeen;
  auto TUDepsOrErr = DepsTool.getTranslationUnitDependencies(
      Opts.BuildArgs, *Opts.WorkingDirectory, DiagConsumer, AlreadySeen,
      std::move(LookupOutput));
  if (!TUDepsOrErr)
    return ReportFailure() << "failed to generate a reproducer\n" << S;

  TranslationUnitDeps TU = *TUDepsOrErr;
  llvm::raw_fd_ostream ScriptOS(ScriptFD, /*shouldClose=*/true);
  ScriptOS << "# Original command:\n#";
  for (StringRef Arg : Opts.BuildArgs)
    ScriptOS << ' ' << Arg;
  ScriptOS << "\n\n";

  const llvm::opt::OptTable &ClangOpts = getDriverOptTable();

  ScriptOS << "# Dependencies:\n";
  // Output the executable as an environment variable with a default value, so
  // it is easier to run the reproducer with a different compiler and to
  // simplify running an individual command manually.
  std::string ReproExecutable = "\"${CLANG:-" + Opts.BuildArgs.front() + "}\"";
  auto PrintArguments = [IsReproducerCASBased, &ReproExecutable, &FileCacheName,
                         &ClangOpts](llvm::raw_fd_ostream &OS,
                                     ArrayRef<std::string> Arguments,
                                     bool RedirectOutput) {
    std::vector<const char *> CharArgs(Arguments.size());
    for (const std::string &Arg : Arguments)
      CharArgs.push_back(Arg.c_str());
    unsigned MissingArgIndex, MissingArgCount;
    llvm::opt::InputArgList ParsedArgs =
        ClangOpts.ParseArgs(CharArgs, MissingArgIndex, MissingArgCount,
                            llvm::opt::Visibility(options::CC1Option));

    // CAS-based reproducer doesn't use VFS overlays.
    bool DidAddVFSOverlay = IsReproducerCASBased;
    OS << ReproExecutable;
    for (const llvm::opt::Arg *Arg : ParsedArgs) {
      const llvm::opt::Option &Opt = Arg->getOption();
      if (Opt.matches(options::OPT_ivfsoverlay)) {
        if (!DidAddVFSOverlay) {
          OS << " -ivfsoverlay \"" << FileCacheName << "/vfs/vfs.yaml\"";
          DidAddVFSOverlay = true;
        }
      }
      if (Opt.matches(options::OPT_fcas_path)) {
        OS << " -fcas-path \"" << FileCacheName << "/cas\"";
        continue;
      }
      bool IsOutputArg = Opt.matches(options::OPT_o) ||
                         Opt.matches(options::OPT_dependency_file);
      llvm::opt::ArgStringList OutArgs;
      Arg->render(ParsedArgs, OutArgs);
      bool IsArgValue = false;
      for (const auto &OutArg : OutArgs) {
        OS << ' ';
        if (RedirectOutput && IsOutputArg && IsArgValue) {
          StringRef OutputFileName = llvm::sys::path::filename(OutArg);
          OS << " \"" << FileCacheName << '/' << OutputFileName << '"';
        } else {
          llvm::sys::printArg(OS, OutArg, /*Quote=*/true);
        }
        IsArgValue = true;
      }
    }
    if (!DidAddVFSOverlay) {
      OS << " -ivfsoverlay \"" << FileCacheName << "/vfs/vfs.yaml\"";
      DidAddVFSOverlay = true;
    }
    OS << '\n';
  };
  // Redirect the output to keep reproducers relocatable. But don't redirect
  // modules as they are already in the appropriate place (see `LookupOutput`).
  for (ModuleDeps &Dep : TU.ModuleGraph)
    PrintArguments(ScriptOS, Dep.getBuildArguments(), /*RedirectOutput=*/false);
  ScriptOS << "\n# Translation unit:\n";
  for (const Command &BuildCommand : TU.Commands)
    PrintArguments(ScriptOS, BuildCommand.Arguments, /*RedirectOutput=*/true);

  auto RealFS = llvm::vfs::getRealFileSystem();
  RealFS->setCurrentWorkingDirectory(*Opts.WorkingDirectory);

  if (IsReproducerCASBased) {
    SmallString<128> CASPath = FileCachePath;
    llvm::sys::path::append(CASPath, "cas");
    clang::CASOptions ReproducerCASOpts;
    ReproducerCASOpts.CASPath = CASPath.str();
    ReproducerCASOpts.PluginPath = Opts.CASOpts.PluginPath;
    ReproducerCASOpts.PluginOptions = Opts.CASOpts.PluginOptions;
    auto DBsOrErr = ReproducerCASOpts.CASConfiguration::createDatabases();
    if (!DBsOrErr)
      return ReportFailure() << "failed to create a CAS database\n"
                             << toString(DBsOrErr.takeError());
    std::shared_ptr<llvm::cas::ObjectStore> ReproCAS = DBsOrErr->first;

    auto transplantCASIncludeTree =
        [UpstreamCAS, ReproCAS](
            const std::optional<std::string> &IncludeTreeID) -> llvm::Error {
      if (!IncludeTreeID.has_value())
        // Missing `IncludeTreeID` likely indicates a problem but ignore it, so
        // can capture enough data to reproduce it later.
        return llvm::Error::success();
      auto IDOrErr = UpstreamCAS->parseID(*IncludeTreeID);
      if (!IDOrErr)
        return llvm::make_error<llvm::StringError>(
            "failure to parse include tree id '" + *IncludeTreeID +
                "':" + toString(IDOrErr.takeError()),
            llvm::inconvertibleErrorCode());
      std::optional<cas::ObjectRef> UpstreamRef =
          UpstreamCAS->getReference(*IDOrErr);
      if (!UpstreamRef.has_value())
        return llvm::make_error<llvm::StringError>(
            "missing include tree with ID '" + *IncludeTreeID +
                "' in the provided CAS object storage",
            llvm::inconvertibleErrorCode());
      auto ReproRefOrErr = ReproCAS->importObject(*UpstreamCAS, *UpstreamRef);
      if (!ReproRefOrErr)
        return llvm::make_error<llvm::StringError>(
            "failure to import an include tree with id '" + *IncludeTreeID +
                "':" + toString(ReproRefOrErr.takeError()),
            llvm::inconvertibleErrorCode());
      return llvm::Error::success();
    };

    if (auto Err = transplantCASIncludeTree(TU.IncludeTreeID))
      return ReportFailure()
             << "failed to transplant a translation unit include tree due to "
             << toString(std::move(Err));
    for (const ModuleDeps &ModuleDep : TU.ModuleGraph) {
      if (auto Err = transplantCASIncludeTree(ModuleDep.IncludeTreeID))
        return ReportFailure()
               << "failed to transplant a module '" + ModuleDep.ID.ModuleName +
                      "' include tree due to "
               << toString(std::move(Err));
    }
  } else {
    SmallString<128> VFSCachePath = FileCachePath;
    llvm::sys::path::append(VFSCachePath, "vfs");
    std::string VFSCachePathStr = VFSCachePath.str().str();
    llvm::FileCollector FileCollector(VFSCachePathStr,
                                      /*OverlayRoot=*/VFSCachePathStr, RealFS);
    for (const auto &FileDep : TU.FileDeps) {
      FileCollector.addFile(FileDep);
    }
    for (ModuleDeps &ModuleDep : TU.ModuleGraph) {
      ModuleDep.forEachFileDep([&FileCollector](StringRef FileDep) {
        FileCollector.addFile(FileDep);
      });
    }
    if (FileCollector.copyFiles(/*StopOnError=*/true))
      return ReportFailure()
             << "failed to copy the files used for the compilation";
    SmallString<128> VFSOverlayPath = VFSCachePath;
    llvm::sys::path::append(VFSOverlayPath, "vfs.yaml");
    if (FileCollector.writeMapping(VFSOverlayPath))
      return ReportFailure() << "failed to write a VFS overlay mapping";
  }

  return Report(CXError_Success)
         << "Created a reproducer. Sources and associated run script(s) are "
            "located at:\n  "
         << FileCachePath << "\n  " << ReproScriptPath;
}

namespace {
typedef llvm::SmallVectorImpl<clang::Module::LinkLibrary>
    DepGraphModuleLinkLibrarySet;
typedef clang::Module::LinkLibrary DepGraphModuleLinkLibrary;
} // namespace

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DepGraphModuleLinkLibrarySet,
                                   CXDepGraphModuleLinkLibrarySet)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DepGraphModuleLinkLibrary,
                                   CXDepGraphModuleLinkLibrary)

CXDepGraphModuleLinkLibrarySet
clang_experimental_DepGraphModule_getLinkLibrarySet(CXDepGraphModule CXDepMod) {
  const ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  return wrap(&ModDeps.LinkLibraries);
}

size_t clang_experimental_DepGraphModuleLinkLibrarySet_getSize(
    CXDepGraphModuleLinkLibrarySet S) {
  const DepGraphModuleLinkLibrarySet *LinkLibraries = unwrap(S);
  return LinkLibraries->size();
}

CXDepGraphModuleLinkLibrary
clang_experimental_DepGraphModuleLinkLibrarySet_getLinkLibrary(
    CXDepGraphModuleLinkLibrarySet S, size_t Idx) {
  const DepGraphModuleLinkLibrarySet *LinkLibraries = unwrap(S);
  return wrap(&(*LinkLibraries)[Idx]);
}

CXString clang_experimental_DepGraphModuleLinkLibrary_getLibrary(
    CXDepGraphModuleLinkLibrary L) {
  const DepGraphModuleLinkLibrary *Lib = unwrap(L);
  return cxstring::createRef(Lib->Library.c_str());
}

bool clang_experimental_DepGraphModuleLinkLibrary_isFramework(
    CXDepGraphModuleLinkLibrary L) {
  const DepGraphModuleLinkLibrary *Lib = unwrap(L);
  return Lib->IsFramework;
}
