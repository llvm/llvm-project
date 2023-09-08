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

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/Support/Process.h"

using namespace clang;
using namespace clang::tooling::dependencies;

namespace {
struct DependencyScannerServiceOptions {
  ScanningOutputFormat ConfiguredFormat = ScanningOutputFormat::Full;
  CASOptions CASOpts;
  std::shared_ptr<cas::ObjectStore> CAS;
  std::shared_ptr<cas::ActionCache> Cache;

  ScanningOutputFormat getFormat() const;
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

void clang_experimental_ModuleDependencySet_dispose(
    CXModuleDependencySet *MDS) {
  for (int I = 0; I < MDS->Count; ++I) {
    CXModuleDependency &MD = MDS->Modules[I];
    clang_disposeString(MD.Name);
    clang_disposeString(MD.ContextHash);
    clang_disposeString(MD.ModuleMapPath);
    clang_disposeStringSet(MD.FileDeps);
    clang_disposeStringSet(MD.ModuleDeps);
    clang_disposeStringSet(MD.BuildArguments);
  }
  delete[] MDS->Modules;
  delete MDS;
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

CXDependencyScannerService
clang_experimental_DependencyScannerService_create_v0(CXDependencyMode Format) {
  // FIXME: Pass default CASOpts and nullptr as CachingOnDiskFileSystem now.
  CASOptions CASOpts;
  IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> FS;
  return wrap(new DependencyScanningService(
      ScanningMode::DependencyDirectivesScan, unwrap(Format), CASOpts,
      /*CAS=*/nullptr, /*ActionCache=*/nullptr, FS,
      /*ReuseFilemanager=*/false));
}

ScanningOutputFormat DependencyScannerServiceOptions::getFormat() const {
  if (ConfiguredFormat != ScanningOutputFormat::Full)
    return ConfiguredFormat;

  if (!CAS || !Cache)
    return ConfiguredFormat;

  if (llvm::sys::Process::GetEnv("CLANG_CACHE_USE_INCLUDE_TREE"))
    return ScanningOutputFormat::FullIncludeTree;

  if (llvm::sys::Process::GetEnv("CLANG_CACHE_USE_CASFS_DEPSCAN"))
    return ScanningOutputFormat::FullTree;

  // Use include-tree by default.
  return ScanningOutputFormat::FullIncludeTree;
}

CXDependencyScannerService
clang_experimental_DependencyScannerService_create_v1(
    CXDependencyScannerServiceOptions Opts) {
  // FIXME: Pass default CASOpts and nullptr as CachingOnDiskFileSystem now.
  std::shared_ptr<llvm::cas::ObjectStore> CAS = unwrap(Opts)->CAS;
  std::shared_ptr<llvm::cas::ActionCache> Cache = unwrap(Opts)->Cache;
  IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> FS;
  ScanningOutputFormat Format = unwrap(Opts)->getFormat();
  bool IsCASFSOutput = Format == ScanningOutputFormat::Tree ||
                       Format == ScanningOutputFormat::FullTree;
  if (CAS && Cache && IsCASFSOutput) {
    assert(unwrap(Opts)->CASOpts.getKind() != CASOptions::UnknownCAS &&
           "CAS and ActionCache must match CASOptions");
    FS = llvm::cantFail(
        llvm::cas::createCachingOnDiskFileSystem(CAS));
  }
  return wrap(new DependencyScanningService(
      ScanningMode::DependencyDirectivesScan, Format, unwrap(Opts)->CASOpts,
      std::move(CAS), std::move(Cache), std::move(FS),
      /*ReuseFilemanager=*/false));
}

void clang_experimental_DependencyScannerService_dispose_v0(
    CXDependencyScannerService Service) {
  delete unwrap(Service);
}

void clang_experimental_FileDependencies_dispose(CXFileDependencies *ID) {
  clang_disposeString(ID->ContextHash);
  clang_disposeStringSet(ID->FileDeps);
  clang_disposeStringSet(ID->ModuleDeps);
  clang_disposeStringSet(ID->BuildArguments);
  delete ID;
}

void clang_experimental_FileDependenciesList_dispose(
    CXFileDependenciesList *FD) {
  for (size_t I = 0; I < FD->NumCommands; ++I) {
    clang_disposeString(FD->Commands[I].ContextHash);
    clang_disposeStringSet(FD->Commands[I].FileDeps);
    clang_disposeStringSet(FD->Commands[I].ModuleDeps);
    clang_disposeString(FD->Commands[I].Executable);
    clang_disposeStringSet(FD->Commands[I].BuildArguments);
  }
  delete[] FD->Commands;
  delete FD;
}

CXDependencyScannerWorker clang_experimental_DependencyScannerWorker_create_v0(
    CXDependencyScannerService Service) {
  ScanningOutputFormat Format = unwrap(Service)->getFormat();
  bool IsIncludeTreeOutput = Format == ScanningOutputFormat::IncludeTree ||
                             Format == ScanningOutputFormat::FullIncludeTree;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPhysicalFileSystem();
  if (IsIncludeTreeOutput)
    FS = llvm::cas::createCASProvidingFileSystem(unwrap(Service)->getCAS(),
                                                 std::move(FS));

  return wrap(new DependencyScanningWorker(*unwrap(Service), FS));
}

void clang_experimental_DependencyScannerWorker_dispose_v0(
    CXDependencyScannerWorker Worker) {
  delete unwrap(Worker);
}

using HandleTUDepsCallback = llvm::function_ref<void(TranslationUnitDeps)>;

static CXErrorCode getFullDependencies(DependencyScanningWorker *Worker,
                                       ArrayRef<std::string> Compilation,
                                       const char *WorkingDirectory,
                                       CXModuleDiscoveredCallback *MDC,
                                       void *Context, CXString *Error,
                                       DiagnosticConsumer *DiagConsumer,
                                       LookupModuleOutputCallback LookupOutput,
                                       std::optional<StringRef> ModuleName,
                                       HandleTUDepsCallback HandleTUDeps) {
  llvm::DenseSet<ModuleID> AlreadySeen;
  FullDependencyConsumer DepConsumer(AlreadySeen);
  auto Controller = DependencyScanningTool::createActionController(
      *Worker, std::move(LookupOutput));

#ifndef NDEBUG
  bool HasDiagConsumer = DiagConsumer;
  bool HasError = Error;
  assert(HasDiagConsumer ^ HasError && "Both DiagConsumer and Error provided");
#endif

  if (DiagConsumer) {
    bool Result =
        Worker->computeDependencies(WorkingDirectory, Compilation, DepConsumer,
                                    *Controller, *DiagConsumer, ModuleName);
    if (!Result)
      return CXError_Failure;
  } else if (Error) {
    auto Result = Worker->computeDependencies(
        WorkingDirectory, Compilation, DepConsumer, *Controller, ModuleName);
    if (Result) {
      *Error = cxstring::createDup(llvm::toString(std::move(Result)));
      return CXError_Failure;
    }
  }

  TranslationUnitDeps TU = DepConsumer.takeTranslationUnitDeps();

  if (MDC && !TU.ModuleGraph.empty()) {
    CXModuleDependencySet *MDS = new CXModuleDependencySet;
    MDS->Count = TU.ModuleGraph.size();
    MDS->Modules = new CXModuleDependency[MDS->Count];
    for (int I = 0; I < MDS->Count; ++I) {
      CXModuleDependency &M = MDS->Modules[I];
      ModuleDeps &MD = TU.ModuleGraph[I];
      M.Name = cxstring::createDup(MD.ID.ModuleName);
      M.ContextHash = cxstring::createDup(MD.ID.ContextHash);
      M.ModuleMapPath = cxstring::createDup(MD.ClangModuleMapFile);
      M.FileDeps = cxstring::createSet(MD.FileDeps);
      std::vector<std::string> Modules;
      for (ModuleID &MID : MD.ClangModuleDeps)
        Modules.push_back(MID.ModuleName + ":" + MID.ContextHash);
      M.ModuleDeps = cxstring::createSet(Modules);
      M.BuildArguments = cxstring::createSet(MD.getBuildArguments());
    }
    MDC(Context, MDS);
  }

  HandleTUDeps(std::move(TU));
  return CXError_Success;
}

static CXErrorCode getFileDependencies(CXDependencyScannerWorker W, int argc,
                                       const char *const *argv,
                                       const char *WorkingDirectory,
                                       CXModuleDiscoveredCallback *MDC,
                                       void *Context, CXString *Error,
                                       DiagnosticConsumer *DiagConsumer,
                                       LookupModuleOutputCallback LookupOutput,
                                       std::optional<StringRef> ModuleName,
                                       HandleTUDepsCallback HandleTUDeps) {
  if (!W || argc < 2 || !argv)
    return CXError_InvalidArguments;

  DependencyScanningWorker *Worker = unwrap(W);

  if (Worker->getScanningFormat() != ScanningOutputFormat::Full &&
      Worker->getScanningFormat() != ScanningOutputFormat::FullTree &&
      Worker->getScanningFormat() != ScanningOutputFormat::FullIncludeTree)
    return CXError_InvalidArguments;

  std::vector<std::string> Compilation{argv, argv + argc};

  return getFullDependencies(Worker, Compilation, WorkingDirectory, MDC,
                             Context, Error, DiagConsumer, LookupOutput,
                             ModuleName, HandleTUDeps);
}

namespace {
class OutputLookup {
public:
  OutputLookup(void *MLOContext, CXModuleLookupOutputCallback *MLO)
      : MLOContext(MLOContext), MLO(MLO) {}
  std::string lookupModuleOutput(const ModuleID &ID, ModuleOutputKind MOK);

private:
  llvm::DenseMap<ModuleID, std::string> PCMPaths;
  void *MLOContext;
  CXModuleLookupOutputCallback *MLO;
};
} // end anonymous namespace

CXFileDependencies *
clang_experimental_DependencyScannerWorker_getFileDependencies_v3(
    CXDependencyScannerWorker W, int argc, const char *const *argv,
    const char *ModuleName, const char *WorkingDirectory, void *MDCContext,
    CXModuleDiscoveredCallback *MDC, void *MLOContext,
    CXModuleLookupOutputCallback *MLO, unsigned, CXString *Error) {
  OutputLookup OL(MLOContext, MLO);
  auto LookupOutputs = [&](const ModuleID &ID, ModuleOutputKind MOK) {
    return OL.lookupModuleOutput(ID, MOK);
  };
  CXFileDependencies *FDeps = nullptr;
  CXErrorCode Result = getFileDependencies(
      W, argc, argv, WorkingDirectory, MDC, MDCContext, Error, nullptr,
      LookupOutputs,
      ModuleName ? std::optional<StringRef>(ModuleName) : std::nullopt,
      [&](TranslationUnitDeps TU) {
        assert(!TU.DriverCommandLine.empty());
        std::vector<std::string> Modules;
        for (const ModuleID &MID : TU.ClangModuleDeps)
          Modules.push_back(MID.ModuleName + ":" + MID.ContextHash);
        FDeps = new CXFileDependencies;
        FDeps->ContextHash = cxstring::createDup(TU.ID.ContextHash);
        FDeps->FileDeps = cxstring::createSet(TU.FileDeps);
        FDeps->ModuleDeps = cxstring::createSet(Modules);
        FDeps->BuildArguments = cxstring::createSet(TU.DriverCommandLine);
      });
  assert(Result != CXError_Success || FDeps);
  (void)Result;
  return FDeps;
}

CXErrorCode clang_experimental_DependencyScannerWorker_getFileDependencies_v4(
    CXDependencyScannerWorker W, int argc, const char *const *argv,
    const char *ModuleName, const char *WorkingDirectory, void *MDCContext,
    CXModuleDiscoveredCallback *MDC, void *MLOContext,
    CXModuleLookupOutputCallback *MLO, unsigned, CXFileDependenciesList **Out,
    CXString *Error) {
  OutputLookup OL(MLOContext, MLO);
  auto LookupOutputs = [&](const ModuleID &ID, ModuleOutputKind MOK) {
    return OL.lookupModuleOutput(ID, MOK);
  };

  if (!Out)
    return CXError_InvalidArguments;
  *Out = nullptr;

  CXErrorCode Result = getFileDependencies(
      W, argc, argv, WorkingDirectory, MDC, MDCContext, Error, nullptr,
      LookupOutputs,
      ModuleName ? std::optional<StringRef>(ModuleName) : std::nullopt,
      [&](TranslationUnitDeps TU) {
        assert(TU.DriverCommandLine.empty());
        std::vector<std::string> Modules;
        for (const ModuleID &MID : TU.ClangModuleDeps)
          Modules.push_back(MID.ModuleName + ":" + MID.ContextHash);
        auto *Commands = new CXTranslationUnitCommand[TU.Commands.size()];
        for (size_t I = 0, E = TU.Commands.size(); I < E; ++I) {
          Commands[I].ContextHash = cxstring::createDup(TU.ID.ContextHash);
          Commands[I].FileDeps = cxstring::createSet(TU.FileDeps);
          Commands[I].ModuleDeps = cxstring::createSet(Modules);
          Commands[I].Executable =
              cxstring::createDup(TU.Commands[I].Executable);
          Commands[I].BuildArguments =
              cxstring::createSet(TU.Commands[I].Arguments);
        }
        *Out = new CXFileDependenciesList{TU.Commands.size(), Commands};
      });

  return Result;
}

namespace {

struct DependencyScannerWorkerScanSettings {
  int argc;
  const char *const *argv;
  const char *ModuleName;
  const char *WorkingDirectory;
  void *MLOContext;
  CXModuleLookupOutputCallback *MLO;
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
  ModuleDeps *ModDeps;
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
  CXModuleLookupOutputCallback *MLO = Settings.MLO;

  OutputLookup OL(MLOContext, MLO);
  auto LookupOutputs = [&](const ModuleID &ID, ModuleOutputKind MOK) {
    return OL.lookupModuleOutput(ID, MOK);
  };

  if (!Out)
    return CXError_InvalidArguments;
  *Out = nullptr;

  DependencyGraph *DepGraph = new DependencyGraph();

  // We capture diagnostics as a serialized diagnostics buffer, so that we don't
  // need to keep a valid SourceManager in order to access diagnostic locations.
  auto DiagOpts = llvm::makeIntrusiveRefCnt<DiagnosticOptions>();
  auto DiagOS =
      std::make_unique<llvm::raw_svector_ostream>(DepGraph->SerialDiagBuf);
  std::unique_ptr<DiagnosticConsumer> SerialDiagConsumer =
      serialized_diags::create("<diagnostics>", DiagOpts.get(),
                               /*MergeChildRecords=*/false, std::move(DiagOS));

  CXErrorCode Result = getFileDependencies(
      W, argc, argv, WorkingDirectory, /*MDC=*/nullptr, /*MDCContext=*/nullptr,
      /*Error=*/nullptr, SerialDiagConsumer.get(), LookupOutputs,
      ModuleName ? std::optional<StringRef>(ModuleName) : std::nullopt,
      [&](TranslationUnitDeps TU) { DepGraph->TUDeps = std::move(TU); });

  *Out = wrap(DepGraph);
  return Result;
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
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  return wrap(new DependencyGraphModule{&TUDeps.ModuleGraph[Index]});
}

void clang_experimental_DepGraphModule_dispose(CXDepGraphModule CXDepMod) {
  delete unwrap(CXDepMod);
}

const char *
clang_experimental_DepGraphModule_getName(CXDepGraphModule CXDepMod) {
  ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  return ModDeps.ID.ModuleName.c_str();
}

const char *
clang_experimental_DepGraphModule_getContextHash(CXDepGraphModule CXDepMod) {
  ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  return ModDeps.ID.ContextHash.c_str();
}

const char *
clang_experimental_DepGraphModule_getModuleMapPath(CXDepGraphModule CXDepMod) {
  ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  if (ModDeps.ClangModuleMapFile.empty())
    return nullptr;
  return ModDeps.ClangModuleMapFile.c_str();
}

CXCStringArray
clang_experimental_DepGraphModule_getFileDeps(CXDepGraphModule CXDepMod) {
  ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  return unwrap(CXDepMod)->StrMgr.createCStringsRef(ModDeps.FileDeps);
}

CXCStringArray
clang_experimental_DepGraphModule_getModuleDeps(CXDepGraphModule CXDepMod) {
  ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  std::vector<std::string> Modules;
  Modules.reserve(ModDeps.ClangModuleDeps.size());
  for (const ModuleID &MID : ModDeps.ClangModuleDeps)
    Modules.push_back(MID.ModuleName + ":" + MID.ContextHash);
  return unwrap(CXDepMod)->StrMgr.createCStringsOwned(std::move(Modules));
}

CXCStringArray
clang_experimental_DepGraphModule_getBuildArguments(CXDepGraphModule CXDepMod) {
  ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  return unwrap(CXDepMod)->StrMgr.createCStringsRef(
      ModDeps.getBuildArguments());
}

const char *
clang_experimental_DepGraphModule_getCacheKey(CXDepGraphModule CXDepMod) {
  ModuleDeps &ModDeps = *unwrap(CXDepMod)->ModDeps;
  if (ModDeps.ModuleCacheKey)
    return ModDeps.ModuleCacheKey->c_str();
  return nullptr;
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

const char *clang_experimental_DepGraph_getTUContextHash(CXDepGraph Graph) {
  TranslationUnitDeps &TUDeps = unwrap(Graph)->TUDeps;
  return TUDeps.ID.ContextHash.c_str();
}

CXDiagnosticSet clang_experimental_DepGraph_getDiagnostics(CXDepGraph Graph) {
  return unwrap(Graph)->getDiagnosticSet();
}

static std::string lookupModuleOutput(const ModuleID &ID, ModuleOutputKind MOK,
                                      void *MLOContext,
                                      CXModuleLookupOutputCallback *MLO) {
  SmallVector<char, 256> Buffer(256);
  size_t Len = MLO(MLOContext, ID.ModuleName.c_str(), ID.ContextHash.c_str(),
                   wrap(MOK), Buffer.data(), Buffer.size());
  if (Len > Buffer.size()) {
    Buffer.resize(Len);
    Len = MLO(MLOContext, ID.ModuleName.c_str(), ID.ContextHash.c_str(),
              wrap(MOK), Buffer.data(), Buffer.size());
  }
  return std::string(Buffer.begin(), Len);
}

std::string OutputLookup::lookupModuleOutput(const ModuleID &ID,
                                             ModuleOutputKind MOK) {
  if (MOK != ModuleOutputKind::ModuleFile)
    return ::lookupModuleOutput(ID, MOK, MLOContext, MLO);
  // PCM paths are looked up repeatedly, so cache them.
  auto PCMPath = PCMPaths.insert({ID, ""});
  if (PCMPath.second)
    PCMPath.first->second = ::lookupModuleOutput(ID, MOK, MLOContext, MLO);
  return PCMPath.first->second;
}
