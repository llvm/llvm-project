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

#include "CXString.h"

#include "clang-c/Dependencies.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"

using namespace clang;
using namespace clang::tooling::dependencies;

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

void clang_experimental_ModuleDependencySet_dispose(
    CXModuleDependencySet *MDS) {
  for (int I = 0; I < MDS->Count; ++I) {
    CXModuleDependency &MD = MDS->Modules[I];
    clang_disposeString(MD.Name);
    clang_disposeString(MD.ContextHash);
    clang_disposeString(MD.ModuleMapPath);
    clang_disposeStringSet(MD.FileDeps);
    clang_disposeStringSet(MD.ModuleDeps);
  }
  delete[] MDS->Modules;
  delete MDS;
}

CXDependencyScannerService
clang_experimental_DependencyScannerService_create_v0(CXDependencyMode Format) {
  return wrap(new DependencyScanningService(
      ScanningMode::MinimizedSourcePreprocessing, unwrap(Format),
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
  delete ID;
}

CXDependencyScannerWorker clang_experimental_DependencyScannerWorker_create_v0(
    CXDependencyScannerService Service) {
  return wrap(new DependencyScanningWorker(*unwrap(Service)));
}

void clang_experimental_DependencyScannerWorker_dispose_v0(
    CXDependencyScannerWorker Worker) {
  delete unwrap(Worker);
}

static CXFileDependencies *
getFlatDependencies(DependencyScanningWorker *Worker,
                    ArrayRef<std::string> Compilation,
                    const char *WorkingDirectory, CXString *error,
                    llvm::Optional<StringRef> ModuleName = None) {
  // TODO: Implement flat deps.
  return nullptr;
}

namespace {
class FullDependencyConsumer : public DependencyConsumer {
public:
  FullDependencyConsumer(const llvm::StringSet<> &AlreadySeen)
      : AlreadySeen(AlreadySeen) {}

  void
  handleDependencyOutputOpts(const DependencyOutputOptions &Opts) override {
    OutputPaths = Opts.Targets;
  }

  void handleFileDependency(StringRef File) override {
    Dependencies.push_back(std::string(File));
  }

  void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {
    PrebuiltModuleDeps.emplace_back(std::move(PMD));
  }

  void handleModuleDependency(ModuleDeps MD) override {
    ClangModuleDeps[MD.ID.ContextHash + MD.ID.ModuleName] = std::move(MD);
  }

  void handleContextHash(std::string Hash) override {
    ContextHash = std::move(Hash);
  }

  FullDependenciesResult getFullDependencies() const {
    FullDependencies FD;

    FD.ID.ContextHash = std::move(ContextHash);

    FD.FileDeps.assign(Dependencies.begin(), Dependencies.end());

    for (auto &&M : ClangModuleDeps) {
      auto &MD = M.second;
      if (MD.ImportedByMainFile)
        FD.ClangModuleDeps.push_back(MD.ID);
    }

    FD.PrebuiltModuleDeps = std::move(PrebuiltModuleDeps);

    FullDependenciesResult FDR;

    for (auto &&M : ClangModuleDeps) {
      // TODO: Avoid handleModuleDependency even being called for modules
      //   we've already seen.
      if (AlreadySeen.count(M.first))
        continue;
      FDR.DiscoveredModules.push_back(std::move(M.second));
    }

    FDR.FullDeps = std::move(FD);
    return FDR;
  }

private:
  std::vector<std::string> Dependencies;
  std::vector<PrebuiltModuleDep> PrebuiltModuleDeps;
  std::unordered_map<std::string, ModuleDeps> ClangModuleDeps;
  std::string ContextHash;
  std::vector<std::string> OutputPaths;
  const llvm::StringSet<> &AlreadySeen;
};
} // namespace

using BuildArgsFn =
    llvm::function_ref<std::vector<std::string>(const ModuleDeps &)>;

static CXFileDependencies *getFullDependencies(
    DependencyScanningWorker *Worker, ArrayRef<std::string> Compilation,
    const char *WorkingDirectory, CXModuleDiscoveredCallback *MDC,
    void *Context, CXString *error, BuildArgsFn GetBuildArgs,
    llvm::Optional<StringRef> ModuleName = None) {
  FullDependencyConsumer Consumer(Worker->AlreadySeen);
  llvm::Error Result = Worker->computeDependenciesForClangInvocation(
      WorkingDirectory, Compilation, Consumer, ModuleName);

  if (Result) {
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    llvm::handleAllErrors(std::move(Result),
                          [&](const llvm::ErrorInfoBase &EI) { EI.log(OS); });
    *error = cxstring::createDup(OS.str());
    return nullptr;
  }

  FullDependenciesResult FDR = Consumer.getFullDependencies();

  if (!FDR.DiscoveredModules.empty()) {
    CXModuleDependencySet *MDS = new CXModuleDependencySet;
    MDS->Count = FDR.DiscoveredModules.size();
    MDS->Modules = new CXModuleDependency[MDS->Count];
    for (int I = 0; I < MDS->Count; ++I) {
      CXModuleDependency &M = MDS->Modules[I];
      const ModuleDeps &MD = FDR.DiscoveredModules[I];
      M.Name = cxstring::createDup(MD.ID.ModuleName);
      M.ContextHash = cxstring::createDup(MD.ID.ContextHash);
      M.ModuleMapPath = cxstring::createDup(MD.ClangModuleMapFile);
      M.FileDeps = cxstring::createSet(MD.FileDeps);
      std::vector<std::string> Modules;
      for (const ModuleID &MID : MD.ClangModuleDeps)
        Modules.push_back(MID.ModuleName + ":" + MID.ContextHash);
      M.ModuleDeps = cxstring::createSet(Modules);
      M.BuildArguments = cxstring::createSet(GetBuildArgs(MD));
    }
    MDC(Context, MDS);
  }

  const FullDependencies &FD = FDR.FullDeps;
  CXFileDependencies *FDeps = new CXFileDependencies;
  FDeps->ContextHash = cxstring::createDup(FD.ID.ContextHash);
  FDeps->FileDeps = cxstring::createSet(FD.FileDeps);
  std::vector<std::string> Modules;
  for (const ModuleID &MID : FD.ClangModuleDeps)
    Modules.push_back(MID.ModuleName + ":" + MID.ContextHash);
  FDeps->ModuleDeps = cxstring::createSet(Modules);
  FDeps->AdditionalArguments =
    cxstring::createSet(FD.getAdditionalArgsWithoutModulePaths());
  return FDeps;
}

static CXFileDependencies *
getFileDependencies(CXDependencyScannerWorker W, int argc,
                    const char *const *argv, const char *WorkingDirectory,
                    CXModuleDiscoveredCallback *MDC, void *Context,
                    CXString *error, BuildArgsFn GetBuildArgs,
                    llvm::Optional<StringRef> ModuleName = None) {
  if (!W || argc < 2)
    return nullptr;
  if (error)
    *error = cxstring::createEmpty();

  DependencyScanningWorker *Worker = unwrap(W);

  std::vector<std::string> Compilation{argv, argv + argc};

  if (Worker->getFormat() == ScanningOutputFormat::Full)
    return getFullDependencies(Worker, Compilation, WorkingDirectory, MDC,
                               Context, error, GetBuildArgs, ModuleName);
  return getFlatDependencies(Worker, Compilation, WorkingDirectory, error,
                             ModuleName);
}

CXFileDependencies *
clang_experimental_DependencyScannerWorker_getFileDependencies_v0(
    CXDependencyScannerWorker W, int argc, const char *const *argv,
    const char *WorkingDirectory, CXModuleDiscoveredCallback *MDC,
    void *Context, CXString *error) {
  return getFileDependencies(W, argc, argv, WorkingDirectory, MDC, Context,
                             error, [](const ModuleDeps &MD) {
                               return MD.getAdditionalArgsWithoutModulePaths();
                             });
}

CXFileDependencies *
clang_experimental_DependencyScannerWorker_getFileDependencies_v1(
    CXDependencyScannerWorker W, int argc, const char *const *argv,
    const char *WorkingDirectory, CXModuleDiscoveredCallback *MDC,
    void *Context, CXString *error) {
  return getFileDependencies(
      W, argc, argv, WorkingDirectory, MDC, Context, error,
      [](const ModuleDeps &MD) {
        return MD.getCanonicalCommandLineWithoutModulePaths();
      });
}

CXFileDependencies *
clang_experimental_DependencyScannerWorker_getDependenciesByModuleName_v0(
    CXDependencyScannerWorker W, int argc, const char *const *argv,
    const char *ModuleName, const char *WorkingDirectory,
    CXModuleDiscoveredCallback *MDC, void *Context, CXString *error) {
  return getFileDependencies(
      W, argc, argv, WorkingDirectory, MDC, Context, error,
      [](const ModuleDeps &MD) {
        return MD.getCanonicalCommandLineWithoutModulePaths();
      },
      StringRef(ModuleName));
}
