//===- CASFSActionController.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CachingActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/PrefixMapper.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;
using llvm::Error;

namespace {
class CASFSActionController : public CallbackActionController {
public:
  CASFSActionController(LookupModuleOutputCallback LookupModuleOutput,
                        llvm::cas::CachingOnDiskFileSystem &CacheFS);

  llvm::Error initialize(CompilerInstance &ScanInstance,
                         CompilerInvocation &NewInvocation) override;
  llvm::Error finalize(CompilerInstance &ScanInstance,
                       CompilerInvocation &NewInvocation) override;
  llvm::Error
  initializeModuleBuild(CompilerInstance &ModuleScanInstance) override;
  llvm::Error
  finalizeModuleBuild(CompilerInstance &ModuleScanInstance) override;
  llvm::Error finalizeModuleInvocation(CowCompilerInvocation &CI,
                                       const ModuleDeps &MD) override;

private:
  llvm::cas::CachingOnDiskFileSystem &CacheFS;
  std::optional<llvm::TreePathPrefixMapper> Mapper;
  CASOptions CASOpts;
};
} // anonymous namespace

CASFSActionController::CASFSActionController(
    LookupModuleOutputCallback LookupModuleOutput,
    llvm::cas::CachingOnDiskFileSystem &CacheFS)
    : CallbackActionController(std::move(LookupModuleOutput)),
      CacheFS(CacheFS) {}

Error CASFSActionController::initialize(CompilerInstance &ScanInstance,
                                        CompilerInvocation &NewInvocation) {
  // Setup prefix mapping.
  Mapper.emplace(&CacheFS);
  DepscanPrefixMapping::configurePrefixMapper(NewInvocation, *Mapper);

  const PreprocessorOptions &PPOpts = ScanInstance.getPreprocessorOpts();
  if (!PPOpts.Includes.empty() || !PPOpts.ImplicitPCHInclude.empty())
    addReversePrefixMappingFileSystem(*Mapper, ScanInstance);

  CacheFS.trackNewAccesses();
  if (auto CWD =
          ScanInstance.getVirtualFileSystem().getCurrentWorkingDirectory())
    CacheFS.setCurrentWorkingDirectory(*CWD);
  // Track paths that are accessed by the scanner before we reach here.
  for (const auto &File : ScanInstance.getHeaderSearchOpts().VFSOverlayFiles)
    (void)CacheFS.status(File);
  // Enable caching in the resulting commands.
  ScanInstance.getFrontendOpts().CacheCompileJob = true;
  CASOpts = ScanInstance.getCASOpts();
  return Error::success();
}

/// Ensure that all files reachable from imported modules/pch are tracked.
/// These could be loaded lazily during compilation.
static void trackASTFileInputs(CompilerInstance &CI,
                               llvm::cas::CachingOnDiskFileSystem &CacheFS) {
  auto Reader = CI.getASTReader();
  if (!Reader)
    return;

  for (serialization::ModuleFile &MF : Reader->getModuleManager()) {
    Reader->visitInputFiles(
        MF, /*IncludeSystem=*/true, /*Complain=*/false,
        [](const serialization::InputFile &IF, bool isSystem) {
          // Visiting input files triggers the file system lookup.
        });
  }
}

/// Ensure files that are not accessed during the scan (or accessed before the
/// tracking scope) are tracked.
static void trackFilesCommon(CompilerInstance &CI,
                             llvm::cas::CachingOnDiskFileSystem &CacheFS) {
  trackASTFileInputs(CI, CacheFS);

  // Exclude the module cache from tracking. The implicit build pcms should
  // not be needed after scanning.
  if (!CI.getHeaderSearchOpts().ModuleCachePath.empty())
    (void)CacheFS.excludeFromTracking(CI.getHeaderSearchOpts().ModuleCachePath);

  // Normally this would be looked up while creating the VFS, but implicit
  // modules share their VFS and it happens too early for the TU scan.
  for (const auto &File : CI.getHeaderSearchOpts().VFSOverlayFiles)
    (void)CacheFS.status(File);

  StringRef Sysroot = CI.getHeaderSearchOpts().Sysroot;
  if (!Sysroot.empty()) {
    // Include 'SDKSettings.json', if it exists, to accomodate availability
    // checks during the compilation.
    llvm::SmallString<256> FilePath = Sysroot;
    llvm::sys::path::append(FilePath, "SDKSettings.json");
    (void)CacheFS.status(FilePath);
  }
}

Error CASFSActionController::finalize(CompilerInstance &ScanInstance,
                                      CompilerInvocation &NewInvocation) {
  // Handle profile mappings.
  (void)CacheFS.status(NewInvocation.getCodeGenOpts().ProfileInstrumentUsePath);
  (void)CacheFS.status(NewInvocation.getCodeGenOpts().SampleProfileFile);
  (void)CacheFS.status(NewInvocation.getCodeGenOpts().ProfileRemappingFile);

  trackFilesCommon(ScanInstance, CacheFS);

  auto CASFileSystemRootID = CacheFS.createTreeFromNewAccesses(
      [&](const llvm::vfs::CachedDirectoryEntry &Entry,
          SmallVectorImpl<char> &Storage) {
        return Mapper->mapDirEntry(Entry, Storage);
      });
  if (!CASFileSystemRootID)
    return CASFileSystemRootID.takeError();

  configureInvocationForCaching(NewInvocation, CASOpts,
                                CASFileSystemRootID->getID().toString(),
                                CacheFS.getCurrentWorkingDirectory().get(),
                                /*ProduceIncludeTree=*/false);

  if (Mapper)
    DepscanPrefixMapping::remapInvocationPaths(NewInvocation, *Mapper);

  return Error::success();
}

Error CASFSActionController::initializeModuleBuild(
    CompilerInstance &ModuleScanInstance) {

  CacheFS.trackNewAccesses();
  // If the working directory is not otherwise accessed by the module build,
  // we still need it due to -fcas-fs-working-directory being set.
  if (auto CWD = CacheFS.getCurrentWorkingDirectory())
    (void)CacheFS.status(*CWD);

  return Error::success();
}

Error CASFSActionController::finalizeModuleBuild(
    CompilerInstance &ModuleScanInstance) {
  trackFilesCommon(ModuleScanInstance, CacheFS);

  std::optional<cas::CASID> RootID;
  auto E = CacheFS
               .createTreeFromNewAccesses(
                   [&](const llvm::vfs::CachedDirectoryEntry &Entry,
                       SmallVectorImpl<char> &Storage) {
                     return Mapper->mapDirEntry(Entry, Storage);
                   })
               .moveInto(RootID);
  if (E)
    return E;

#ifndef NDEBUG
  Module *M = ModuleScanInstance.getPreprocessor().getCurrentModule();
  assert(M && "finalizing without a module");
#endif

  ModuleScanInstance.getASTContext().setCASFileSystemRootID(RootID->toString());
  return Error::success();
}

Error CASFSActionController::finalizeModuleInvocation(
    CowCompilerInvocation &CowCI, const ModuleDeps &MD) {
  // TODO: Avoid this copy.
  CompilerInvocation CI(CowCI);

  if (auto ID = MD.CASFileSystemRootID) {
    configureInvocationForCaching(CI, CASOpts, ID->toString(),
                                  CacheFS.getCurrentWorkingDirectory().get(),
                                  /*ProduceIncludeTree=*/false);
  }

  if (Mapper)
    DepscanPrefixMapping::remapInvocationPaths(CI, *Mapper);

  CowCI = CI;
  return llvm::Error::success();
}

std::unique_ptr<DependencyActionController>
dependencies::createCASFSActionController(
    LookupModuleOutputCallback LookupModuleOutput,
    llvm::cas::CachingOnDiskFileSystem &CacheFS) {
  return std::make_unique<CASFSActionController>(LookupModuleOutput, CacheFS);
}
