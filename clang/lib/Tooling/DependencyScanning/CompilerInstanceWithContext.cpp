//===- CompilerInstanceWithContext.cpp - clang scanning compiler instance -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/CompilerInstanceWithContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "llvm/TargetParser/Host.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

const std::string CompilerInstanceWithContext::FakeFileBuffer =
    std::string(MAX_NUM_NAMES, ' ');

llvm::Error CompilerInstanceWithContext::initialize() {
  // Virtual file system setup
  // - Set the current working directory.
  Worker.BaseFS->setCurrentWorkingDirectory(CWD);
  OverlayFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(Worker.BaseFS);
  InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory(CWD);

  // - Create the fake file as scanning input source file and setup overlay
  //   FS.
  SmallString<128> FakeInputPath;
  llvm::sys::fs::createUniquePath("ScanningCI-%%%%%%%%.input", FakeInputPath,
                                  /*MakeAbsolute=*/false);
  InMemoryFS->addFile(FakeInputPath, 0,
                      llvm::MemoryBuffer::getMemBuffer(FakeFileBuffer));
  InMemoryOverlay = InMemoryFS;
  // TODO: we need to handle CAS/CASFS here.
  //    if (Worker.CAS && !Worker.DepCASFS)
  //     InMemoryOverlay = llvm::cas::createCASProvidingFileSystem(
  //         Worker.CAS, std::move(InMemoryFS));
  OverlayFS->pushOverlay(InMemoryOverlay);

  // Augument the command line.
  CommandLine.emplace_back(FakeInputPath);

  // Create the file manager, the diagnostics engine, and the source manager.
  FileMgr = std::make_unique<FileManager>(FileSystemOptions{}, OverlayFS);
  DiagnosticOutput.clear();
  auto DiagOpts = createDiagOptions(CommandLine);
  DiagPrinter = std::make_unique<TextDiagnosticPrinter>(DiagnosticsOS,
                                                        *(DiagOpts.release()));
  std::vector<const char *> CCommandLine(CommandLine.size(), nullptr);
  llvm::transform(CommandLine, CCommandLine.begin(),
                  [](const std::string &Str) { return Str.c_str(); });
  DiagOpts = CreateAndPopulateDiagOpts(CCommandLine);
  sanitizeDiagOpts(*DiagOpts);
  Diags = CompilerInstance::createDiagnostics(*OverlayFS, *(DiagOpts.release()),
                                              DiagPrinter.get(),
                                              /*ShouldOwnClient=*/false);
  SrcMgr = std::make_unique<SourceManager>(*Diags, *FileMgr);
  Diags->setSourceManager(SrcMgr.get());

  // Create the compiler invocation.
  Driver = std::make_unique<driver::Driver>(
      CCommandLine[0], llvm::sys::getDefaultTargetTriple(), *Diags,
      "clang LLVM compiler", OverlayFS);
  Driver->setTitle("clang_based_tool");
  Compilation.reset(Driver->BuildCompilation(llvm::ArrayRef(CCommandLine)));

  if (Compilation->containsError()) {
    return llvm::make_error<llvm::StringError>("Failed to build compilation",
                                               llvm::inconvertibleErrorCode());
  }

  const driver::Command &Command = *(Compilation->getJobs().begin());
  const auto &CommandArgs = Command.getArguments();
  size_t ArgSize = CommandArgs.size();
  assert(ArgSize >= 1 && "Cannot have a command with 0 args");
  const char *FirstArg = CommandArgs[0];
  if (strcmp(FirstArg, "-cc1"))
    return llvm::make_error<llvm::StringError>(
        "Incorrect compilation command, missing cc1",
        llvm::inconvertibleErrorCode());
  Invocation = std::make_unique<CompilerInvocation>();
  CompilerInvocation::CreateFromArgs(*Invocation, Command.getArguments(),
                                     *Diags, Command.getExecutable());
  Invocation->getFrontendOpts().DisableFree = false;
  Invocation->getCodeGenOpts().DisableFree = false;

  if (any(Worker.Service.getOptimizeArgs() & ScanningOptimizations::Macros))
    canonicalizeDefines(Invocation->getPreprocessorOpts());

  // Create the CompilerInstance.
  ModCache = makeInProcessModuleCache(Worker.Service.getModuleCacheEntries());
  CIPtr = std::make_unique<CompilerInstance>(
      std::make_shared<CompilerInvocation>(*Invocation), Worker.PCHContainerOps,
      ModCache.get());
  auto &CI = *CIPtr;

  // TODO: the commented out code here should be un-commented when
  // we enable CAS.
  // CI.getInvocation().getCASOpts() = Worker.CASOpts;
  CI.setBuildingModule(false);
  CI.createVirtualFileSystem(OverlayFS, Diags->getClient());
  sanitizeDiagOpts(CI.getDiagnosticOpts());
  CI.createDiagnostics(DiagPrinter.get(), false);
  CI.getPreprocessorOpts().AllowPCHWithDifferentModulesCachePath = true;
  CI.getFrontendOpts().GenerateGlobalModuleIndex = false;
  CI.getFrontendOpts().UseGlobalModuleIndex = false;
  // CI.getFrontendOpts().ModulesShareFileManager = Worker.DepCASFS ? false :
  // true;
  CI.getHeaderSearchOpts().ModuleFormat = "raw";
  CI.getHeaderSearchOpts().ModulesIncludeVFSUsage =
      any(Worker.Service.getOptimizeArgs() & ScanningOptimizations::VFS);
  CI.getHeaderSearchOpts().ModulesStrictContextHash = true;
  CI.getHeaderSearchOpts().ModulesSerializeOnlyPreprocessor = true;
  CI.getHeaderSearchOpts().ModulesSkipDiagnosticOptions = true;
  CI.getHeaderSearchOpts().ModulesSkipHeaderSearchPaths = true;
  CI.getHeaderSearchOpts().ModulesSkipPragmaDiagnosticMappings = true;
  CI.getPreprocessorOpts().ModulesCheckRelocated = false;

  if (CI.getHeaderSearchOpts().ModulesValidateOncePerBuildSession)
    CI.getHeaderSearchOpts().BuildSessionTimestamp =
        Worker.Service.getBuildSessionTimestamp();

  CI.setDiagnostics(Diags.get());

  auto *FileMgr = CI.createFileManager();

  if (Worker.DepFS) {
    Worker.DepFS->resetBypassedPathPrefix();
    if (!CI.getHeaderSearchOpts().ModuleCachePath.empty()) {
      SmallString<256> ModulesCachePath;
      normalizeModuleCachePath(
          *FileMgr, CI.getHeaderSearchOpts().ModuleCachePath, ModulesCachePath);
      Worker.DepFS->setBypassedPathPrefix(ModulesCachePath);
    }

    CI.setDependencyDirectivesGetter(
        std::make_unique<ScanningDependencyDirectivesGetter>(*FileMgr));
  }

  CI.createSourceManager(*FileMgr);

  const StringRef Sysroot = CI.getHeaderSearchOpts().Sysroot;
  if (!Sysroot.empty() && (llvm::sys::path::root_directory(Sysroot) != Sysroot))
    StableDirs = {Sysroot, CI.getHeaderSearchOpts().ResourceDir};
  if (!CI.getPreprocessorOpts().ImplicitPCHInclude.empty())
    if (visitPrebuiltModule(CI.getPreprocessorOpts().ImplicitPCHInclude, CI,
                            CI.getHeaderSearchOpts().PrebuiltModuleFiles,
                            PrebuiltModuleVFSMap, CI.getDiagnostics(),
                            StableDirs))
      return llvm::make_error<llvm::StringError>(
          "Prebuilt module scanning failed", llvm::inconvertibleErrorCode());

  OutputOpts = std::make_unique<DependencyOutputOptions>();
  std::swap(*OutputOpts, CI.getInvocation().getDependencyOutputOpts());
  // We need at least one -MT equivalent for the generator of make dependency
  // files to work.
  if (OutputOpts->Targets.empty())
    OutputOpts->Targets = {deduceDepTarget(CI.getFrontendOpts().OutputFile,
                                           CI.getFrontendOpts().Inputs)};
  OutputOpts->IncludeSystemHeaders = true;

  CI.createTarget();
  // CI.initializeDelayedInputFileFromCAS();

  return llvm::Error::success();
}

llvm::Error CompilerInstanceWithContext::computeDependencies(
    StringRef ModuleName, DependencyConsumer &Consumer,
    DependencyActionController &Controller) {
  auto &CI = *CIPtr;
  CompilerInvocation Inv(*Invocation);

  auto Opts = std::make_unique<DependencyOutputOptions>(*OutputOpts);
  auto MDC = std::make_shared<ModuleDepCollector>(
      Worker.Service, std::move(Opts), CI, Consumer, Controller, Inv,
      PrebuiltModuleVFSMap, StableDirs);

  CI.clearDependencyCollectors();
  CI.addDependencyCollector(MDC);

  std::unique_ptr<FrontendAction> Action =
      std::make_unique<GetDependenciesByModuleNameAction>(ModuleName);
  auto InputFile = CI.getFrontendOpts().Inputs.begin();

  if (!SrcLocOffset)
    Action->BeginSourceFile(CI, *InputFile);
  else {
    CI.getPreprocessor().removePPCallbacks();
  }

  Preprocessor &PP = CI.getPreprocessor();
  SourceManager &SM = PP.getSourceManager();
  FileID MainFileID = SM.getMainFileID();
  SourceLocation FileStart = SM.getLocForStartOfFile(MainFileID);
  SourceLocation IDLocation = FileStart.getLocWithOffset(SrcLocOffset);
  if (!SrcLocOffset)
    PP.EnterSourceFile(MainFileID, nullptr, SourceLocation());
  else {
    auto DCs = CI.getDependencyCollectors();
    for (auto &DC : DCs) {
      DC->attachToPreprocessor(PP);
      auto *CB = DC->getPPCallback();

      FileID PrevFID;
      SrcMgr::CharacteristicKind FileType =
          SM.getFileCharacteristic(IDLocation);
      CB->LexedFileChanged(MainFileID,
                           PPChainedCallbacks::LexedFileChangeReason::EnterFile,
                           FileType, PrevFID, IDLocation);
    }
  }

  SrcLocOffset++;
  SmallVector<IdentifierLoc, 2> Path;
  IdentifierInfo *ModuleID = PP.getIdentifierInfo(ModuleName);
  Path.emplace_back(IDLocation, ModuleID);
  auto ModResult = CI.loadModule(IDLocation, Path, Module::Hidden, false);

  auto DCs = CI.getDependencyCollectors();
  for (auto &DC : DCs) {
    auto *CB = DC->getPPCallback();
    assert(CB && "DC must have dependency collector callback");
    CB->moduleImport(SourceLocation(), Path, ModResult);
    CB->EndOfMainFile();
  }

  MDC->applyDiscoveredDependencies(Inv);

  // TODO: enable CAS
  //   std::string ID = Inv.getFileSystemOpts().CASFileSystemRootID;
  //   if (!ID.empty())
  //     Consumer.handleCASFileSystemRootID(std::move(ID));
  //   ID = Inv.getFrontendOpts().CASIncludeTreeID;
  //   if (!ID.empty())
  //     Consumer.handleIncludeTreeID(std::move(ID));

  return llvm::Error::success();
}

llvm::Error CompilerInstanceWithContext::finalize() {
  DiagPrinter->finish();
  return llvm::Error::success();
}