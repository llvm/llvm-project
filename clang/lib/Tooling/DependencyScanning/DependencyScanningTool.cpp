//===- DependencyScanningTool.cpp - clang-scan-deps service ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/ObjectStore.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

static std::vector<std::string>
makeTUCommandLineWithoutPaths(ArrayRef<std::string> OriginalCommandLine) {
  std::vector<std::string> Args = OriginalCommandLine;

  Args.push_back("-fno-implicit-modules");
  Args.push_back("-fno-implicit-module-maps");

  // These arguments are unused in explicit compiles.
  llvm::erase_if(Args, [](StringRef Arg) {
    if (Arg.consume_front("-fmodules-")) {
      return Arg.startswith("cache-path=") ||
             Arg.startswith("prune-interval=") ||
             Arg.startswith("prune-after=") ||
             Arg == "validate-once-per-build-session";
    }
    return Arg.startswith("-fbuild-session-file=");
  });

  return Args;
}

DependencyScanningTool::DependencyScanningTool(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : Worker(Service, std::move(FS)) {}

llvm::Expected<std::string> DependencyScanningTool::getDependencyFile(
    const std::vector<std::string> &CommandLine, StringRef CWD,
    llvm::Optional<StringRef> ModuleName) {
  /// Prints out all of the gathered dependencies into a string.
  class MakeDependencyPrinterConsumer : public DependencyConsumer {
  public:
    void handleBuildCommand(Command) override {}

    void
    handleDependencyOutputOpts(const DependencyOutputOptions &Opts) override {
      this->Opts = std::make_unique<DependencyOutputOptions>(Opts);
    }

    void handleFileDependency(StringRef File) override {
      Dependencies.push_back(std::string(File));
    }

    void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {
      // Same as `handleModuleDependency`.
    }

    void handleModuleDependency(ModuleDeps MD) override {
      // These are ignored for the make format as it can't support the full
      // set of deps, and handleFileDependency handles enough for implicitly
      // built modules to work.
    }

    void handleContextHash(std::string Hash) override {}
    void handleCASFileSystemRootID(cas::CASID) override {}

    std::string lookupModuleOutput(const ModuleID &ID,
                                   ModuleOutputKind Kind) override {
      llvm::report_fatal_error("unexpected call to lookupModuleOutput");
    }

    void printDependencies(std::string &S) {
      assert(Opts && "Handled dependency output options.");

      class DependencyPrinter : public DependencyFileGenerator {
      public:
        DependencyPrinter(DependencyOutputOptions &Opts,
                          ArrayRef<std::string> Dependencies)
            : DependencyFileGenerator(Opts) {
          for (const auto &Dep : Dependencies)
            addDependency(Dep);
        }

        void printDependencies(std::string &S) {
          llvm::raw_string_ostream OS(S);
          outputDependencyFile(OS);
        }
      };

      DependencyPrinter Generator(*Opts, Dependencies);
      Generator.printDependencies(S);
    }

  private:
    std::unique_ptr<DependencyOutputOptions> Opts;
    std::vector<std::string> Dependencies;
  };

  MakeDependencyPrinterConsumer Consumer;
  auto Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, ModuleName);
  if (Result)
    return std::move(Result);
  std::string Output;
  Consumer.printDependencies(Output);
  return Output;
}

namespace {
/// Returns a CAS tree containing the dependencies.
class GetDependencyTree : public DependencyConsumer {
public:
  void handleBuildCommand(Command) override {}
  void handleFileDependency(StringRef File) override {}
  void handleModuleDependency(ModuleDeps) override {}
  void handlePrebuiltModuleDependency(PrebuiltModuleDep) override {}
  void handleDependencyOutputOpts(const DependencyOutputOptions &) override {}
  void handleContextHash(std::string) override {}

  void handleCASFileSystemRootID(cas::CASID ID) override {
    CASFileSystemRootID = ID;
  }

  std::string lookupModuleOutput(const ModuleID &, ModuleOutputKind) override {
    llvm::report_fatal_error("unexpected call to lookupModuleOutput");
  }

  Expected<llvm::cas::ObjectProxy> getTree() {
    if (CASFileSystemRootID)
      return CAS.getProxy(*CASFileSystemRootID);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to get casfs");
  }

  GetDependencyTree(cas::ObjectStore &CAS) : CAS(CAS) {}

private:
  cas::ObjectStore &CAS;
  Optional<cas::CASID> CASFileSystemRootID;
};
}

llvm::Expected<llvm::cas::ObjectProxy>
DependencyScanningTool::getDependencyTree(
    const std::vector<std::string> &CommandLine, StringRef CWD) {
  GetDependencyTree Consumer(Worker.getCASFS().getCAS());
  auto Result = Worker.computeDependencies(CWD, CommandLine, Consumer);
  if (Result)
    return std::move(Result);
  return Consumer.getTree();
}

llvm::Expected<llvm::cas::ObjectProxy>
DependencyScanningTool::getDependencyTreeFromCompilerInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, StringRef CWD,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    bool DiagGenerationAsCompilation,
    llvm::function_ref<StringRef(const llvm::vfs::CachedDirectoryEntry &)>
        RemapPath) {
  GetDependencyTree Consumer(Worker.getCASFS().getCAS());
  Worker.computeDependenciesFromCompilerInvocation(
      std::move(Invocation), CWD, Consumer, RemapPath, DiagsConsumer, VerboseOS,
      DiagGenerationAsCompilation);
  return Consumer.getTree();
}

namespace {
class IncludeTreePPConsumer : public PPIncludeActionsConsumer {
public:
  explicit IncludeTreePPConsumer(cas::ObjectStore &DB) : DB(DB) {}

  Expected<cas::IncludeTreeRoot> getIncludeTree();

private:
  void enteredInclude(Preprocessor &PP, FileID FID) override;

  void exitedInclude(Preprocessor &PP, FileID IncludedBy, FileID Include,
                     SourceLocation ExitLoc) override;

  void handleHasIncludeCheck(Preprocessor &PP, bool Result) override;

  void finalize(CompilerInstance &CI) override;

  Expected<cas::ObjectRef> getObjectForFile(Preprocessor &PP, FileID FID);
  Expected<cas::ObjectRef>
  getObjectForFileNonCached(FileManager &FM, const SrcMgr::FileInfo &FI);
  Expected<cas::ObjectRef> getObjectForBuffer(const SrcMgr::FileInfo &FI);
  Expected<cas::ObjectRef> addToFileList(FileManager &FM, const FileEntry *FE);

  struct FilePPState {
    SrcMgr::CharacteristicKind FileCharacteristic;
    cas::ObjectRef File;
    SmallVector<std::pair<cas::ObjectRef, uint32_t>, 6> Includes;
    llvm::SmallBitVector HasIncludeChecks;
  };

  Expected<cas::IncludeTree> getCASTreeForFileIncludes(FilePPState &&PPState);

  bool hasErrorOccurred() const { return ErrorToReport.has_value(); }

  template <typename T> Optional<T> check(Expected<T> &&E) {
    if (!E) {
      ErrorToReport = E.takeError();
      return std::nullopt;
    }
    return *E;
  }

  cas::ObjectStore &DB;
  Optional<cas::ObjectRef> PCHRef;
  bool StartedEnteringIncludes = false;
  // When a PCH is used this lists the filenames of the included files as they
  // are recorded in the PCH, ordered by \p FileEntry::UID index.
  SmallVector<StringRef> PreIncludedFileNames;
  llvm::BitVector SeenIncludeFiles;
  SmallVector<cas::IncludeFileList::FileEntry> IncludedFiles;
  Optional<cas::ObjectRef> PredefinesBufferRef;
  SmallVector<FilePPState> IncludeStack;
  llvm::DenseMap<const FileEntry *, Optional<cas::ObjectRef>> ObjectForFile;
  Optional<llvm::Error> ErrorToReport;
};
} // namespace

void IncludeTreePPConsumer::enteredInclude(Preprocessor &PP, FileID FID) {
  if (hasErrorOccurred())
    return;

  if (!StartedEnteringIncludes) {
    StartedEnteringIncludes = true;

    // Get the included files (coming from a PCH), and keep track of the
    // filenames that were recorded in the PCH.
    for (const FileEntry *FE : PP.getIncludedFiles()) {
      unsigned UID = FE->getUID();
      if (UID >= PreIncludedFileNames.size())
        PreIncludedFileNames.resize(UID + 1);
      PreIncludedFileNames[UID] = FE->getName();
    }
  }

  Optional<cas::ObjectRef> FileRef = check(getObjectForFile(PP, FID));
  if (!FileRef)
    return;
  const SrcMgr::FileInfo &FI =
      PP.getSourceManager().getSLocEntry(FID).getFile();
  IncludeStack.push_back({FI.getFileCharacteristic(), *FileRef, {}, {}});
}

void IncludeTreePPConsumer::exitedInclude(Preprocessor &PP, FileID IncludedBy,
                                          FileID Include,
                                          SourceLocation ExitLoc) {
  if (hasErrorOccurred())
    return;

  assert(*check(getObjectForFile(PP, Include)) == IncludeStack.back().File);
  Optional<cas::IncludeTree> IncludeTree =
      check(getCASTreeForFileIncludes(IncludeStack.pop_back_val()));
  if (!IncludeTree)
    return;
  assert(*check(getObjectForFile(PP, IncludedBy)) == IncludeStack.back().File);
  SourceManager &SM = PP.getSourceManager();
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedExpansionLoc(ExitLoc);
  IncludeStack.back().Includes.push_back(
      {IncludeTree->getRef(), LocInfo.second});
}

void IncludeTreePPConsumer::handleHasIncludeCheck(Preprocessor &PP,
                                                  bool Result) {
  if (hasErrorOccurred())
    return;

  IncludeStack.back().HasIncludeChecks.push_back(Result);
}

void IncludeTreePPConsumer::finalize(CompilerInstance &CI) {
  FileManager &FM = CI.getFileManager();

  auto addFile = [&](StringRef FilePath, bool IgnoreFileError = false) -> bool {
    llvm::ErrorOr<const FileEntry *> FE = FM.getFile(FilePath);
    if (!FE) {
      if (IgnoreFileError)
        return true;
      ErrorToReport = llvm::errorCodeToError(FE.getError());
      return false;
    }
    Expected<cas::ObjectRef> Ref = addToFileList(FM, *FE);
    if (!Ref) {
      ErrorToReport = Ref.takeError();
      return false;
    }
    return true;
  };

  for (StringRef FilePath : CI.getLangOpts().NoSanitizeFiles) {
    if (!addFile(FilePath))
      return;
  }
  // Add profile files.
  // FIXME: Do not have the logic here to determine which path should be set
  // but ideally only the path needed for the compilation is set and we already
  // checked the file needed exists. Just try load and ignore errors.
  addFile(CI.getCodeGenOpts().ProfileInstrumentUsePath,
          /*IgnoreFileError=*/true);
  addFile(CI.getCodeGenOpts().SampleProfileFile, /*IgnoreFileError=*/true);
  addFile(CI.getCodeGenOpts().ProfileRemappingFile, /*IgnoreFileError=*/true);

  StringRef Sysroot = CI.getHeaderSearchOpts().Sysroot;
  if (!Sysroot.empty()) {
    // Include 'SDKSettings.json', if it exists, to accomodate availability
    // checks during the compilation.
    llvm::SmallString<256> FilePath = Sysroot;
    llvm::sys::path::append(FilePath, "SDKSettings.json");
    addFile(FilePath, /*IgnoreFileError*/ true);
  }

  PreprocessorOptions &PPOpts = CI.getPreprocessorOpts();
  if (PPOpts.ImplicitPCHInclude.empty())
    return; // no need for additional work.

  // Go through all the recorded included files; we'll get additional files from
  // the PCH that we need to include in the file list, in case they are
  // referenced while replaying the include-tree.
  SmallVector<const FileEntry *, 32> NotSeenIncludes;
  for (const FileEntry *FE : CI.getPreprocessor().getIncludedFiles()) {
    if (FE->getUID() >= SeenIncludeFiles.size() ||
        !SeenIncludeFiles[FE->getUID()])
      NotSeenIncludes.push_back(FE);
  }
  // Sort so we can visit the files in deterministic order.
  llvm::sort(NotSeenIncludes, [](const FileEntry *LHS, const FileEntry *RHS) {
    return LHS->getUID() < RHS->getUID();
  });

  for (const FileEntry *FE : NotSeenIncludes) {
    auto FileNode = addToFileList(FM, FE);
    if (!FileNode) {
      ErrorToReport = FileNode.takeError();
      return;
    }
  }

  llvm::ErrorOr<Optional<cas::ObjectRef>> CASContents =
      FM.getObjectRefForFileContent(PPOpts.ImplicitPCHInclude);
  if (!CASContents) {
    ErrorToReport = llvm::errorCodeToError(CASContents.getError());
    return;
  }
  PCHRef = **CASContents;
}

Expected<cas::ObjectRef>
IncludeTreePPConsumer::getObjectForFile(Preprocessor &PP, FileID FID) {
  SourceManager &SM = PP.getSourceManager();
  const SrcMgr::FileInfo &FI = SM.getSLocEntry(FID).getFile();
  if (PP.getPredefinesFileID() == FID) {
    if (!PredefinesBufferRef) {
      auto Ref = getObjectForBuffer(FI);
      if (!Ref)
        return Ref.takeError();
      PredefinesBufferRef = *Ref;
    }
    return *PredefinesBufferRef;
  }
  assert(FI.getContentCache().OrigEntry);
  auto &FileRef = ObjectForFile[FI.getContentCache().OrigEntry];
  if (!FileRef) {
    auto Ref = getObjectForFileNonCached(SM.getFileManager(), FI);
    if (!Ref)
      return Ref.takeError();
    FileRef = *Ref;
  }
  return *FileRef;
}

Expected<cas::ObjectRef>
IncludeTreePPConsumer::getObjectForFileNonCached(FileManager &FM,
                                                 const SrcMgr::FileInfo &FI) {
  const FileEntry *FE = FI.getContentCache().OrigEntry;
  assert(FE);

  // Mark the include as already seen.
  if (FE->getUID() >= SeenIncludeFiles.size())
    SeenIncludeFiles.resize(FE->getUID() + 1);
  SeenIncludeFiles.set(FE->getUID());

  return addToFileList(FM, FE);
}

Expected<cas::ObjectRef>
IncludeTreePPConsumer::getObjectForBuffer(const SrcMgr::FileInfo &FI) {
  // This is a non-file buffer, like the predefines.
  auto Ref = DB.storeFromString(
      {}, FI.getContentCache().getBufferIfLoaded()->getBuffer());
  if (!Ref)
    return Ref.takeError();
  Expected<cas::IncludeFile> FileNode =
      cas::IncludeFile::create(DB, FI.getName(), *Ref);
  if (!FileNode)
    return FileNode.takeError();
  return FileNode->getRef();
}

Expected<cas::ObjectRef>
IncludeTreePPConsumer::addToFileList(FileManager &FM, const FileEntry *FE) {
  StringRef Filename = FE->getName();
  llvm::ErrorOr<Optional<cas::ObjectRef>> CASContents =
      FM.getObjectRefForFileContent(Filename);
  if (!CASContents)
    return llvm::errorCodeToError(CASContents.getError());
  assert(*CASContents);

  auto addFile = [&](StringRef Filename) -> Expected<cas::ObjectRef> {
    assert(!Filename.empty());
    auto FileNode = cas::IncludeFile::create(DB, Filename, **CASContents);
    if (!FileNode)
      return FileNode.takeError();
    IncludedFiles.push_back(
        {FileNode->getRef(),
         static_cast<cas::IncludeFileList::FileSizeTy>(FE->getSize())});
    return FileNode->getRef();
  };

  // Check whether another path coming from the PCH is associated with the same
  // file.
  unsigned UID = FE->getUID();
  if (UID < PreIncludedFileNames.size() && !PreIncludedFileNames[UID].empty() &&
      PreIncludedFileNames[UID] != Filename) {
    auto FileNode = addFile(PreIncludedFileNames[UID]);
    if (!FileNode)
      return FileNode.takeError();
  }

  return addFile(Filename);
}

Expected<cas::IncludeTree>
IncludeTreePPConsumer::getCASTreeForFileIncludes(FilePPState &&PPState) {
  return cas::IncludeTree::create(DB, PPState.FileCharacteristic, PPState.File,
                                  PPState.Includes, PPState.HasIncludeChecks);
}

Expected<cas::IncludeTreeRoot> IncludeTreePPConsumer::getIncludeTree() {
  if (ErrorToReport)
    return std::move(*ErrorToReport);

  assert(IncludeStack.size() == 1);
  Expected<cas::IncludeTree> MainIncludeTree =
      getCASTreeForFileIncludes(IncludeStack.pop_back_val());
  if (!MainIncludeTree)
    return MainIncludeTree.takeError();
  auto FileList = cas::IncludeFileList::create(DB, IncludedFiles);
  if (!FileList)
    return FileList.takeError();

  return cas::IncludeTreeRoot::create(DB, MainIncludeTree->getRef(),
                                      FileList->getRef(), PCHRef);
}

Expected<cas::IncludeTreeRoot> DependencyScanningTool::getIncludeTree(
    cas::ObjectStore &DB, const std::vector<std::string> &CommandLine,
    StringRef CWD) {
  IncludeTreePPConsumer Consumer(DB);
  llvm::Error Result = Worker.computeDependencies(CWD, CommandLine, Consumer);
  if (Result)
    return std::move(Result);
  return Consumer.getIncludeTree();
}

Expected<cas::IncludeTreeRoot>
DependencyScanningTool::getIncludeTreeFromCompilerInvocation(
    cas::ObjectStore &DB, std::shared_ptr<CompilerInvocation> Invocation,
    StringRef CWD, DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    bool DiagGenerationAsCompilation) {
  IncludeTreePPConsumer Consumer(DB);
  Worker.computeDependenciesFromCompilerInvocation(
      std::move(Invocation), CWD, Consumer, /*RemapPath=*/nullptr,
      DiagsConsumer, VerboseOS, DiagGenerationAsCompilation);
  return Consumer.getIncludeTree();
}

llvm::Expected<FullDependenciesResult>
DependencyScanningTool::getFullDependencies(
    const std::vector<std::string> &CommandLine, StringRef CWD,
    const llvm::StringSet<> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput,
    llvm::Optional<StringRef> ModuleName) {
  FullDependencyConsumer Consumer(AlreadySeen, LookupModuleOutput,
                                  Worker.shouldEagerLoadModules());
  llvm::Error Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, ModuleName);
  if (Result)
    return std::move(Result);
  return Consumer.takeFullDependencies();
}

llvm::Expected<FullDependenciesResult>
DependencyScanningTool::getFullDependenciesLegacyDriverCommand(
    const std::vector<std::string> &CommandLine, StringRef CWD,
    const llvm::StringSet<> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput,
    llvm::Optional<StringRef> ModuleName) {
  FullDependencyConsumer Consumer(AlreadySeen, LookupModuleOutput,
                                  Worker.shouldEagerLoadModules());
  llvm::Error Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, ModuleName);
  if (Result)
    return std::move(Result);
  return Consumer.getFullDependenciesLegacyDriverCommand(CommandLine);
}

FullDependenciesResult FullDependencyConsumer::takeFullDependencies() {
  FullDependenciesResult FDR;
  FullDependencies &FD = FDR.FullDeps;

  FD.ID.ContextHash = std::move(ContextHash);
  FD.FileDeps = std::move(Dependencies);
  FD.PrebuiltModuleDeps = std::move(PrebuiltModuleDeps);
  FD.Commands = std::move(Commands);
  FD.CASFileSystemRootID = CASFileSystemRootID;

  for (auto &&M : ClangModuleDeps) {
    auto &MD = M.second;
    if (MD.ImportedByMainFile)
      FD.ClangModuleDeps.push_back(MD.ID);
    // TODO: Avoid handleModuleDependency even being called for modules
    //   we've already seen.
    if (AlreadySeen.count(M.first))
      continue;
    FDR.DiscoveredModules.push_back(std::move(MD));
  }

  return FDR;
}

FullDependenciesResult
FullDependencyConsumer::getFullDependenciesLegacyDriverCommand(
    const std::vector<std::string> &OriginalCommandLine) const {
  FullDependencies FD;

  FD.DriverCommandLine = makeTUCommandLineWithoutPaths(
      ArrayRef<std::string>(OriginalCommandLine).slice(1));

  FD.ID.ContextHash = std::move(ContextHash);

  FD.FileDeps.assign(Dependencies.begin(), Dependencies.end());

  for (const PrebuiltModuleDep &PMD : PrebuiltModuleDeps)
    FD.DriverCommandLine.push_back("-fmodule-file=" + PMD.PCMFile);

  for (auto &&M : ClangModuleDeps) {
    auto &MD = M.second;
    if (MD.ImportedByMainFile) {
      FD.ClangModuleDeps.push_back(MD.ID);
      auto PCMPath = LookupModuleOutput(MD.ID, ModuleOutputKind::ModuleFile);
      if (EagerLoadModules) {
        FD.DriverCommandLine.push_back("-fmodule-file=" + PCMPath);
      } else {
        FD.DriverCommandLine.push_back("-fmodule-map-file=" +
                                       MD.ClangModuleMapFile);
        FD.DriverCommandLine.push_back("-fmodule-file=" + MD.ID.ModuleName +
                                       "=" + PCMPath);
      }
    }
  }

  FD.PrebuiltModuleDeps = std::move(PrebuiltModuleDeps);

  FD.CASFileSystemRootID = CASFileSystemRootID;

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
