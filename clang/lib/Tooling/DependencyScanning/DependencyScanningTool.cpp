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
#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/PrefixMapper.h"
#include "llvm/Support/PrefixMappingFileSystem.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;
using llvm::Error;

DependencyScanningTool::DependencyScanningTool(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : Worker(Service, std::move(FS)) {}

namespace {
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

protected:
  std::unique_ptr<DependencyOutputOptions> Opts;
  std::vector<std::string> Dependencies;
};
} // anonymous namespace

llvm::Expected<std::string> DependencyScanningTool::getDependencyFile(
    const std::vector<std::string> &CommandLine, StringRef CWD) {
  MakeDependencyPrinterConsumer Consumer;
  CallbackActionController Controller(nullptr);
  auto Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, Controller);
  if (Result)
    return std::move(Result);
  std::string Output;
  Consumer.printDependencies(Output);
  return Output;
}

namespace {
class EmptyDependencyConsumer : public DependencyConsumer {
  void
  handleDependencyOutputOpts(const DependencyOutputOptions &Opts) override {}

  void handleFileDependency(StringRef Filename) override {}

  void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {}

  void handleModuleDependency(ModuleDeps MD) override {}

  void handleContextHash(std::string Hash) override {}
};

/// Returns a CAS tree containing the dependencies.
class GetDependencyTree : public EmptyDependencyConsumer {
public:
  void handleCASFileSystemRootID(std::string ID) override {
    CASFileSystemRootID = ID;
  }

  Expected<llvm::cas::ObjectProxy> getTree() {
    if (CASFileSystemRootID) {
      auto ID = FS.getCAS().parseID(*CASFileSystemRootID);
      if (!ID)
        return ID.takeError();
      return FS.getCAS().getProxy(*ID);
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to get casfs");
  }

  GetDependencyTree(llvm::cas::CachingOnDiskFileSystem &FS) : FS(FS) {}

private:
  llvm::cas::CachingOnDiskFileSystem &FS;
  std::optional<std::string> CASFileSystemRootID;
};
}

llvm::Expected<llvm::cas::ObjectProxy>
DependencyScanningTool::getDependencyTree(
    const std::vector<std::string> &CommandLine, StringRef CWD,
    DepscanPrefixMapping PrefixMapping) {
  GetDependencyTree Consumer(*Worker.getCASFS());
  CASFSActionController Controller(nullptr, *Worker.getCASFS(),
                                   std::move(PrefixMapping));
  auto Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, Controller);
  if (Result)
    return std::move(Result);
  return Consumer.getTree();
}

llvm::Expected<llvm::cas::ObjectProxy>
DependencyScanningTool::getDependencyTreeFromCompilerInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, StringRef CWD,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    bool DiagGenerationAsCompilation, DepscanPrefixMapping PrefixMapping) {
  GetDependencyTree Consumer(*Worker.getCASFS());
  CASFSActionController Controller(nullptr, *Worker.getCASFS(),
                                   std::move(PrefixMapping));
  Worker.computeDependenciesFromCompilerInvocation(
      std::move(Invocation), CWD, Consumer, Controller, DiagsConsumer,
      VerboseOS, DiagGenerationAsCompilation);
  return Consumer.getTree();
}

namespace {
class IncludeTreeActionController : public CallbackActionController {
public:
  IncludeTreeActionController(cas::ObjectStore &DB,
                              const DepscanPrefixMapping &PrefixMapping)
      : CallbackActionController(nullptr), DB(DB),
        PrefixMapping(PrefixMapping) {}

  Expected<cas::IncludeTreeRoot> getIncludeTree();

private:
  Error initialize(CompilerInstance &ScanInstance,
                   CompilerInvocation &NewInvocation) override;

  void enteredInclude(Preprocessor &PP, FileID FID) override;

  void exitedInclude(Preprocessor &PP, FileID IncludedBy, FileID Include,
                     SourceLocation ExitLoc) override;

  void handleHasIncludeCheck(Preprocessor &PP, bool Result) override;

  const DepscanPrefixMapping *getPrefixMapping() override {
    return &PrefixMapping;
  }

  Error finalize(CompilerInstance &ScanInstance,
                 CompilerInvocation &NewInvocation) override;

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

  Expected<cas::IncludeFile> createIncludeFile(StringRef Filename,
                                               cas::ObjectRef Contents);

  bool hasErrorOccurred() const { return ErrorToReport.has_value(); }

  template <typename T> std::optional<T> check(Expected<T> &&E) {
    if (!E) {
      ErrorToReport = E.takeError();
      return std::nullopt;
    }
    return *E;
  }

  cas::ObjectStore &DB;
  const DepscanPrefixMapping &PrefixMapping;
  llvm::PrefixMapper PrefixMapper;
  std::optional<cas::ObjectRef> PCHRef;
  bool StartedEnteringIncludes = false;
  // When a PCH is used this lists the filenames of the included files as they
  // are recorded in the PCH, ordered by \p FileEntry::UID index.
  SmallVector<StringRef> PreIncludedFileNames;
  llvm::BitVector SeenIncludeFiles;
  SmallVector<cas::IncludeFileList::FileEntry> IncludedFiles;
  std::optional<cas::ObjectRef> PredefinesBufferRef;
  SmallVector<FilePPState> IncludeStack;
  llvm::DenseMap<const FileEntry *, std::optional<cas::ObjectRef>>
      ObjectForFile;
  std::optional<llvm::Error> ErrorToReport;
};
} // namespace

/// The PCH recorded file paths with canonical paths, create a VFS that
/// allows remapping back to the non-canonical source paths so that they are
/// found during dep-scanning.
static void
addReversePrefixMappingFileSystem(const llvm::PrefixMapper &PrefixMapper,
                                  CompilerInstance &ScanInstance) {
  llvm::PrefixMapper ReverseMapper;
  ReverseMapper.addInverseRange(PrefixMapper.getMappings());
  ReverseMapper.sort();
  std::unique_ptr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPrefixMappingFileSystem(
          std::move(ReverseMapper), &ScanInstance.getVirtualFileSystem());

  ScanInstance.getFileManager().setVirtualFileSystem(std::move(FS));
}

Error IncludeTreeActionController::initialize(
    CompilerInstance &ScanInstance, CompilerInvocation &NewInvocation) {
  if (Error E =
          PrefixMapping.configurePrefixMapper(NewInvocation, PrefixMapper))
    return E;

  auto ensurePathRemapping = [&]() {
    if (PrefixMapper.empty())
      return;

    PreprocessorOptions &PPOpts = ScanInstance.getPreprocessorOpts();
    if (PPOpts.Includes.empty() && PPOpts.ImplicitPCHInclude.empty())
      return;

    addReversePrefixMappingFileSystem(PrefixMapper, ScanInstance);

    // These are written in the predefines buffer, so we need to remap them.
    for (std::string &Include : PPOpts.Includes)
      PrefixMapper.mapInPlace(Include);
  };
  ensurePathRemapping();

  return Error::success();
}

void IncludeTreeActionController::enteredInclude(Preprocessor &PP, FileID FID) {
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

  std::optional<cas::ObjectRef> FileRef = check(getObjectForFile(PP, FID));
  if (!FileRef)
    return;
  const SrcMgr::FileInfo &FI =
      PP.getSourceManager().getSLocEntry(FID).getFile();
  IncludeStack.push_back({FI.getFileCharacteristic(), *FileRef, {}, {}});
}

void IncludeTreeActionController::exitedInclude(Preprocessor &PP,
                                                FileID IncludedBy,
                                                FileID Include,
                                                SourceLocation ExitLoc) {
  if (hasErrorOccurred())
    return;

  assert(*check(getObjectForFile(PP, Include)) == IncludeStack.back().File);
  std::optional<cas::IncludeTree> IncludeTree =
      check(getCASTreeForFileIncludes(IncludeStack.pop_back_val()));
  if (!IncludeTree)
    return;
  assert(*check(getObjectForFile(PP, IncludedBy)) == IncludeStack.back().File);
  SourceManager &SM = PP.getSourceManager();
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedExpansionLoc(ExitLoc);
  IncludeStack.back().Includes.push_back(
      {IncludeTree->getRef(), LocInfo.second});
}

void IncludeTreeActionController::handleHasIncludeCheck(Preprocessor &PP,
                                                        bool Result) {
  if (hasErrorOccurred())
    return;

  IncludeStack.back().HasIncludeChecks.push_back(Result);
}

Error IncludeTreeActionController::finalize(CompilerInstance &ScanInstance,
                                            CompilerInvocation &NewInvocation) {
  FileManager &FM = ScanInstance.getFileManager();

  auto addFile = [&](StringRef FilePath,
                     bool IgnoreFileError = false) -> Error {
    llvm::ErrorOr<const FileEntry *> FE = FM.getFile(FilePath);
    if (!FE) {
      if (IgnoreFileError)
        return Error::success();
      return llvm::errorCodeToError(FE.getError());
    }
    std::optional<cas::ObjectRef> Ref;
    return addToFileList(FM, *FE).moveInto(Ref);
  };

  for (StringRef FilePath : NewInvocation.getLangOpts()->NoSanitizeFiles) {
    if (Error E = addFile(FilePath))
      return E;
  }
  // Add profile files.
  // FIXME: Do not have the logic here to determine which path should be set
  // but ideally only the path needed for the compilation is set and we already
  // checked the file needed exists. Just try load and ignore errors.
  if (Error E = addFile(NewInvocation.getCodeGenOpts().ProfileInstrumentUsePath,
                        /*IgnoreFileError=*/true))
    return E;
  if (Error E = addFile(NewInvocation.getCodeGenOpts().SampleProfileFile,
                        /*IgnoreFileError=*/true))
    return E;
  if (Error E = addFile(NewInvocation.getCodeGenOpts().ProfileRemappingFile,
                        /*IgnoreFileError=*/true))
    return E;

  StringRef Sysroot = NewInvocation.getHeaderSearchOpts().Sysroot;
  if (!Sysroot.empty()) {
    // Include 'SDKSettings.json', if it exists, to accomodate availability
    // checks during the compilation.
    llvm::SmallString<256> FilePath = Sysroot;
    llvm::sys::path::append(FilePath, "SDKSettings.json");
    if (Error E = addFile(FilePath, /*IgnoreFileError*/ true))
      return E;
  }

  PreprocessorOptions &PPOpts = NewInvocation.getPreprocessorOpts();
  if (PPOpts.ImplicitPCHInclude.empty())
    return Error::success(); // no need for additional work.

  // Go through all the recorded included files; we'll get additional files from
  // the PCH that we need to include in the file list, in case they are
  // referenced while replaying the include-tree.
  SmallVector<const FileEntry *, 32> NotSeenIncludes;
  for (const FileEntry *FE :
       ScanInstance.getPreprocessor().getIncludedFiles()) {
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
    if (!FileNode)
      return FileNode.takeError();
  }

  llvm::ErrorOr<std::optional<cas::ObjectRef>> CASContents =
      FM.getObjectRefForFileContent(PPOpts.ImplicitPCHInclude);
  if (!CASContents)
    return llvm::errorCodeToError(CASContents.getError());
  PCHRef = **CASContents;

  return Error::success();
}

Expected<cas::ObjectRef>
IncludeTreeActionController::getObjectForFile(Preprocessor &PP, FileID FID) {
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

Expected<cas::ObjectRef> IncludeTreeActionController::getObjectForFileNonCached(
    FileManager &FM, const SrcMgr::FileInfo &FI) {
  const FileEntry *FE = FI.getContentCache().OrigEntry;
  assert(FE);

  // Mark the include as already seen.
  if (FE->getUID() >= SeenIncludeFiles.size())
    SeenIncludeFiles.resize(FE->getUID() + 1);
  SeenIncludeFiles.set(FE->getUID());

  return addToFileList(FM, FE);
}

Expected<cas::ObjectRef>
IncludeTreeActionController::getObjectForBuffer(const SrcMgr::FileInfo &FI) {
  // This is a non-file buffer, like the predefines.
  auto Ref = DB.storeFromString(
      {}, FI.getContentCache().getBufferIfLoaded()->getBuffer());
  if (!Ref)
    return Ref.takeError();
  Expected<cas::IncludeFile> FileNode = createIncludeFile(FI.getName(), *Ref);
  if (!FileNode)
    return FileNode.takeError();
  return FileNode->getRef();
}

Expected<cas::ObjectRef>
IncludeTreeActionController::addToFileList(FileManager &FM,
                                           const FileEntry *FE) {
  StringRef Filename = FE->getName();
  llvm::ErrorOr<std::optional<cas::ObjectRef>> CASContents =
      FM.getObjectRefForFileContent(Filename);
  if (!CASContents)
    return llvm::errorCodeToError(CASContents.getError());
  assert(*CASContents);

  auto addFile = [&](StringRef Filename) -> Expected<cas::ObjectRef> {
    assert(!Filename.empty());
    auto FileNode = createIncludeFile(Filename, **CASContents);
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
IncludeTreeActionController::getCASTreeForFileIncludes(FilePPState &&PPState) {
  return cas::IncludeTree::create(DB, PPState.FileCharacteristic, PPState.File,
                                  PPState.Includes, PPState.HasIncludeChecks);
}

Expected<cas::IncludeFile>
IncludeTreeActionController::createIncludeFile(StringRef Filename,
                                               cas::ObjectRef Contents) {
  SmallString<256> MappedPath;
  if (!PrefixMapper.empty()) {
    PrefixMapper.map(Filename, MappedPath);
    Filename = MappedPath;
  }
  return cas::IncludeFile::create(DB, Filename, std::move(Contents));
}

Expected<cas::IncludeTreeRoot> IncludeTreeActionController::getIncludeTree() {
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
    StringRef CWD, const DepscanPrefixMapping &PrefixMapping) {
  EmptyDependencyConsumer Consumer;
  IncludeTreeActionController Controller(DB, PrefixMapping);
  llvm::Error Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, Controller);
  if (Result)
    return std::move(Result);
  return Controller.getIncludeTree();
}

Expected<cas::IncludeTreeRoot>
DependencyScanningTool::getIncludeTreeFromCompilerInvocation(
    cas::ObjectStore &DB, std::shared_ptr<CompilerInvocation> Invocation,
    StringRef CWD, const DepscanPrefixMapping &PrefixMapping,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    bool DiagGenerationAsCompilation) {
  EmptyDependencyConsumer Consumer;
  IncludeTreeActionController Controller(DB, PrefixMapping);
  Worker.computeDependenciesFromCompilerInvocation(
      std::move(Invocation), CWD, Consumer, Controller, DiagsConsumer,
      VerboseOS, DiagGenerationAsCompilation);
  return Controller.getIncludeTree();
}

llvm::Expected<P1689Rule> DependencyScanningTool::getP1689ModuleDependencyFile(
    const CompileCommand &Command, StringRef CWD, std::string &MakeformatOutput,
    std::string &MakeformatOutputPath) {
  class P1689ModuleDependencyPrinterConsumer
      : public MakeDependencyPrinterConsumer {
  public:
    P1689ModuleDependencyPrinterConsumer(P1689Rule &Rule,
                                         const CompileCommand &Command)
        : Filename(Command.Filename), Rule(Rule) {
      Rule.PrimaryOutput = Command.Output;
    }

    void handleProvidedAndRequiredStdCXXModules(
        std::optional<P1689ModuleInfo> Provided,
        std::vector<P1689ModuleInfo> Requires) override {
      Rule.Provides = Provided;
      if (Rule.Provides)
        Rule.Provides->SourcePath = Filename.str();
      Rule.Requires = Requires;
    }

    StringRef getMakeFormatDependencyOutputPath() {
      if (Opts->OutputFormat != DependencyOutputFormat::Make)
        return {};
      return Opts->OutputFile;
    }

  private:
    StringRef Filename;
    P1689Rule &Rule;
  };

  class P1689ActionController : public DependencyActionController {
  public:
    // The lookupModuleOutput is for clang modules. P1689 format don't need it.
    std::string lookupModuleOutput(const ModuleID &,
                                   ModuleOutputKind Kind) override {
      return "";
    }
  };

  P1689Rule Rule;
  P1689ModuleDependencyPrinterConsumer Consumer(Rule, Command);
  P1689ActionController Controller;
  auto Result = Worker.computeDependencies(CWD, Command.CommandLine, Consumer,
                                           Controller);
  if (Result)
    return std::move(Result);

  MakeformatOutputPath = Consumer.getMakeFormatDependencyOutputPath();
  if (!MakeformatOutputPath.empty())
    Consumer.printDependencies(MakeformatOutput);
  return Rule;
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::getTranslationUnitDependencies(
    const std::vector<std::string> &CommandLine, StringRef CWD,
    const llvm::StringSet<> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput,
    DepscanPrefixMapping PrefixMapping) {
  FullDependencyConsumer Consumer(AlreadySeen);
  auto Controller =
      createActionController(LookupModuleOutput, std::move(PrefixMapping));
  llvm::Error Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, *Controller);
  if (Result)
    return std::move(Result);
  return Consumer.takeTranslationUnitDeps();
}

llvm::Expected<ModuleDepsGraph> DependencyScanningTool::getModuleDependencies(
    StringRef ModuleName, const std::vector<std::string> &CommandLine,
    StringRef CWD, const llvm::StringSet<> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput,
    DepscanPrefixMapping PrefixMapping) {
  FullDependencyConsumer Consumer(AlreadySeen);
  auto Controller =
      createActionController(LookupModuleOutput, std::move(PrefixMapping));
  llvm::Error Result = Worker.computeDependencies(CWD, CommandLine, Consumer,
                                                  *Controller, ModuleName);
  if (Result)
    return std::move(Result);
  return Consumer.takeModuleGraphDeps();
}

TranslationUnitDeps FullDependencyConsumer::takeTranslationUnitDeps() {
  TranslationUnitDeps TU;

  TU.ID.ContextHash = std::move(ContextHash);
  TU.FileDeps = std::move(Dependencies);
  TU.PrebuiltModuleDeps = std::move(PrebuiltModuleDeps);
  TU.Commands = std::move(Commands);
  TU.CASFileSystemRootID = std::move(CASFileSystemRootID);

  for (auto &&M : ClangModuleDeps) {
    auto &MD = M.second;
    if (MD.ImportedByMainFile)
      TU.ClangModuleDeps.push_back(MD.ID);
    // TODO: Avoid handleModuleDependency even being called for modules
    //   we've already seen.
    if (AlreadySeen.count(M.first))
      continue;
    TU.ModuleGraph.push_back(std::move(MD));
  }

  return TU;
}

ModuleDepsGraph FullDependencyConsumer::takeModuleGraphDeps() {
  ModuleDepsGraph ModuleGraph;

  for (auto &&M : ClangModuleDeps) {
    auto &MD = M.second;
    // TODO: Avoid handleModuleDependency even being called for modules
    //   we've already seen.
    if (AlreadySeen.count(M.first))
      continue;
    ModuleGraph.push_back(std::move(MD));
  }

  return ModuleGraph;
}

CallbackActionController::~CallbackActionController() {}

std::unique_ptr<DependencyActionController>
DependencyScanningTool::createActionController(
    DependencyScanningWorker &Worker,
    LookupModuleOutputCallback LookupModuleOutput,
    DepscanPrefixMapping PrefixMapping) {
  if (auto CacheFS = Worker.getCASFS())
    return std::make_unique<CASFSActionController>(LookupModuleOutput, *CacheFS,
                                                   std::move(PrefixMapping));
  return std::make_unique<CallbackActionController>(LookupModuleOutput);
}

std::unique_ptr<DependencyActionController>
DependencyScanningTool::createActionController(
    LookupModuleOutputCallback LookupModuleOutput,
    DepscanPrefixMapping PrefixMapping) {
  return createActionController(Worker, std::move(LookupModuleOutput),
                                std::move(PrefixMapping));
}

CASFSActionController::CASFSActionController(
    LookupModuleOutputCallback LookupModuleOutput,
    llvm::cas::CachingOnDiskFileSystem &CacheFS,
    DepscanPrefixMapping PrefixMapping)
    : CallbackActionController(std::move(LookupModuleOutput)), CacheFS(CacheFS),
      PrefixMapping(std::move(PrefixMapping)) {}

Error CASFSActionController::initialize(CompilerInstance &ScanInstance,
                                        CompilerInvocation &NewInvocation) {
  // Setup prefix mapping.
  Mapper.emplace(&CacheFS);
  if (Error E = PrefixMapping.configurePrefixMapper(NewInvocation, *Mapper))
    return E;

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

  Module *M = ModuleScanInstance.getPreprocessor().getCurrentModule();
  assert(M && "finalizing without a module");

  M->setCASFileSystemRootID(RootID->toString());
  return Error::success();
}

Error CASFSActionController::finalizeModuleInvocation(CompilerInvocation &CI,
                                                      const ModuleDeps &MD) {
  if (auto ID = MD.CASFileSystemRootID) {
    configureInvocationForCaching(CI, CASOpts, ID->toString(),
                                  CacheFS.getCurrentWorkingDirectory().get(),
                                  /*ProduceIncludeTree=*/false);
  }

  if (Mapper)
    DepscanPrefixMapping::remapInvocationPaths(CI, *Mapper);

  return llvm::Error::success();
}
