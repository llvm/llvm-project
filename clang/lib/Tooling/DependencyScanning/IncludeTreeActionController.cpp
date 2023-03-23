//===- IncludeTreeActionController.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CachingActions.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/PrefixMapper.h"
#include "llvm/Support/PrefixMappingFileSystem.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;
using llvm::Error;

namespace {
class IncludeTreeBuilder;

class IncludeTreeActionController : public CallbackActionController {
public:
  IncludeTreeActionController(cas::ObjectStore &DB,
                              DepscanPrefixMapping PrefixMapping,
                              LookupModuleOutputCallback LookupOutput)
      : CallbackActionController(LookupOutput), DB(DB),
        PrefixMapping(std::move(PrefixMapping)) {}

  Expected<cas::IncludeTreeRoot> getIncludeTree();

private:
  Error initialize(CompilerInstance &ScanInstance,
                   CompilerInvocation &NewInvocation) override;
  Error finalize(CompilerInstance &ScanInstance,
                 CompilerInvocation &NewInvocation) override;

  Error initializeModuleBuild(CompilerInstance &ModuleScanInstance) override;
  Error finalizeModuleBuild(CompilerInstance &ModuleScanInstance) override;
  Error finalizeModuleInvocation(CompilerInvocation &CI,
                                 const ModuleDeps &MD) override;

  const DepscanPrefixMapping *getPrefixMapping() override {
    return &PrefixMapping;
  }

private:
  IncludeTreeBuilder &current() {
    assert(!BuilderStack.empty());
    return *BuilderStack.back();
  }

private:
  cas::ObjectStore &DB;
  CASOptions CASOpts;
  DepscanPrefixMapping PrefixMapping;
  llvm::PrefixMapper PrefixMapper;
  // IncludeTreePPCallbacks keeps a pointer to the current builder, so use a
  // pointer so the builder cannot move when resizing.
  SmallVector<std::unique_ptr<IncludeTreeBuilder>> BuilderStack;
  std::optional<cas::IncludeTreeRoot> IncludeTreeResult;
};

/// Callbacks for building an include-tree for a given translation unit or
/// module. The \c IncludeTreeActionController is responsiblee for pushing and
/// popping builders from the stack as modules are required.
class IncludeTreeBuilder {
public:
  IncludeTreeBuilder(cas::ObjectStore &DB, llvm::PrefixMapper &PrefixMapper)
      : DB(DB), PrefixMapper(PrefixMapper) {}

  Expected<cas::IncludeTreeRoot>
  finishIncludeTree(CompilerInstance &ScanInstance,
                    CompilerInvocation &NewInvocation);

  void enteredInclude(Preprocessor &PP, FileID FID);

  void exitedInclude(Preprocessor &PP, FileID IncludedBy, FileID Include,
                     SourceLocation ExitLoc);

  void handleHasIncludeCheck(Preprocessor &PP, bool Result);

  void moduleImport(Preprocessor &PP, const Module *M, SourceLocation EndLoc);

  void enteredSubmodule(Preprocessor &PP, Module *M, SourceLocation ImportLoc,
                        bool ForPragma);
  void exitedSubmodule(Preprocessor &PP, Module *M, SourceLocation ImportLoc,
                       bool ForPragma);

private:
  struct FilePPState {
    SrcMgr::CharacteristicKind FileCharacteristic;
    cas::ObjectRef File;
    SmallVector<cas::IncludeTree::IncludeInfo, 6> Includes;
    std::optional<cas::ObjectRef> SubmoduleName;
    llvm::SmallBitVector HasIncludeChecks;
  };

  Expected<cas::ObjectRef> getObjectForFile(Preprocessor &PP, FileID FID);
  Expected<cas::ObjectRef>
  getObjectForFileNonCached(FileManager &FM, const SrcMgr::FileInfo &FI);
  Expected<cas::ObjectRef> getObjectForBuffer(const SrcMgr::FileInfo &FI);
  Expected<cas::ObjectRef> addToFileList(FileManager &FM, const FileEntry *FE);
  Expected<cas::IncludeTree> getCASTreeForFileIncludes(FilePPState &&PPState);
  Expected<cas::IncludeTree::File> createIncludeFile(StringRef Filename,
                                                     cas::ObjectRef Contents);

  bool hasErrorOccurred() const { return ErrorToReport.has_value(); }

  template <typename T> std::optional<T> check(Expected<T> &&E) {
    if (!E) {
      ErrorToReport = E.takeError();
      return std::nullopt;
    }
    return *E;
  }

private:
  cas::ObjectStore &DB;
  llvm::PrefixMapper &PrefixMapper;

  std::optional<cas::ObjectRef> PCHRef;
  bool StartedEnteringIncludes = false;
  // When a PCH is used this lists the filenames of the included files as they
  // are recorded in the PCH, ordered by \p FileEntry::UID index.
  SmallVector<StringRef> PreIncludedFileNames;
  llvm::BitVector SeenIncludeFiles;
  SmallVector<cas::IncludeTree::FileList::FileEntry> IncludedFiles;
  std::optional<cas::ObjectRef> PredefinesBufferRef;
  std::optional<cas::ObjectRef> ModuleIncludesBufferRef;
  std::optional<cas::ObjectRef> ModuleMapFileRef;
  /// When the builder is created from an existing tree, the main include tree.
  std::optional<cas::ObjectRef> MainIncludeTreeRef;
  SmallVector<FilePPState> IncludeStack;
  llvm::DenseMap<const FileEntry *, std::optional<cas::ObjectRef>>
      ObjectForFile;
  std::optional<llvm::Error> ErrorToReport;
};

/// A utility for adding \c PPCallbacks to a compiler instance at the
/// appropriate time.
struct PPCallbacksDependencyCollector : public DependencyCollector {
  using MakeCB =
      llvm::unique_function<std::unique_ptr<PPCallbacks>(Preprocessor &)>;
  MakeCB Create;
  PPCallbacksDependencyCollector(MakeCB Create) : Create(std::move(Create)) {}
  void attachToPreprocessor(Preprocessor &PP) final {
    std::unique_ptr<PPCallbacks> CB = Create(PP);
    assert(CB);
    PP.addPPCallbacks(std::move(CB));
  }
};

struct IncludeTreePPCallbacks : public PPCallbacks {
  IncludeTreeBuilder &Builder;
  Preprocessor &PP;

public:
  IncludeTreePPCallbacks(IncludeTreeBuilder &Builder, Preprocessor &PP)
      : Builder(Builder), PP(PP) {}

  void LexedFileChanged(FileID FID, LexedFileChangeReason Reason,
                        SrcMgr::CharacteristicKind FileType, FileID PrevFID,
                        SourceLocation Loc) override {
    switch (Reason) {
    case LexedFileChangeReason::EnterFile:
      Builder.enteredInclude(PP, FID);
      break;
    case LexedFileChangeReason::ExitFile: {
      Builder.exitedInclude(PP, FID, PrevFID, Loc);
      break;
    }
    }
  }

  void HasInclude(SourceLocation Loc, StringRef FileName, bool IsAngled,
                  OptionalFileEntryRef File,
                  SrcMgr::CharacteristicKind FileType) override {
    Builder.handleHasIncludeCheck(PP, File.has_value());
  }

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override {
    if (!Imported)
      return; // File includes handled by LexedFileChanged.

    // Calculate EndLoc for the directive
    // FIXME: pass EndLoc through PPCallbacks; it is already calculated
    SourceManager &SM = PP.getSourceManager();
    std::pair<FileID, unsigned> LocInfo = SM.getDecomposedExpansionLoc(HashLoc);
    StringRef Buffer = SM.getBufferData(LocInfo.first);
    Lexer L(SM.getLocForStartOfFile(LocInfo.first), PP.getLangOpts(),
            Buffer.begin(), Buffer.begin() + LocInfo.second, Buffer.end());
    L.setParsingPreprocessorDirective(true);
    Token Tok;
    do {
      L.LexFromRawLexer(Tok);
    } while (!Tok.isOneOf(tok::eod, tok::eof));
    SourceLocation EndLoc = L.getSourceLocation();

    Builder.moduleImport(PP, Imported, EndLoc);
  }

  void EnteredSubmodule(Module *M, SourceLocation ImportLoc,
                        bool ForPragma) override {
    Builder.enteredSubmodule(PP, M, ImportLoc, ForPragma);
  }
  void LeftSubmodule(Module *M, SourceLocation ImportLoc,
                     bool ForPragma) override {
    Builder.exitedSubmodule(PP, M, ImportLoc, ForPragma);
  }
};
} // namespace

/// The PCH recorded file paths with canonical paths, create a VFS that
/// allows remapping back to the non-canonical source paths so that they are
/// found during dep-scanning.
void dependencies::addReversePrefixMappingFileSystem(
    const llvm::PrefixMapper &PrefixMapper, CompilerInstance &ScanInstance) {
  llvm::PrefixMapper ReverseMapper;
  ReverseMapper.addInverseRange(PrefixMapper.getMappings());
  ReverseMapper.sort();
  std::unique_ptr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPrefixMappingFileSystem(
          std::move(ReverseMapper), &ScanInstance.getVirtualFileSystem());

  ScanInstance.getFileManager().setVirtualFileSystem(std::move(FS));
}

Expected<cas::IncludeTreeRoot> IncludeTreeActionController::getIncludeTree() {
  if (IncludeTreeResult)
    return *IncludeTreeResult;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "failed to produce include-tree");
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

  BuilderStack.push_back(
      std::make_unique<IncludeTreeBuilder>(DB, PrefixMapper));

  // Attach callbacks for the IncludeTree of the TU. The preprocessor
  // does not exist yet, so we need to indirect this via DependencyCollector.
  auto DC = std::make_shared<PPCallbacksDependencyCollector>(
      [&Builder = current()](Preprocessor &PP) {
        return std::make_unique<IncludeTreePPCallbacks>(Builder, PP);
      });
  ScanInstance.addDependencyCollector(std::move(DC));

  // Enable caching in the resulting commands.
  ScanInstance.getFrontendOpts().CacheCompileJob = true;
  CASOpts = ScanInstance.getCASOpts();

  return Error::success();
}

Error IncludeTreeActionController::finalize(CompilerInstance &ScanInstance,
                                            CompilerInvocation &NewInvocation) {
  assert(!IncludeTreeResult);
  assert(BuilderStack.size() == 1);
  auto Builder = BuilderStack.pop_back_val();
  Error E = Builder->finishIncludeTree(ScanInstance, NewInvocation)
                .moveInto(IncludeTreeResult);
  if (E)
    return E;

  configureInvocationForCaching(NewInvocation, CASOpts,
                                IncludeTreeResult->getID().toString(),
                                // FIXME: working dir?
                                /*CASFSWorkingDir=*/"",
                                /*ProduceIncludeTree=*/true);

  DepscanPrefixMapping::remapInvocationPaths(NewInvocation, PrefixMapper);

  return Error::success();
}

Error IncludeTreeActionController::initializeModuleBuild(
    CompilerInstance &ModuleScanInstance) {
  BuilderStack.push_back(
      std::make_unique<IncludeTreeBuilder>(DB, PrefixMapper));

  // Attach callbacks for the IncludeTree of the module. The preprocessor
  // does not exist yet, so we need to indirect this via DependencyCollector.
  auto DC = std::make_shared<PPCallbacksDependencyCollector>(
      [&Builder = current()](Preprocessor &PP) {
        return std::make_unique<IncludeTreePPCallbacks>(Builder, PP);
      });
  ModuleScanInstance.addDependencyCollector(std::move(DC));

  return Error::success();
}

Error IncludeTreeActionController::finalizeModuleBuild(
    CompilerInstance &ModuleScanInstance) {
  // FIXME: the scan invocation is incorrect here; we need the `NewInvocation`
  // from `finalizeModuleInvocation` to finish the tree.
  auto Builder = BuilderStack.pop_back_val();
  auto Tree = Builder->finishIncludeTree(ModuleScanInstance,
                                         ModuleScanInstance.getInvocation());
  if (!Tree)
    return Tree.takeError();

  ModuleScanInstance.getASTContext().setCASIncludeTreeID(
      Tree->getID().toString());

  return Error::success();
}

Error IncludeTreeActionController::finalizeModuleInvocation(
    CompilerInvocation &CI, const ModuleDeps &MD) {
  if (auto ID = MD.IncludeTreeID) {
    configureInvocationForCaching(CI, CASOpts, std::move(*ID),
                                  /*CASFSWorkingDir=*/"",
                                  /*ProduceIncludeTree=*/true);
  }

  DepscanPrefixMapping::remapInvocationPaths(CI, PrefixMapper);
  return Error::success();
}

void IncludeTreeBuilder::enteredInclude(Preprocessor &PP, FileID FID) {
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
  IncludeStack.push_back({FI.getFileCharacteristic(), *FileRef, {}, {}, {}});
}

void IncludeTreeBuilder::exitedInclude(Preprocessor &PP, FileID IncludedBy,
                                       FileID Include, SourceLocation ExitLoc) {
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
  IncludeStack.back().Includes.push_back({IncludeTree->getRef(), LocInfo.second,
                                          cas::IncludeTree::NodeKind::Tree});
}

void IncludeTreeBuilder::handleHasIncludeCheck(Preprocessor &PP, bool Result) {
  if (hasErrorOccurred())
    return;

  IncludeStack.back().HasIncludeChecks.push_back(Result);
}

void IncludeTreeBuilder::moduleImport(Preprocessor &PP, const Module *M,
                                      SourceLocation EndLoc) {
  auto Import =
      check(cas::IncludeTree::ModuleImport::create(DB, M->getFullModuleName()));
  if (!Import)
    return;

  std::pair<FileID, unsigned> EndLocInfo =
      PP.getSourceManager().getDecomposedExpansionLoc(EndLoc);
  IncludeStack.back().Includes.push_back(
      {Import->getRef(), EndLocInfo.second,
       cas::IncludeTree::NodeKind::ModuleImport});
}

void IncludeTreeBuilder::enteredSubmodule(Preprocessor &PP, Module *M,
                                          SourceLocation ImportLoc,
                                          bool ForPragma) {
  if (ForPragma)
    return; // Will be parsed as normal.
  if (hasErrorOccurred())
    return;
  assert(!IncludeStack.back().SubmoduleName && "repeated enteredSubmodule");
  auto Ref = check(DB.storeFromString({}, M->getFullModuleName()));
  IncludeStack.back().SubmoduleName = Ref;
}
void IncludeTreeBuilder::exitedSubmodule(Preprocessor &PP, Module *M,
                                         SourceLocation ImportLoc,
                                         bool ForPragma) {
  // Submodule exit is handled automatically when leaving a modular file.
}

Expected<cas::IncludeTreeRoot>
IncludeTreeBuilder::finishIncludeTree(CompilerInstance &ScanInstance,
                                      CompilerInvocation &NewInvocation) {
  if (ErrorToReport)
    return std::move(*ErrorToReport);

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
      return std::move(E);
  }
  // Add profile files.
  // FIXME: Do not have the logic here to determine which path should be set
  // but ideally only the path needed for the compilation is set and we already
  // checked the file needed exists. Just try load and ignore errors.
  if (Error E = addFile(NewInvocation.getCodeGenOpts().ProfileInstrumentUsePath,
                        /*IgnoreFileError=*/true))
    return std::move(E);
  if (Error E = addFile(NewInvocation.getCodeGenOpts().SampleProfileFile,
                        /*IgnoreFileError=*/true))
    return std::move(E);
  if (Error E = addFile(NewInvocation.getCodeGenOpts().ProfileRemappingFile,
                        /*IgnoreFileError=*/true))
    return std::move(E);

  StringRef Sysroot = NewInvocation.getHeaderSearchOpts().Sysroot;
  if (!Sysroot.empty()) {
    // Include 'SDKSettings.json', if it exists, to accomodate availability
    // checks during the compilation.
    llvm::SmallString<256> FilePath = Sysroot;
    llvm::sys::path::append(FilePath, "SDKSettings.json");
    if (Error E = addFile(FilePath, /*IgnoreFileError*/ true))
      return std::move(E);
  }

  auto &FrontendOpts = NewInvocation.getFrontendOpts();
  if (!FrontendOpts.Inputs.empty() &&
      FrontendOpts.Inputs[0].getKind().getFormat() == InputKind::ModuleMap) {
    // FIXME: handle inferred module maps
    Expected<FileEntryRef> FE = FM.getFileRef(FrontendOpts.Inputs[0].getFile());
    if (!FE)
      return FE.takeError();
    if (Error E = addToFileList(FM, *FE).moveInto(ModuleMapFileRef))
      return std::move(E);
  }

  for (StringRef ModuleMap : FrontendOpts.ModuleMapFiles)
    if (Error E = addFile(ModuleMap))
      return std::move(E);

  auto FinishIncludeTree = [&]() -> Error {
    IntrusiveRefCntPtr<ASTReader> Reader = ScanInstance.getASTReader();
    if (!Reader)
      return Error::success(); // no need for additional work.

    // Go through all the recorded input files.
    SmallVector<const FileEntry *, 32> NotSeenIncludes;
    for (serialization::ModuleFile &MF : Reader->getModuleManager()) {
      if (hasErrorOccurred())
        break;
      Reader->visitInputFiles(
          MF, /*IncludeSystem=*/true, /*Complain=*/false,
          [&](const serialization::InputFile &IF, bool isSystem) {
            OptionalFileEntryRef FE = IF.getFile();
            assert(FE);
            if (FE->getUID() >= SeenIncludeFiles.size() ||
                !SeenIncludeFiles[FE->getUID()])
              NotSeenIncludes.push_back(*FE);
          });
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

    PreprocessorOptions &PPOpts = NewInvocation.getPreprocessorOpts();
    if (PPOpts.ImplicitPCHInclude.empty())
      return Error::success(); // no need for additional work.

    llvm::ErrorOr<std::optional<cas::ObjectRef>> CASContents =
        FM.getObjectRefForFileContent(PPOpts.ImplicitPCHInclude);
    if (!CASContents)
      return llvm::errorCodeToError(CASContents.getError());
    PCHRef = **CASContents;

    return Error::success();
  };

  if (Error E = FinishIncludeTree())
    return std::move(E);

  if (ErrorToReport)
    return std::move(*ErrorToReport);

  assert(IncludeStack.size() == 1);
  Expected<cas::IncludeTree> MainIncludeTree =
      getCASTreeForFileIncludes(IncludeStack.pop_back_val());
  if (!MainIncludeTree)
    return MainIncludeTree.takeError();
  auto FileList = cas::IncludeTree::FileList::create(DB, IncludedFiles);
  if (!FileList)
    return FileList.takeError();

  return cas::IncludeTreeRoot::create(DB, MainIncludeTree->getRef(),
                                      FileList->getRef(), PCHRef,
                                      ModuleMapFileRef);
}

Expected<cas::ObjectRef> IncludeTreeBuilder::getObjectForFile(Preprocessor &PP,
                                                              FileID FID) {
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
  if (!FI.getContentCache().OrigEntry &&
      FI.getName() == Module::getModuleInputBufferName()) {
    // Virtual <module-includes> buffer
    if (!ModuleIncludesBufferRef) {
      if (Error E = getObjectForBuffer(FI).moveInto(ModuleIncludesBufferRef))
        return std::move(E);
    }
    return *ModuleIncludesBufferRef;
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
IncludeTreeBuilder::getObjectForFileNonCached(FileManager &FM,
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
IncludeTreeBuilder::getObjectForBuffer(const SrcMgr::FileInfo &FI) {
  // This is a non-file buffer, like the predefines.
  auto Ref = DB.storeFromString(
      {}, FI.getContentCache().getBufferIfLoaded()->getBuffer());
  if (!Ref)
    return Ref.takeError();
  Expected<cas::IncludeTree::File> FileNode =
      createIncludeFile(FI.getName(), *Ref);
  if (!FileNode)
    return FileNode.takeError();
  return FileNode->getRef();
}

Expected<cas::ObjectRef>
IncludeTreeBuilder::addToFileList(FileManager &FM, const FileEntry *FE) {
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
         static_cast<cas::IncludeTree::FileList::FileSizeTy>(FE->getSize())});
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
IncludeTreeBuilder::getCASTreeForFileIncludes(FilePPState &&PPState) {
  return cas::IncludeTree::create(DB, PPState.FileCharacteristic, PPState.File,
                                  PPState.Includes, PPState.SubmoduleName,
                                  PPState.HasIncludeChecks);
}

Expected<cas::IncludeTree::File>
IncludeTreeBuilder::createIncludeFile(StringRef Filename,
                                      cas::ObjectRef Contents) {
  SmallString<256> MappedPath;
  if (!PrefixMapper.empty()) {
    PrefixMapper.map(Filename, MappedPath);
    Filename = MappedPath;
  }
  return cas::IncludeTree::File::create(DB, Filename, std::move(Contents));
}

std::unique_ptr<DependencyActionController>
dependencies::createIncludeTreeActionController(
    LookupModuleOutputCallback LookupModuleOutput, cas::ObjectStore &DB,
    DepscanPrefixMapping PrefixMapping) {
  return std::make_unique<IncludeTreeActionController>(
      DB, std::move(PrefixMapping), LookupModuleOutput);
}
