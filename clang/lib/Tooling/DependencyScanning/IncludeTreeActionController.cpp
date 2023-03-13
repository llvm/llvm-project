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
class IncludeTreeActionController : public CallbackActionController {
public:
  IncludeTreeActionController(cas::ObjectStore &DB,
                              DepscanPrefixMapping PrefixMapping)
      : CallbackActionController(nullptr), DB(DB),
        PrefixMapping(std::move(PrefixMapping)) {}

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
  DepscanPrefixMapping PrefixMapping;
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
  DependencyActionController &Controller;
  Preprocessor &PP;

public:
  IncludeTreePPCallbacks(DependencyActionController &Controller,
                         Preprocessor &PP)
      : Controller(Controller), PP(PP) {}

  void LexedFileChanged(FileID FID, LexedFileChangeReason Reason,
                        SrcMgr::CharacteristicKind FileType, FileID PrevFID,
                        SourceLocation Loc) override {
    switch (Reason) {
    case LexedFileChangeReason::EnterFile:
      Controller.enteredInclude(PP, FID);
      break;
    case LexedFileChangeReason::ExitFile: {
      Controller.exitedInclude(PP, FID, PrevFID, Loc);
      break;
    }
    }
  }

  void HasInclude(SourceLocation Loc, StringRef FileName, bool IsAngled,
                  OptionalFileEntryRef File,
                  SrcMgr::CharacteristicKind FileType) override {
    Controller.handleHasIncludeCheck(PP, File.has_value());
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

  // Attach callbacks for the IncludeTree of the TU. The preprocessor
  // does not exist yet, so we need to indirect this via DependencyCollector.
  auto DC = std::make_shared<PPCallbacksDependencyCollector>(
      [this](Preprocessor &PP) {
        return std::make_unique<IncludeTreePPCallbacks>(*this, PP);
      });
  ScanInstance.addDependencyCollector(std::move(DC));

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

  auto FinishIncludeTree = [&]() -> Error {
    PreprocessorOptions &PPOpts = NewInvocation.getPreprocessorOpts();
    if (PPOpts.ImplicitPCHInclude.empty())
      return Error::success(); // no need for additional work.

    // Go through all the recorded included files; we'll get additional files
    // from the PCH that we need to include in the file list, in case they are
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
  };

  if (Error E = FinishIncludeTree())
    return E;

  auto IncludeTreeRoot = getIncludeTree();
  if (!IncludeTreeRoot)
    return IncludeTreeRoot.takeError();

  // FIXME: use configureInvocationForCaching
  NewInvocation.getFrontendOpts().CASIncludeTreeID =
      IncludeTreeRoot->getID().toString();

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

std::unique_ptr<DependencyActionController>
dependencies::createIncludeTreeActionController(
    cas::ObjectStore &DB, DepscanPrefixMapping PrefixMapping) {
  return std::make_unique<IncludeTreeActionController>(
      DB, std::move(PrefixMapping));
}
