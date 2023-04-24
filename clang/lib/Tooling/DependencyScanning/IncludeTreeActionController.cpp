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
                              LookupModuleOutputCallback LookupOutput)
      : CallbackActionController(LookupOutput), DB(DB) {}

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

private:
  IncludeTreeBuilder &current() {
    assert(!BuilderStack.empty());
    return *BuilderStack.back();
  }

private:
  cas::ObjectStore &DB;
  CASOptions CASOpts;
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

  Error addModuleInputs(ASTReader &Reader);
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
  SmallVector<cas::ObjectRef> IncludedFileLists;
  std::optional<cas::ObjectRef> PredefinesBufferRef;
  std::optional<cas::ObjectRef> ModuleIncludesBufferRef;
  std::optional<cas::ObjectRef> ModuleMapRef;
  /// When the builder is created from an existing tree, the main include tree.
  std::optional<cas::ObjectRef> MainIncludeTreeRef;
  SmallVector<FilePPState> IncludeStack;
  llvm::DenseMap<const FileEntry *, std::optional<cas::ObjectRef>>
      ObjectForFile;
  std::optional<llvm::Error> ErrorToReport;
};

/// A utility for adding \c PPCallbacks and/or \cASTReaderListener to a compiler
/// instance at the appropriate time.
struct AttachOnlyDependencyCollector : public DependencyCollector {
  using MakePPCB =
      llvm::unique_function<std::unique_ptr<PPCallbacks>(Preprocessor &)>;
  using MakeASTReaderL =
      llvm::unique_function<std::unique_ptr<ASTReaderListener>(ASTReader &R)>;
  MakePPCB CreatePPCB;
  MakeASTReaderL CreateASTReaderL;
  AttachOnlyDependencyCollector(MakePPCB CreatePPCB, MakeASTReaderL CreateL)
      : CreatePPCB(std::move(CreatePPCB)),
        CreateASTReaderL(std::move(CreateL)) {}

  void attachToPreprocessor(Preprocessor &PP) final {
    if (CreatePPCB) {
      std::unique_ptr<PPCallbacks> CB = CreatePPCB(PP);
      assert(CB);
      PP.addPPCallbacks(std::move(CB));
    }
  }

  void attachToASTReader(ASTReader &R) final {
    if (CreateASTReaderL) {
      std::unique_ptr<ASTReaderListener> L = CreateASTReaderL(R);
      assert(L);
      R.addListener(std::move(L));
    }
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

/// Utility to trigger module lookup in header search for modules loaded via
/// PCH. This causes dependency scanning via PCH to parse modulemap files at
/// roughly the same point they would with modulemap files embedded in the pcms,
/// which is disabled with include-tree modules. Without this, we can fail to
/// find modules that are in the same directory as a named import, since
/// it may be skipped during search (see \c loadFrameworkModule).
///
/// The specific lookup we do matches what happens in ASTReader for the
/// MODULE_DIRECTORY record, and ignores the result.
class LookupPCHModulesListener : public ASTReaderListener {
public:
  LookupPCHModulesListener(ASTReader &R) : Reader(R) {}

private:
  void visitModuleFile(StringRef Filename,
                       serialization::ModuleKind Kind) final {
    // Any prebuilt or explicit modules seen during scanning are "full" modules
    // rather than implicitly built scanner modules.
    if (Kind == serialization::MK_PrebuiltModule ||
        Kind == serialization::MK_ExplicitModule) {
      serialization::ModuleManager &Manager = Reader.getModuleManager();
      serialization::ModuleFile *MF = Manager.lookupByFileName(Filename);
      assert(MF && "module file missing in visitModuleFile");
      // Match MODULE_DIRECTORY: allow full search and ignore failure to find
      // the module.
      HeaderSearch &HS = Reader.getPreprocessor().getHeaderSearchInfo();
      (void)HS.lookupModule(MF->ModuleName, SourceLocation(),
                            /*AllowSearch=*/true,
                            /*AllowExtraModuleMapSearch=*/true);
    }
  }

private:
  ASTReader &Reader;
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
  DepscanPrefixMapping::configurePrefixMapper(NewInvocation, PrefixMapper);

  auto ensurePathRemapping = [&]() {
    if (PrefixMapper.empty())
      return;

    PreprocessorOptions &PPOpts = ScanInstance.getPreprocessorOpts();
    if (PPOpts.Includes.empty() && PPOpts.ImplicitPCHInclude.empty() &&
        !ScanInstance.getLangOpts().Modules)
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
  auto DC = std::make_shared<AttachOnlyDependencyCollector>(
      [&Builder = current()](Preprocessor &PP) {
        return std::make_unique<IncludeTreePPCallbacks>(Builder, PP);
      },
      [](ASTReader &R) {
        return std::make_unique<LookupPCHModulesListener>(R);
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
  auto DC = std::make_shared<AttachOnlyDependencyCollector>(
      [&Builder = current()](Preprocessor &PP) {
        return std::make_unique<IncludeTreePPCallbacks>(Builder, PP);
      },
      [](ASTReader &R) {
        return std::make_unique<LookupPCHModulesListener>(R);
      });
  ModuleScanInstance.addDependencyCollector(std::move(DC));
  ModuleScanInstance.setPrefixMapper(PrefixMapper);

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
  if (!MD.IncludeTreeID)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "missing include-tree for module '%s'",
                                   MD.ID.ModuleName.c_str());

  configureInvocationForCaching(CI, CASOpts, *MD.IncludeTreeID,
                                /*CASFSWorkingDir=*/"",
                                /*ProduceIncludeTree=*/true);

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
  bool VisibilityOnly = M->isForBuilding(PP.getLangOpts());
  auto Import = check(cas::IncludeTree::ModuleImport::create(
      DB, M->getFullModuleName(), VisibilityOnly));
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

static Expected<cas::IncludeTree::Module>
getIncludeTreeModule(cas::ObjectStore &DB, Module *M) {
  using ITModule = cas::IncludeTree::Module;
  SmallVector<cas::ObjectRef> Submodules;
  for (Module *Sub : M->submodules()) {
    Expected<ITModule> SubTree = getIncludeTreeModule(DB, Sub);
    if (!SubTree)
      return SubTree.takeError();
    Submodules.push_back(SubTree->getRef());
  }

  ITModule::ModuleFlags Flags;
  Flags.IsFramework = M->IsFramework;
  Flags.IsExplicit = M->IsExplicit;
  Flags.IsExternC = M->IsExternC;
  Flags.IsSystem = M->IsSystem;
  Flags.InferSubmodules = M->InferSubmodules;
  Flags.InferExplicitSubmodules = M->InferExplicitSubmodules;
  Flags.InferExportWildcard = M->InferExportWildcard;

  bool GlobalWildcardExport = false;
  SmallVector<ITModule::ExportList::Export> Exports;
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  for (Module::ExportDecl &Export : M->Exports) {
    if (Export.getPointer() == nullptr && Export.getInt()) {
      GlobalWildcardExport = true;
    } else if (Export.getPointer()) {
      StringRef Name = Saver.save(Export.getPointer()->getFullModuleName());
      Exports.push_back({Name, Export.getInt()});
    }
  }
  std::optional<cas::ObjectRef> ExportList;
  if (GlobalWildcardExport || !Exports.empty()) {
    auto EL = ITModule::ExportList::create(DB, Exports, GlobalWildcardExport);
    if (!EL)
      return EL.takeError();
    ExportList = EL->getRef();
  }

  SmallVector<ITModule::LinkLibraryList::LinkLibrary> Libraries;
  for (Module::LinkLibrary &LL : M->LinkLibraries) {
    Libraries.push_back({LL.Library, LL.IsFramework});
  }
  std::optional<cas::ObjectRef> LinkLibraries;
  if (!Libraries.empty()) {
    auto LL = ITModule::LinkLibraryList::create(DB, Libraries);
    if (!LL)
      return LL.takeError();
    LinkLibraries = LL->getRef();
  }

  return ITModule::create(DB, M->Name, Flags, Submodules, ExportList,
                          LinkLibraries);
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

  auto FinishIncludeTree = [&]() -> Error {
    IntrusiveRefCntPtr<ASTReader> Reader = ScanInstance.getASTReader();
    if (!Reader)
      return Error::success(); // no need for additional work.

    // Go through all the recorded input files.
    if (Error E = addModuleInputs(*Reader))
      return E;

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

  if (!ScanInstance.getLangOpts().CurrentModule.empty()) {
    SmallVector<cas::ObjectRef> Modules;
    auto AddModule = [&](Module *M) -> llvm::Error {
      Expected<cas::IncludeTree::Module> Mod = getIncludeTreeModule(DB, M);
      if (!Mod)
        return Mod.takeError();
      Modules.push_back(Mod->getRef());
      return Error::success();
    };
    if (Module *M = ScanInstance.getPreprocessor().getCurrentModule()) {
      if (Error E = AddModule(M))
        return std::move(E);
    } else {
      // When building a TU or PCH, we can have headers files that are part of
      // both the public and private modules that are included textually. In
      // that case we need both of those modules.
      ModuleMap &MMap =
          ScanInstance.getPreprocessor().getHeaderSearchInfo().getModuleMap();
      if (Module *M = MMap.findModule(ScanInstance.getLangOpts().CurrentModule))
        if (Error E = AddModule(M))
          return std::move(E);
      if (Module *PM =
          MMap.findModule(ScanInstance.getLangOpts().ModuleName + "_Private"))
        if (Error E = AddModule(PM))
          return std::move(E);
    }

    auto ModMap = cas::IncludeTree::ModuleMap::create(DB, Modules);
    if (!ModMap)
      return ModMap.takeError();
    ModuleMapRef = ModMap->getRef();
  }

  auto FileList =
      cas::IncludeTree::FileList::create(DB, IncludedFiles, IncludedFileLists);
  if (!FileList)
    return FileList.takeError();

  return cas::IncludeTreeRoot::create(DB, MainIncludeTree->getRef(),
                                      FileList->getRef(), PCHRef, ModuleMapRef);
}

Error IncludeTreeBuilder::addModuleInputs(ASTReader &Reader) {
  for (serialization::ModuleFile &MF : Reader.getModuleManager()) {
    // Only add direct imports to avoid duplication. Each include tree is a
    // superset of its imported modules' include trees.
    if (!MF.isDirectlyImported())
      continue;

    assert(!MF.IncludeTreeID.empty() && "missing include-tree for import");

    std::optional<cas::CASID> ID;
    if (Error E = DB.parseID(MF.IncludeTreeID).moveInto(ID))
      return E;
    std::optional<cas::ObjectRef> Ref = DB.getReference(*ID);
    if (!Ref)
      return DB.createUnknownObjectError(*ID);
    std::optional<cas::IncludeTreeRoot> Root;
    if (Error E = cas::IncludeTreeRoot::get(DB, *Ref).moveInto(Root))
      return E;

    IncludedFileLists.push_back(Root->getFileListRef());
  }

  return Error::success();
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
    LookupModuleOutputCallback LookupModuleOutput, cas::ObjectStore &DB) {
  return std::make_unique<IncludeTreeActionController>(DB, LookupModuleOutput);
}
