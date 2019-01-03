//===- IndexingAction.cpp - Frontend index action -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexingAction.h"
#include "ClangIndexRecordWriter.h"
#include "FileIndexRecord.h"
#include "IndexDataStoreUtils.h"
#include "IndexingContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexUnitWriter.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace clang;
using namespace clang::index;

bool IndexDataConsumer::handleDeclOccurence(const Decl *D, SymbolRoleSet Roles,
                                            ArrayRef<SymbolRelation> Relations,
                                            SourceLocation Loc,
                                            ASTNodeInfo ASTNode) {
  return true;
}

bool IndexDataConsumer::handleMacroOccurence(const IdentifierInfo *Name,
                                             const MacroInfo *MI,
                                             SymbolRoleSet Roles,
                                             SourceLocation Loc) {
  return true;
}

bool IndexDataConsumer::handleModuleOccurence(const ImportDecl *ImportD,
                                              const Module *Mod,
                                              SymbolRoleSet Roles,
                                              SourceLocation Loc) {
  return true;
}

namespace {

class IndexASTConsumer : public ASTConsumer {
  std::shared_ptr<Preprocessor> PP;
  std::shared_ptr<IndexingContext> IndexCtx;

public:
  IndexASTConsumer(std::shared_ptr<Preprocessor> PP,
                   std::shared_ptr<IndexingContext> IndexCtx)
      : PP(std::move(PP)), IndexCtx(std::move(IndexCtx)) {}

protected:
  void Initialize(ASTContext &Context) override {
    IndexCtx->setASTContext(Context);
    IndexCtx->getDataConsumer().initialize(Context);
    IndexCtx->getDataConsumer().setPreprocessor(PP);
  }

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    return IndexCtx->indexDeclGroupRef(DG);
  }

  void HandleInterestingDecl(DeclGroupRef DG) override {
    // Ignore deserialized decls.
  }

  void HandleTopLevelDeclInObjCContainer(DeclGroupRef DG) override {
    IndexCtx->indexDeclGroupRef(DG);
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {}
};

class IndexPPCallbacks : public PPCallbacks {
  std::shared_ptr<IndexingContext> IndexCtx;

public:
  IndexPPCallbacks(std::shared_ptr<IndexingContext> IndexCtx)
      : IndexCtx(std::move(IndexCtx)) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   Range.getBegin(), *MD.getMacroInfo());
  }

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    IndexCtx->handleMacroDefined(*MacroNameTok.getIdentifierInfo(),
                                 MacroNameTok.getLocation(),
                                 *MD->getMacroInfo());
  }

  void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                      const MacroDirective *Undef) override {
    if (!MD.getMacroInfo())  // Ignore noop #undef.
      return;
    IndexCtx->handleMacroUndefined(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
};

class IndexActionBase {
protected:
  std::shared_ptr<IndexDataConsumer> DataConsumer;
  std::shared_ptr<IndexingContext> IndexCtx;

  IndexActionBase(std::shared_ptr<IndexDataConsumer> dataConsumer,
                  IndexingOptions Opts)
      : DataConsumer(std::move(dataConsumer)),
        IndexCtx(new IndexingContext(Opts, *DataConsumer)) {}

  std::unique_ptr<IndexASTConsumer>
  createIndexASTConsumer(CompilerInstance &CI) {
    IndexCtx->setSysrootPath(CI.getHeaderSearchOpts().Sysroot);
    return llvm::make_unique<IndexASTConsumer>(CI.getPreprocessorPtr(),
                                               IndexCtx);
  }

  std::unique_ptr<PPCallbacks> createIndexPPCallbacks() {
    return llvm::make_unique<IndexPPCallbacks>(IndexCtx);
  }

  void finish() {
    DataConsumer->finish();
  }
};

class IndexAction : public ASTFrontendAction, IndexActionBase {
public:
  IndexAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
              IndexingOptions Opts)
      : IndexActionBase(std::move(DataConsumer), Opts) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return createIndexASTConsumer(CI);
  }

  bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
    CI.getPreprocessor().addPPCallbacks(createIndexPPCallbacks());
    return true;
  }

  void EndSourceFileAction() override {
    FrontendAction::EndSourceFileAction();
    finish();
  }
};

class WrappingIndexAction : public WrapperFrontendAction, IndexActionBase {
  bool CreatedASTConsumer = false;

public:
  WrappingIndexAction(std::unique_ptr<FrontendAction> WrappedAction,
                      std::shared_ptr<IndexDataConsumer> DataConsumer,
                      IndexingOptions Opts)
      : WrapperFrontendAction(std::move(WrappedAction)),
        IndexActionBase(std::move(DataConsumer), Opts) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    auto OtherConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
    if (!OtherConsumer)
      return nullptr;

    CreatedASTConsumer = true;
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    Consumers.push_back(std::move(OtherConsumer));
    Consumers.push_back(createIndexASTConsumer(CI));
    return llvm::make_unique<MultiplexConsumer>(std::move(Consumers));
  }

  bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
    WrapperFrontendAction::BeginSourceFileAction(CI);
    CI.getPreprocessor().addPPCallbacks(createIndexPPCallbacks());
    return true;
  }

  void EndSourceFileAction() override {
    // Invoke wrapped action's method.
    WrapperFrontendAction::EndSourceFileAction();
    if (CreatedASTConsumer)
      finish();
  }
};

} // anonymous namespace

std::unique_ptr<FrontendAction>
index::createIndexingAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
                            IndexingOptions Opts,
                            std::unique_ptr<FrontendAction> WrappedAction) {
  if (WrappedAction)
    return llvm::make_unique<WrappingIndexAction>(
        std::move(WrappedAction), std::move(DataConsumer), Opts);
  return llvm::make_unique<IndexAction>(std::move(DataConsumer), Opts);
}

static bool topLevelDeclVisitor(void *context, const Decl *D) {
  IndexingContext &IndexCtx = *static_cast<IndexingContext *>(context);
  return IndexCtx.indexTopLevelDecl(D);
}

static void indexTranslationUnit(ASTUnit &Unit, IndexingContext &IndexCtx) {
  Unit.visitLocalTopLevelDecls(&IndexCtx, topLevelDeclVisitor);
}

static void indexPreprocessorMacros(const Preprocessor &PP,
                                    IndexDataConsumer &DataConsumer) {
  for (const auto &M : PP.macros())
    if (MacroDirective *MD = M.second.getLatest())
      DataConsumer.handleMacroOccurence(
          M.first, MD->getMacroInfo(),
          static_cast<unsigned>(index::SymbolRole::Definition),
          MD->getLocation());
}

void index::indexASTUnit(ASTUnit &Unit, IndexDataConsumer &DataConsumer,
                         IndexingOptions Opts) {
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Unit.getASTContext());
  DataConsumer.initialize(Unit.getASTContext());
  DataConsumer.setPreprocessor(Unit.getPreprocessorPtr());

  if (Opts.IndexMacrosInPreprocessor)
    indexPreprocessorMacros(Unit.getPreprocessor(), DataConsumer);
  indexTranslationUnit(Unit, IndexCtx);
  DataConsumer.finish();
}

void index::indexTopLevelDecls(ASTContext &Ctx, Preprocessor &PP,
                               ArrayRef<const Decl *> Decls,
                               IndexDataConsumer &DataConsumer,
                               IndexingOptions Opts) {
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Ctx);

  DataConsumer.initialize(Ctx);

  if (Opts.IndexMacrosInPreprocessor)
    indexPreprocessorMacros(PP, DataConsumer);

  for (const Decl *D : Decls)
    IndexCtx.indexTopLevelDecl(D);
  DataConsumer.finish();
}

std::unique_ptr<PPCallbacks>
index::indexMacrosCallback(IndexDataConsumer &Consumer, IndexingOptions Opts) {
  return llvm::make_unique<IndexPPCallbacks>(
      std::make_shared<IndexingContext>(Opts, Consumer));
}

void index::indexModuleFile(serialization::ModuleFile &Mod, ASTReader &Reader,
                            IndexDataConsumer &DataConsumer,
                            IndexingOptions Opts) {
  ASTContext &Ctx = Reader.getContext();
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Ctx);
  DataConsumer.initialize(Ctx);

  if (Opts.IndexMacrosInPreprocessor)
    indexPreprocessorMacros(Reader.getPreprocessor(), DataConsumer);

  for (const Decl *D : Reader.getModuleFileLevelDecls(Mod)) {
    IndexCtx.indexTopLevelDecl(D);
  }
  DataConsumer.finish();
}

//===----------------------------------------------------------------------===//
// Index Data Recording
//===----------------------------------------------------------------------===//

namespace {

class IndexDataRecorder : public IndexDataConsumer {
  IndexingContext *IndexCtx = nullptr;
  const Preprocessor *PP = nullptr;
  typedef llvm::DenseMap<FileID, std::unique_ptr<FileIndexRecord>>
      RecordByFileTy;
  RecordByFileTy RecordByFile;

public:
  void init(IndexingContext *idxCtx, const CompilerInstance &CI) {
    IndexCtx = idxCtx;
    PP = &CI.getPreprocessor();
    initialize(CI.getASTContext());
  }

  RecordByFileTy::const_iterator record_begin() const {
    return RecordByFile.begin();
  }
  RecordByFileTy::const_iterator record_end() const {
    return RecordByFile.end();
  }
  bool record_empty() const { return RecordByFile.empty(); }

private:
  bool handleDeclOccurence(const Decl *D, SymbolRoleSet Roles,
                           ArrayRef<SymbolRelation> Relations,
                           SourceLocation Loc, ASTNodeInfo ASTNode) override {
    SourceManager &SM = PP->getSourceManager();
    Loc = SM.getFileLoc(Loc);
    if (Loc.isInvalid())
      return true;

    FileID FID;
    unsigned Offset;
    std::tie(FID, Offset) = SM.getDecomposedLoc(Loc);

    if (FID.isInvalid())
      return true;

    // Ignore the predefines buffer.
    const FileEntry *FE = PP->getSourceManager().getFileEntryForID(FID);
    if (!FE)
      return true;

    FileIndexRecord &Rec = getFileIndexRecord(FID);
    Rec.addDeclOccurence(Roles, Offset, D, Relations);
    return true;
  }

  FileIndexRecord &getFileIndexRecord(FileID FID) {
    auto &Entry = RecordByFile[FID];
    if (!Entry) {
      Entry.reset(new FileIndexRecord(FID, IndexCtx->isSystemFile(FID)));
    }
    return *Entry;
  }
};

struct IncludeLocation {
  const FileEntry *Source;
  const FileEntry *Target;
  unsigned Line;
};

class IncludePPCallbacks : public PPCallbacks {
  IndexingContext &IndexCtx;
  RecordingOptions RecordOpts;
  std::vector<IncludeLocation> &Includes;
  SourceManager &SourceMgr;

public:
  IncludePPCallbacks(IndexingContext &indexCtx, RecordingOptions recordOpts,
                     std::vector<IncludeLocation> &IncludesForFile,
                     SourceManager &SourceMgr)
      : IndexCtx(indexCtx), RecordOpts(recordOpts), Includes(IncludesForFile),
        SourceMgr(SourceMgr) {}

private:
  void addInclude(SourceLocation From, const FileEntry *To) {
    assert(To);
    if (RecordOpts.RecordIncludes ==
        RecordingOptions::IncludesRecordingKind::None)
      return;

    std::pair<FileID, unsigned> LocInfo =
        SourceMgr.getDecomposedExpansionLoc(From);
    switch (RecordOpts.RecordIncludes) {
    case RecordingOptions::IncludesRecordingKind::None:
      llvm_unreachable("should have already checked in the beginning");
    case RecordingOptions::IncludesRecordingKind::UserOnly:
      if (IndexCtx.isSystemFile(LocInfo.first))
        return; // Ignore includes of system headers.
      break;
    case RecordingOptions::IncludesRecordingKind::All:
      break;
    }
    auto *FE = SourceMgr.getFileEntryForID(LocInfo.first);
    if (!FE)
      return;
    auto lineNo = SourceMgr.getLineNumber(LocInfo.first, LocInfo.second);
    Includes.push_back({FE, To, lineNo});
  }

  virtual void InclusionDirective(
      SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
      bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
      StringRef SearchPath, StringRef RelativePath, const Module *Imported,
      SrcMgr::CharacteristicKind FileType) override {
    if (HashLoc.isFileID() && File && File->isValid())
      addInclude(HashLoc, File);
  }
};

class IndexDependencyProvider {
public:
  virtual ~IndexDependencyProvider() {}

  virtual void visitFileDependencies(
      const CompilerInstance &CI,
      llvm::function_ref<void(const FileEntry *FE, bool isSystem)> visitor) = 0;
  virtual void
  visitIncludes(llvm::function_ref<void(const FileEntry *Source, unsigned Line,
                                        const FileEntry *Target)>
                    visitor) = 0;
  virtual void visitModuleImports(
      const CompilerInstance &CI,
      llvm::function_ref<void(serialization::ModuleFile &Mod, bool isSystem)>
          visitor) = 0;
};

class SourceFilesIndexDependencyCollector : public DependencyCollector,
                                            public IndexDependencyProvider {
  IndexingContext &IndexCtx;
  RecordingOptions RecordOpts;
  llvm::SetVector<const FileEntry *> Entries;
  llvm::BitVector IsSystemByUID;
  std::vector<IncludeLocation> Includes;
  SourceManager *SourceMgr = nullptr;
  std::string SysrootPath;

public:
  SourceFilesIndexDependencyCollector(IndexingContext &indexCtx,
                                      RecordingOptions recordOpts)
      : IndexCtx(indexCtx), RecordOpts(recordOpts) {}

  virtual void attachToPreprocessor(Preprocessor &PP) override {
    DependencyCollector::attachToPreprocessor(PP);
    PP.addPPCallbacks(llvm::make_unique<IncludePPCallbacks>(
        IndexCtx, RecordOpts, Includes, PP.getSourceManager()));
  }

  void setSourceManager(SourceManager *SourceMgr) {
    this->SourceMgr = SourceMgr;
  }
  void setSysrootPath(StringRef sysroot) { SysrootPath = sysroot; }

  void visitFileDependencies(
      const CompilerInstance &CI,
      llvm::function_ref<void(const FileEntry *FE, bool isSystem)> visitor)
      override {
    for (auto *FE : getEntries()) {
      visitor(FE, isSystemFile(FE));
    }
  }

  void
  visitIncludes(llvm::function_ref<void(const FileEntry *Source, unsigned Line,
                                        const FileEntry *Target)>
                    visitor) override {
    for (auto &Include : Includes) {
      visitor(Include.Source, Include.Line, Include.Target);
    }
  }

  void visitModuleImports(
      const CompilerInstance &CI,
      llvm::function_ref<void(serialization::ModuleFile &Mod, bool isSystem)>
          visitor) override {
    HeaderSearch &HS = CI.getPreprocessor().getHeaderSearchInfo();

    if (auto Reader = CI.getModuleManager()) {
      Reader->getModuleManager().visit(
          [&](serialization::ModuleFile &Mod) -> bool {
            bool isSystemMod = false;
            if (Mod.isModule()) {
              if (auto *M =
                      HS.lookupModule(Mod.ModuleName, /*AllowSearch=*/false))
                isSystemMod = M->IsSystem;
            }
            if (!isSystemMod || needSystemDependencies())
              visitor(Mod, isSystemMod);
            return true; // skip module dependencies.
          });
    }
  }

private:
  bool isSystemFile(const FileEntry *FE) {
    auto UID = FE->getUID();
    return IsSystemByUID.size() > UID && IsSystemByUID[UID];
  }

  ArrayRef<const FileEntry *> getEntries() const {
    return Entries.getArrayRef();
  }

  bool needSystemDependencies() override {
    return RecordOpts.RecordSystemDependencies;
  }

  bool sawDependency(StringRef Filename, bool FromModule, bool IsSystem,
                     bool IsModuleFile, bool IsMissing) override {
    bool sawIt = DependencyCollector::sawDependency(
        Filename, FromModule, IsSystem, IsModuleFile, IsMissing);
    if (auto *FE = SourceMgr->getFileManager().getFile(Filename)) {
      if (sawIt)
        Entries.insert(FE);
      // Record system-ness for all files that we pass through.
      if (IsSystemByUID.size() < FE->getUID() + 1)
        IsSystemByUID.resize(FE->getUID() + 1);
      IsSystemByUID[FE->getUID()] = IsSystem || isInSysroot(Filename);
    }
    return sawIt;
  }

  bool isInSysroot(StringRef Filename) {
    return !SysrootPath.empty() && Filename.startswith(SysrootPath);
  }
};

class IndexRecordActionBase {
protected:
  RecordingOptions RecordOpts;
  IndexDataRecorder Recorder;
  std::shared_ptr<IndexingContext> IndexCtx;
  SourceFilesIndexDependencyCollector DepCollector;

  IndexRecordActionBase(IndexingOptions IndexOpts, RecordingOptions recordOpts)
      : RecordOpts(std::move(recordOpts)),
        IndexCtx(new IndexingContext(IndexOpts, Recorder)),
        DepCollector(*IndexCtx, RecordOpts) {}

  std::unique_ptr<IndexASTConsumer>
  createIndexASTConsumer(CompilerInstance &CI) {
    IndexCtx->setSysrootPath(CI.getHeaderSearchOpts().Sysroot);
    Recorder.init(IndexCtx.get(), CI);

    Preprocessor &PP = CI.getPreprocessor();
    DepCollector.setSourceManager(&CI.getSourceManager());
    DepCollector.setSysrootPath(IndexCtx->getSysrootPath());
    DepCollector.attachToPreprocessor(PP);

    return llvm::make_unique<IndexASTConsumer>(CI.getPreprocessorPtr(),
                                               IndexCtx);
  }

  void finish(CompilerInstance &CI);
};

class IndexRecordAction : public ASTFrontendAction, IndexRecordActionBase {
public:
  IndexRecordAction(IndexingOptions IndexOpts, RecordingOptions RecordOpts)
      : IndexRecordActionBase(std::move(IndexOpts), std::move(RecordOpts)) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return createIndexASTConsumer(CI);
  }

  void EndSourceFileAction() override {
    FrontendAction::EndSourceFileAction();
    finish(getCompilerInstance());
  }
};

class WrappingIndexRecordAction : public WrapperFrontendAction,
                                  IndexRecordActionBase {
  bool CreatedASTConsumer = false;

public:
  WrappingIndexRecordAction(std::unique_ptr<FrontendAction> WrappedAction,
                            IndexingOptions IndexOpts,
                            RecordingOptions RecordOpts)
      : WrapperFrontendAction(std::move(WrappedAction)),
        IndexRecordActionBase(std::move(IndexOpts), std::move(RecordOpts)) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    auto OtherConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
    if (!OtherConsumer)
      return nullptr;

    CreatedASTConsumer = true;
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    Consumers.push_back(std::move(OtherConsumer));
    Consumers.push_back(createIndexASTConsumer(CI));
    return llvm::make_unique<MultiplexConsumer>(std::move(Consumers));
  }

  void EndSourceFileAction() override {
    // Invoke wrapped action's method.
    WrapperFrontendAction::EndSourceFileAction();
    if (CreatedASTConsumer)
      finish(getCompilerInstance());
  }
};

} // anonymous namespace

static std::string getClangVersion() {
  // Try picking the version from an Apple Clang tag.
  std::string RepositoryPath = getClangRepositoryPath();
  StringRef BuildNumber = StringRef(RepositoryPath);
  size_t DashOffset = BuildNumber.find('-');
  if (BuildNumber.startswith("clang") && DashOffset != StringRef::npos) {
    BuildNumber = BuildNumber.substr(DashOffset + 1);
    return BuildNumber;
  }
  // Fallback to the generic version.
  return CLANG_VERSION_STRING;
}

static void writeUnitData(const CompilerInstance &CI,
                          IndexDataRecorder &Recorder,
                          IndexDependencyProvider &DepProvider,
                          IndexingOptions IndexOpts,
                          RecordingOptions RecordOpts, StringRef OutputFile,
                          const FileEntry *RootFile, Module *UnitModule,
                          StringRef SysrootPath);

void IndexRecordActionBase::finish(CompilerInstance &CI) {
  // We may emit more diagnostics so do the begin/end source file invocations
  // on the diagnostic client.
  // FIXME: FrontendAction::EndSourceFile() should probably not call
  // CI.getDiagnosticClient().EndSourceFile()' until after it has called
  // 'EndSourceFileAction()', so that code executing during
  // EndSourceFileAction() can emit diagnostics. If this is fixed,
  // DiagClientBeginEndRAII can go away.
  struct DiagClientBeginEndRAII {
    CompilerInstance &CI;
    DiagClientBeginEndRAII(CompilerInstance &CI) : CI(CI) {
      CI.getDiagnosticClient().BeginSourceFile(CI.getLangOpts());
    }
    ~DiagClientBeginEndRAII() { CI.getDiagnosticClient().EndSourceFile(); }
  } diagClientBeginEndRAII(CI);

  SourceManager &SM = CI.getSourceManager();
  DiagnosticsEngine &Diag = CI.getDiagnostics();
  HeaderSearch &HS = CI.getPreprocessor().getHeaderSearchInfo();
  StringRef DataPath = RecordOpts.DataDirPath;

  std::string Error;
  if (IndexUnitWriter::initIndexDirectory(DataPath, Error)) {
    unsigned DiagID = Diag.getCustomDiagID(
        DiagnosticsEngine::Error, "failed creating index directory %0");
    Diag.Report(DiagID) << Error;
    return;
  }

  std::string OutputFile = CI.getFrontendOpts().OutputFile;
  if (OutputFile.empty()) {
    OutputFile = CI.getFrontendOpts().Inputs[0].getFile();
    OutputFile += ".o";
  }

  const FileEntry *RootFile = nullptr;
  Module *UnitMod = nullptr;
  bool isModuleGeneration = CI.getLangOpts().isCompilingModule();
  if (!isModuleGeneration &&
      CI.getFrontendOpts().ProgramAction != frontend::GeneratePCH) {
    RootFile = SM.getFileEntryForID(SM.getMainFileID());
  }
  if (isModuleGeneration) {
    UnitMod = HS.lookupModule(CI.getLangOpts().CurrentModule,
                              /*AllowSearch=*/false);
  }

  writeUnitData(CI, Recorder, DepCollector, IndexCtx->getIndexOpts(), RecordOpts,
                OutputFile, RootFile, UnitMod, IndexCtx->getSysrootPath());
}

/// Checks if the unit file exists for module file, if it doesn't it generates
/// index data for it.
static bool produceIndexDataForModuleFile(serialization::ModuleFile &Mod,
                                          const CompilerInstance &CI,
                                          IndexingOptions IndexOpts,
                                          RecordingOptions RecordOpts,
                                          IndexUnitWriter &ParentUnitWriter);

static void writeUnitData(const CompilerInstance &CI,
                          IndexDataRecorder &Recorder,
                          IndexDependencyProvider &DepProvider,
                          IndexingOptions IndexOpts,
                          RecordingOptions RecordOpts, StringRef OutputFile,
                          const FileEntry *RootFile, Module *UnitModule,
                          StringRef SysrootPath) {

  SourceManager &SM = CI.getSourceManager();
  DiagnosticsEngine &Diag = CI.getDiagnostics();
  HeaderSearch &HS = CI.getPreprocessor().getHeaderSearchInfo();
  StringRef DataPath = RecordOpts.DataDirPath;
  bool IsSystemUnit = UnitModule ? UnitModule->IsSystem : false;
  bool IsModuleUnit = UnitModule != nullptr;
  bool IsDebugCompilation = CI.getCodeGenOpts().OptimizationLevel == 0;
  std::string ModuleName =
      UnitModule ? UnitModule->getFullModuleName() : std::string();

  auto getModuleInfo =
      [](writer::OpaqueModule mod,
         SmallVectorImpl<char> &Scratch) -> writer::ModuleInfo {
    assert(mod);
    writer::ModuleInfo info;
    std::string fullName =
        static_cast<const Module *>(mod)->getFullModuleName();
    unsigned offset = Scratch.size();
    Scratch.append(fullName.begin(), fullName.end());
    info.Name = StringRef(Scratch.data() + offset, fullName.size());
    return info;
  };

  auto findModuleForHeader = [&](const FileEntry *FE) -> Module * {
    if (!UnitModule)
      return nullptr;
    if (auto Mod = HS.findModuleForHeader(FE).getModule())
      if (Mod->isSubModuleOf(UnitModule))
        return Mod;
    return nullptr;
  };

  IndexUnitWriter UnitWriter(
      CI.getFileManager(), DataPath, "clang", getClangVersion(), OutputFile,
      ModuleName, RootFile, IsSystemUnit, IsModuleUnit, IsDebugCompilation,
      CI.getTargetOpts().Triple, SysrootPath, getModuleInfo);

  DepProvider.visitFileDependencies(
      CI, [&](const FileEntry *FE, bool isSystemFile) {
        UnitWriter.addFileDependency(FE, isSystemFile, findModuleForHeader(FE));
      });
  DepProvider.visitIncludes(
      [&](const FileEntry *Source, unsigned Line, const FileEntry *Target) {
        UnitWriter.addInclude(Source, Line, Target);
      });
  DepProvider.visitModuleImports(CI, [&](serialization::ModuleFile &Mod,
                                         bool isSystemMod) {
    Module *UnitMod = HS.lookupModule(Mod.ModuleName, /*AllowSearch=*/false);
    UnitWriter.addASTFileDependency(Mod.File, isSystemMod, UnitMod);
    if (Mod.isModule()) {
      produceIndexDataForModuleFile(Mod, CI, IndexOpts, RecordOpts, UnitWriter);
    }
  });

  ClangIndexRecordWriter RecordWriter(CI.getASTContext(), RecordOpts);
  for (auto I = Recorder.record_begin(), E = Recorder.record_end(); I != E;
       ++I) {
    FileID FID = I->first;
    const FileIndexRecord &Rec = *I->second;
    const FileEntry *FE = SM.getFileEntryForID(FID);
    std::string RecordFile;
    std::string Error;

    if (RecordWriter.writeRecord(FE->getName(), Rec, Error, &RecordFile)) {
      unsigned DiagID = Diag.getCustomDiagID(DiagnosticsEngine::Error,
                                             "failed writing record '%0': %1");
      Diag.Report(DiagID) << RecordFile << Error;
      return;
    }
    UnitWriter.addRecordFile(RecordFile, FE, Rec.isSystem(),
                             findModuleForHeader(FE));
  }

  std::string Error;
  if (UnitWriter.write(Error)) {
    unsigned DiagID = Diag.getCustomDiagID(DiagnosticsEngine::Error,
                                           "failed writing unit data: %0");
    Diag.Report(DiagID) << Error;
    return;
  }
}

namespace {
class ModuleFileIndexDependencyCollector : public IndexDependencyProvider {
  serialization::ModuleFile &ModFile;
  RecordingOptions RecordOpts;

public:
  ModuleFileIndexDependencyCollector(serialization::ModuleFile &Mod,
                                     RecordingOptions recordOpts)
      : ModFile(Mod), RecordOpts(recordOpts) {}

  void visitFileDependencies(
      const CompilerInstance &CI,
      llvm::function_ref<void(const FileEntry *FE, bool isSystem)> visitor)
      override {
    auto Reader = CI.getModuleManager();
    Reader->visitInputFiles(
        ModFile, RecordOpts.RecordSystemDependencies,
        /*Complain=*/false,
        [&](const serialization::InputFile &IF, bool isSystem) {
          auto *FE = IF.getFile();
          if (!FE)
            return;
          // Ignore module map files, they are not as important to track as
          // source files and they may be auto-generated which would create an
          // undesirable dependency on an intermediate build byproduct.
          if (FE->getName().endswith("module.modulemap"))
            return;

          visitor(FE, isSystem);
        });
  }

  void
  visitIncludes(llvm::function_ref<void(const FileEntry *Source, unsigned Line,
                                        const FileEntry *Target)>
                    visitor) override {
    // FIXME: Module files without a preprocessing record do not have info about
    // include locations. Serialize enough data to be able to retrieve such
    // info.
  }

  void visitModuleImports(
      const CompilerInstance &CI,
      llvm::function_ref<void(serialization::ModuleFile &Mod, bool isSystem)>
          visitor) override {
    HeaderSearch &HS = CI.getPreprocessor().getHeaderSearchInfo();
    for (auto *Mod : ModFile.Imports) {
      bool isSystemMod = false;
      if (auto *M = HS.lookupModule(Mod->ModuleName, /*AllowSearch=*/false))
        isSystemMod = M->IsSystem;
      if (!isSystemMod || RecordOpts.RecordSystemDependencies)
        visitor(*Mod, isSystemMod);
    }
  }
};
} // anonymous namespace.

static void indexModule(serialization::ModuleFile &Mod,
                        const CompilerInstance &CI, IndexingOptions IndexOpts,
                        RecordingOptions RecordOpts) {
  DiagnosticsEngine &Diag = CI.getDiagnostics();
  Diag.Report(Mod.ImportLoc, diag::remark_index_producing_module_file_data)
      << Mod.FileName;

  StringRef SysrootPath = CI.getHeaderSearchOpts().Sysroot;
  HeaderSearch &HS = CI.getPreprocessor().getHeaderSearchInfo();
  Module *UnitMod = HS.lookupModule(Mod.ModuleName, /*AllowSearch=*/false);

  IndexDataRecorder Recorder;
  IndexingContext IndexCtx(IndexOpts, Recorder);

  IndexCtx.setASTContext(CI.getASTContext());
  IndexCtx.setSysrootPath(SysrootPath);
  Recorder.init(&IndexCtx, CI);

  for (const Decl *D : CI.getModuleManager()->getModuleFileLevelDecls(Mod)) {
    IndexCtx.indexTopLevelDecl(D);
  }
  Recorder.finish();

  ModuleFileIndexDependencyCollector DepCollector(Mod, RecordOpts);
  writeUnitData(CI, Recorder, DepCollector, IndexOpts, RecordOpts, Mod.FileName,
                /*RootFile=*/nullptr, UnitMod, SysrootPath);
}

static bool produceIndexDataForModuleFile(serialization::ModuleFile &Mod,
                                          const CompilerInstance &CI,
                                          IndexingOptions IndexOpts,
                                          RecordingOptions RecordOpts,
                                          IndexUnitWriter &ParentUnitWriter) {
  DiagnosticsEngine &Diag = CI.getDiagnostics();
  std::string Error;
  // We don't do timestamp check with the PCM file, on purpose. The PCM may get
  // touched for various reasons which would cause unnecessary work to emit
  // index data. User modules normally will get rebuilt and their index data
  // re-emitted, and system modules are generally stable (and they can also can
  // get rebuilt along with their index data).
  auto IsUptodateOpt =
      ParentUnitWriter.isUnitUpToDateForOutputFile(Mod.FileName, None, Error);
  if (!IsUptodateOpt.hasValue()) {
    unsigned DiagID = Diag.getCustomDiagID(DiagnosticsEngine::Error,
                                           "failed file status check: %0");
    Diag.Report(DiagID) << Error;
    return false;
  }
  if (*IsUptodateOpt)
    return false;

  indexModule(Mod, CI, IndexOpts, RecordOpts);
  return true;
}

static std::unique_ptr<FrontendAction>
createIndexDataRecordingAction(IndexingOptions IndexOpts,
                               RecordingOptions RecordOpts,
                               std::unique_ptr<FrontendAction> WrappedAction) {
  if (WrappedAction)
    return llvm::make_unique<WrappingIndexRecordAction>(
        std::move(WrappedAction), std::move(IndexOpts), std::move(RecordOpts));
  return llvm::make_unique<IndexRecordAction>(std::move(IndexOpts),
                                              std::move(RecordOpts));
}

static std::pair<IndexingOptions, RecordingOptions>
getIndexOptionsFromFrontendOptions(const FrontendOptions &FEOpts) {
  index::IndexingOptions IndexOpts;
  index::RecordingOptions RecordOpts;
  RecordOpts.DataDirPath = FEOpts.IndexStorePath;
  if (FEOpts.IndexIgnoreSystemSymbols) {
    IndexOpts.SystemSymbolFilter =
        index::IndexingOptions::SystemSymbolFilterKind::None;
  }
  RecordOpts.RecordSymbolCodeGenName = FEOpts.IndexRecordCodegenName;
  return {IndexOpts, RecordOpts};
}

std::unique_ptr<FrontendAction> index::createIndexDataRecordingAction(
    const FrontendOptions &FEOpts,
    std::unique_ptr<FrontendAction> WrappedAction) {
  index::IndexingOptions IndexOpts;
  index::RecordingOptions RecordOpts;
  std::tie(IndexOpts, RecordOpts) = getIndexOptionsFromFrontendOptions(FEOpts);
  return ::createIndexDataRecordingAction(IndexOpts, RecordOpts,
                                          std::move(WrappedAction));
}

bool index::emitIndexDataForModuleFile(const Module *Mod,
                                       const CompilerInstance &CI,
                                       IndexUnitWriter &ParentUnitWriter) {
  index::IndexingOptions IndexOpts;
  index::RecordingOptions RecordOpts;
  std::tie(IndexOpts, RecordOpts) =
      getIndexOptionsFromFrontendOptions(CI.getFrontendOpts());

  auto astReader = CI.getModuleManager();
  serialization::ModuleFile *ModFile =
      astReader->getModuleManager().lookup(Mod->getASTFile());
  assert(ModFile && "no module file loaded for module ?");
  return produceIndexDataForModuleFile(*ModFile, CI, IndexOpts, RecordOpts,
                                       ParentUnitWriter);
}
