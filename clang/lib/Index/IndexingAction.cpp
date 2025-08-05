//===- IndexingAction.cpp - Frontend index action -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexingAction.h"
#include "ClangIndexRecordWriter.h"
#include "FileIndexRecord.h"
#include "IndexDataStoreUtils.h"
#include "IndexingContext.h"
#include "clang/Basic/PathRemapper.h"
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
#include <memory>

using namespace clang;
using namespace clang::index;

namespace {

class IndexPPCallbacks final : public PPCallbacks {
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

  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override {
    if (!MD.getMacroInfo()) // Ignore nonexistent macro.
      return;
    // Note: this is defined(M), not #define M
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override {
    if (!MD.getMacroInfo()) // Ignore non-existent macro.
      return;
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    if (!MD.getMacroInfo()) // Ignore nonexistent macro.
      return;
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }

  using PPCallbacks::Elifdef;
  using PPCallbacks::Elifndef;
  void Elifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) override {
    if (!MD.getMacroInfo()) // Ignore non-existent macro.
      return;
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
  void Elifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDefinition &MD) override {
    if (!MD.getMacroInfo()) // Ignore non-existent macro.
      return;
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
};

class IndexASTConsumer final : public ASTConsumer {
  std::shared_ptr<IndexDataConsumer> DataConsumer;
  std::shared_ptr<IndexingContext> IndexCtx;
  std::shared_ptr<Preprocessor> PP;
  std::function<bool(const Decl *)> ShouldSkipFunctionBody;

public:
  IndexASTConsumer(std::shared_ptr<IndexDataConsumer> DataConsumer,
                   const IndexingOptions &Opts,
                   std::shared_ptr<Preprocessor> PP,
                   std::function<bool(const Decl *)> ShouldSkipFunctionBody)
      : DataConsumer(std::move(DataConsumer)),
        IndexCtx(new IndexingContext(Opts, *this->DataConsumer)),
        PP(std::move(PP)),
        ShouldSkipFunctionBody(std::move(ShouldSkipFunctionBody)) {
    assert(this->DataConsumer != nullptr);
    assert(this->PP != nullptr);
  }

protected:
  void Initialize(ASTContext &Context) override {
    IndexCtx->setASTContext(Context);
    IndexCtx->getDataConsumer().initialize(Context);
    IndexCtx->getDataConsumer().setPreprocessor(PP);
    PP->addPPCallbacks(std::make_unique<IndexPPCallbacks>(IndexCtx));
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

  void HandleTranslationUnit(ASTContext &Ctx) override {
    DataConsumer->finish();
  }

  bool shouldSkipFunctionBody(Decl *D) override {
    return ShouldSkipFunctionBody(D);
  }
};

class IndexAction final : public ASTFrontendAction {
  std::shared_ptr<IndexDataConsumer> DataConsumer;
  IndexingOptions Opts;

public:
  IndexAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
              const IndexingOptions &Opts)
      : DataConsumer(std::move(DataConsumer)), Opts(Opts) {
    assert(this->DataConsumer != nullptr);
  }

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<IndexASTConsumer>(
        DataConsumer, Opts, CI.getPreprocessorPtr(),
        /*ShouldSkipFunctionBody=*/[](const Decl *) { return false; });
  }
};

} // anonymous namespace

std::unique_ptr<ASTConsumer> index::createIndexingASTConsumer(
    std::shared_ptr<IndexDataConsumer> DataConsumer,
    const IndexingOptions &Opts, std::shared_ptr<Preprocessor> PP,
    std::function<bool(const Decl *)> ShouldSkipFunctionBody) {
  return std::make_unique<IndexASTConsumer>(DataConsumer, Opts, PP,
                                            ShouldSkipFunctionBody);
}

std::unique_ptr<ASTConsumer> clang::index::createIndexingASTConsumer(
    std::shared_ptr<IndexDataConsumer> DataConsumer,
    const IndexingOptions &Opts, std::shared_ptr<Preprocessor> PP) {
  std::function<bool(const Decl *)> ShouldSkipFunctionBody = [](const Decl *) {
    return false;
  };
  if (Opts.ShouldTraverseDecl)
    ShouldSkipFunctionBody =
        [ShouldTraverseDecl(Opts.ShouldTraverseDecl)](const Decl *D) {
          return !ShouldTraverseDecl(D);
        };
  return createIndexingASTConsumer(std::move(DataConsumer), Opts, std::move(PP),
                                   std::move(ShouldSkipFunctionBody));
}

std::unique_ptr<FrontendAction>
index::createIndexingAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
                            const IndexingOptions &Opts) {
  assert(DataConsumer != nullptr);
  return std::make_unique<IndexAction>(std::move(DataConsumer), Opts);
}

static bool topLevelDeclVisitor(void *context, const Decl *D) {
  IndexingContext &IndexCtx = *static_cast<IndexingContext *>(context);
  return IndexCtx.indexTopLevelDecl(D);
}

static void indexTranslationUnit(ASTUnit &Unit, IndexingContext &IndexCtx) {
  Unit.visitLocalTopLevelDecls(&IndexCtx, topLevelDeclVisitor);
}

static void indexPreprocessorMacro(const IdentifierInfo *II,
                                   const MacroInfo *MI,
                                   MacroDirective::Kind DirectiveKind,
                                   SourceLocation Loc,
                                   IndexDataConsumer &DataConsumer) {
  // When using modules, it may happen that we find #undef of a macro that
  // was defined in another module. In such case, MI may be nullptr, since
  // we only look for macro definitions in the current TU. In that case,
  // there is nothing to index.
  if (!MI)
    return;

  // Skip implicit visibility change.
  if (DirectiveKind == MacroDirective::MD_Visibility)
    return;

  auto Role = DirectiveKind == MacroDirective::MD_Define
                  ? SymbolRole::Definition
                  : SymbolRole::Undefinition;
  DataConsumer.handleMacroOccurrence(II, MI, static_cast<unsigned>(Role), Loc);
}

static void indexPreprocessorMacros(Preprocessor &PP,
                                    IndexDataConsumer &DataConsumer) {
  for (const auto &M : PP.macros()) {
    for (auto *MD = M.second.getLatest(); MD; MD = MD->getPrevious()) {
      indexPreprocessorMacro(M.first, MD->getMacroInfo(), MD->getKind(),
                             MD->getLocation(), DataConsumer);
    }
  }
}

static void indexPreprocessorModuleMacros(Preprocessor &PP,
                                          serialization::ModuleFile &Mod,
                                          IndexDataConsumer &DataConsumer) {
  for (const auto &M : PP.macros()) {
    if (M.second.getLatest() == nullptr) {
      for (auto *MM : PP.getLeafModuleMacros(M.first)) {
        auto *OwningMod = MM->getOwningModule();
        if (OwningMod && OwningMod->getASTFile() == Mod.File) {
          if (auto *MI = MM->getMacroInfo()) {
            indexPreprocessorMacro(M.first, MI, MacroDirective::MD_Define,
                                   MI->getDefinitionLoc(), DataConsumer);
          }
        }
      }
    }
  }
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
  return std::make_unique<IndexPPCallbacks>(
      std::make_shared<IndexingContext>(Opts, Consumer));
}

void index::indexModuleFile(serialization::ModuleFile &Mod, ASTReader &Reader,
                            IndexDataConsumer &DataConsumer,
                            IndexingOptions Opts) {
  ASTContext &Ctx = Reader.getContext();
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Ctx);
  DataConsumer.initialize(Ctx);

  if (Opts.IndexMacrosInPreprocessor) {
    indexPreprocessorModuleMacros(Reader.getPreprocessor(), Mod, DataConsumer);
  }

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
  bool handleDeclOccurrence(const Decl *D, SymbolRoleSet Roles,
                            ArrayRef<SymbolRelation> Relations,
                            SourceLocation Loc, ASTNodeInfo ASTNode) override {
    FileID FID;
    unsigned Offset;
    if (!getFileIDAndOffset(Loc, FID, Offset))
      return true;

    FileIndexRecord &Rec = getFileIndexRecord(FID);
    Rec.addDeclOccurence(Roles, Offset, D, Relations);
    return true;
  }

  bool handleMacroOccurrence(const IdentifierInfo *Name, const MacroInfo *MI,
                             SymbolRoleSet Roles, SourceLocation Loc) override {
    FileID FID;
    unsigned Offset;
    if (!getFileIDAndOffset(Loc, FID, Offset))
      return true;

    FileIndexRecord &Rec = getFileIndexRecord(FID);
    Rec.addMacroOccurence(Roles, Offset, Name, MI);
    return true;
  }

  FileIndexRecord &getFileIndexRecord(FileID FID) {
    auto &Entry = RecordByFile[FID];
    if (!Entry) {
      Entry.reset(new FileIndexRecord(FID, IndexCtx->isSystemFile(FID)));
    }
    return *Entry;
  }

  bool getFileIDAndOffset(SourceLocation Loc, FileID &FID, unsigned &Offset) {
    SourceManager &SM = PP->getSourceManager();
    Loc = SM.getFileLoc(Loc);
    if (Loc.isInvalid())
      return false;

    std::tie(FID, Offset) = SM.getDecomposedLoc(Loc);

    if (FID.isInvalid())
      return false;

    // Ignore the predefines buffer.
    const FileEntry *FE = PP->getSourceManager().getFileEntryForID(FID);
    return FE != nullptr;
  }

public:
  void finish() override {
    if (IndexCtx->getIndexOpts().IndexMacros) {
      SmallVector<FileID, 8> ToRemove;
      for (auto &pair : RecordByFile) {
        pair.second->removeHeaderGuardMacros();
        // Remove now-empty records.
        if (pair.second->getDeclOccurrencesSortedByOffset().empty())
          ToRemove.push_back(pair.first);
      }
      for (auto FID : ToRemove) {
        RecordByFile.erase(FID);
      }
    }
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

  virtual void
  InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                     StringRef FileName, bool IsAngled,
                     CharSourceRange FilenameRange, OptionalFileEntryRef File,
                     StringRef SearchPath, StringRef RelativePath,
                     const Module *SuggestedModule, bool ModuleImported,
                     SrcMgr::CharacteristicKind FileType) override {
    if (HashLoc.isFileID() && File)
      addInclude(HashLoc, *File);
  }
};

class IndexDependencyProvider {
public:
  virtual ~IndexDependencyProvider() {}

  virtual void visitFileDependencies(
      const CompilerInstance &CI,
      llvm::function_ref<void(FileEntryRef FE, bool isSystem)> visitor) = 0;
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
  llvm::SetVector<FileEntryRef> Entries;
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
    PP.addPPCallbacks(std::make_unique<IncludePPCallbacks>(
        IndexCtx, RecordOpts, Includes, PP.getSourceManager()));
  }

  void setSourceManager(SourceManager *SourceMgr) {
    this->SourceMgr = SourceMgr;
  }
  void setSysrootPath(StringRef sysroot) { SysrootPath = std::string(sysroot); }

  void visitFileDependencies(
      const CompilerInstance &CI,
      llvm::function_ref<void(FileEntryRef FE, bool isSystem)> visitor)
      override {
    for (FileEntryRef FE : getEntries()) {
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

    if (auto Reader = CI.getASTReader()) {
      Reader->getModuleManager().visit(
          [&](serialization::ModuleFile &Mod) -> bool {
            bool isSystemMod = false;
            if (Mod.isModule()) {
              if (auto *M = HS.lookupModule(Mod.ModuleName, SourceLocation(),
                                            /*AllowSearch=*/false))
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

  ArrayRef<FileEntryRef> getEntries() const {
    return Entries.getArrayRef();
  }

  bool needSystemDependencies() override {
    return RecordOpts.RecordSystemDependencies;
  }

  bool sawDependency(StringRef Filename, bool FromModule, bool IsSystem,
                     bool IsModuleFile, bool IsMissing) override {
    bool sawIt = DependencyCollector::sawDependency(
        Filename, FromModule, IsSystem, IsModuleFile, IsMissing);
    if (auto FE = SourceMgr->getFileManager().getOptionalFileRef(Filename)) {
      if (sawIt)
        Entries.insert(*FE);
      // Record system-ness for all files that we pass through.
      if (IsSystemByUID.size() < FE->getUID() + 1)
        IsSystemByUID.resize(FE->getUID() + 1);
      IsSystemByUID[FE->getUID()] = IsSystem || isInSysroot(Filename);
    }
    return sawIt;
  }

  bool isInSysroot(StringRef Filename) {
    return !SysrootPath.empty() && Filename.starts_with(SysrootPath);
  }
};


class IndexRecordASTConsumer : public ASTConsumer {
  std::shared_ptr<Preprocessor> PP;
  std::shared_ptr<IndexingContext> IndexCtx;

public:
  IndexRecordASTConsumer(std::shared_ptr<Preprocessor> PP,
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

  std::unique_ptr<IndexRecordASTConsumer>
  createIndexASTConsumer(CompilerInstance &CI) {
    IndexCtx->setSysrootPath(CI.getHeaderSearchOpts().Sysroot);
    Recorder.init(IndexCtx.get(), CI);

    Preprocessor &PP = CI.getPreprocessor();
    DepCollector.setSourceManager(&CI.getSourceManager());
    DepCollector.setSysrootPath(IndexCtx->getSysrootPath());
    DepCollector.attachToPreprocessor(PP);

    if (IndexCtx->getIndexOpts().IndexMacros)
      PP.addPPCallbacks(std::make_unique<IndexPPCallbacks>(IndexCtx));

    return std::make_unique<IndexRecordASTConsumer>(CI.getPreprocessorPtr(),
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

    if (CI.getFrontendOpts().IndexStorePath.empty()) {
      // We are not generating an index store. Nothing to do.
      return OtherConsumer;
    }

    CreatedASTConsumer = true;
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    if (OtherConsumer) {
      Consumers.push_back(std::move(OtherConsumer));
    }
    Consumers.push_back(createIndexASTConsumer(CI));
    return std::make_unique<MultiplexConsumer>(std::move(Consumers));
  }

  void EndSourceFile() override {
    FrontendAction::EndSourceFile();
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
  if (BuildNumber.starts_with("clang") && DashOffset != StringRef::npos) {
    BuildNumber = BuildNumber.substr(DashOffset + 1);
    return std::string(BuildNumber);
  }
  // Fallback to the generic version.
  return CLANG_VERSION_STRING;
}

static void writeUnitData(const CompilerInstance &CI,
                          IndexDataRecorder &Recorder,
                          IndexDependencyProvider &DepProvider,
                          IndexingOptions IndexOpts,
                          RecordingOptions RecordOpts, StringRef OutputFile,
                          OptionalFileEntryRef RootFile, Module *UnitModule,
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

  Recorder.finish();

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

  std::string OutputFile = CI.getFrontendOpts().IndexUnitOutputPath;
  if (OutputFile.empty())
    OutputFile = CI.getFrontendOpts().OutputFile;
  if (OutputFile.empty()) {
    OutputFile = std::string(CI.getFrontendOpts().Inputs[0].getFile());
    OutputFile += ".o";
  }

  OptionalFileEntryRef RootFile;
  Module *UnitMod = nullptr;
  bool isModuleGeneration = CI.getLangOpts().isCompilingModule();
  if (!isModuleGeneration &&
      CI.getFrontendOpts().ProgramAction != frontend::GeneratePCH) {
    RootFile = SM.getFileEntryRefForID(SM.getMainFileID());
  }
  if (isModuleGeneration) {
    UnitMod = HS.lookupModule(CI.getLangOpts().CurrentModule, SourceLocation(),
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
                          OptionalFileEntryRef RootFile, Module *UnitModule,
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

  auto findModuleForHeader = [&](FileEntryRef FE) -> Module * {
    if (!UnitModule)
      return nullptr;
    if (Module *Mod = HS.findModuleForHeader(FE).getModule())
      if (Mod->isSubModuleOf(UnitModule))
        return Mod;
    return nullptr;
  };
  PathRemapper Remapper;
  auto &PrefixMap = CI.getCodeGenOpts().DebugPrefixMap;
  // We need to add in reverse order since the `DebugPrefixMap` currently sorts
  // ascending instead of descending, but we want `foo/subpath/` to come before
  // `foo/`.
  for (auto It = PrefixMap.rbegin(); It != PrefixMap.rend(); ++It)
    Remapper.addMapping(It->first, It->second);

  IndexUnitWriter UnitWriter(
      CI.getFileManager(), DataPath, "clang", getClangVersion(),
      CI.getFrontendOpts().IndexStoreCompress, OutputFile, ModuleName, RootFile,
      IsSystemUnit, IsModuleUnit, IsDebugCompilation, CI.getTargetOpts().Triple,
      SysrootPath, Remapper, getModuleInfo);

  DepProvider.visitFileDependencies(
      CI, [&](FileEntryRef FE, bool isSystemFile) {
        UnitWriter.addFileDependency(FE, isSystemFile, findModuleForHeader(FE));
      });
  DepProvider.visitIncludes(
      [&](const FileEntry *Source, unsigned Line, const FileEntry *Target) {
        UnitWriter.addInclude(Source, Line, Target);
      });
  bool IndexPcms = IndexOpts.IndexPcms;
  bool WithoutUnitName = !IndexPcms;
  DepProvider.visitModuleImports(CI, [&](serialization::ModuleFile &Mod,
                                         bool isSystemMod) {
    Module *UnitMod = HS.lookupModule(Mod.ModuleName, Mod.ImportLoc,
                                      /*AllowSearch=*/false);
    UnitWriter.addASTFileDependency(Mod.File, isSystemMod, UnitMod,
                                    WithoutUnitName);
    if (Mod.isModule() && IndexPcms) {
      produceIndexDataForModuleFile(Mod, CI, IndexOpts, RecordOpts, UnitWriter);
    }
  });

  ClangIndexRecordWriter RecordWriter(
      CI.getASTContext(), CI.getFrontendOpts().IndexStoreCompress, RecordOpts);
  for (auto I = Recorder.record_begin(), E = Recorder.record_end(); I != E;
       ++I) {
    FileID FID = I->first;
    const FileIndexRecord &Rec = *I->second;
    OptionalFileEntryRef FE = SM.getFileEntryRefForID(FID);
    std::string RecordFile;
    std::string Error;

    if (RecordWriter.writeRecord(FE->getName(), Rec, Error, &RecordFile)) {
      unsigned DiagID = Diag.getCustomDiagID(DiagnosticsEngine::Error,
                                             "failed writing record '%0': %1");
      Diag.Report(DiagID) << RecordFile << Error;
      return;
    }
    UnitWriter.addRecordFile(RecordFile, *FE, Rec.isSystem(),
                             findModuleForHeader(*FE));
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
      llvm::function_ref<void(FileEntryRef FE, bool isSystem)> visitor)
      override {
    auto Reader = CI.getASTReader();
    Reader->visitInputFiles(
        ModFile, RecordOpts.RecordSystemDependencies,
        /*Complain=*/false,
        [&](const serialization::InputFile &IF, bool isSystem) {
          auto FE = IF.getFile();
          if (!FE)
            return;
          // Ignore module map files, they are not as important to track as
          // source files and they may be auto-generated which would create an
          // undesirable dependency on an intermediate build byproduct.
          if (FE->getName().ends_with("module.modulemap"))
            return;
          // Ignore SDKSettings.json, they are not important to track for
          // indexing.
          if (FE->getName().ends_with("SDKSettings.json"))
            return;

          visitor(*FE, isSystem);
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
      if (auto *M = HS.lookupModule(Mod->ModuleName, Mod->ImportLoc,
                                    /*AllowSearch=*/false))
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
  Module *UnitMod =
      HS.lookupModule(Mod.ModuleName, Mod.ImportLoc, /*AllowSearch=*/false);

  IndexDataRecorder Recorder;
  IndexingContext IndexCtx(IndexOpts, Recorder);

  IndexCtx.setASTContext(CI.getASTContext());
  IndexCtx.setSysrootPath(SysrootPath);
  Recorder.init(&IndexCtx, CI);

  for (const Decl *D : CI.getASTReader()->getModuleFileLevelDecls(Mod)) {
    IndexCtx.indexTopLevelDecl(D);
  }
  if (IndexOpts.IndexMacrosInPreprocessor) {
    indexPreprocessorModuleMacros(CI.getPreprocessor(), Mod, Recorder);
  }
  Recorder.finish();

  ModuleFileIndexDependencyCollector DepCollector(Mod, RecordOpts);
  writeUnitData(CI, Recorder, DepCollector, IndexOpts, RecordOpts, Mod.FileName,
                /*RootFile=*/std::nullopt, UnitMod, SysrootPath);
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
  auto IsUptodateOpt = ParentUnitWriter.isUnitUpToDateForOutputFile(
      Mod.FileName, std::nullopt, Error);
  if (!IsUptodateOpt) {
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
    return std::make_unique<WrappingIndexRecordAction>(
        std::move(WrappedAction), std::move(IndexOpts), std::move(RecordOpts));
  return std::make_unique<IndexRecordAction>(std::move(IndexOpts),
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
  IndexOpts.IndexMacros = !FEOpts.IndexIgnoreMacros;
  IndexOpts.IndexMacrosInPreprocessor = !FEOpts.IndexIgnoreMacros;
  IndexOpts.IndexPcms = !FEOpts.IndexIgnorePcms;
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

  auto astReader = CI.getASTReader();
  serialization::ModuleFile *ModFile =
      astReader->getModuleManager().lookup(*Mod->getASTFile());
  assert(ModFile && "no module file loaded for module ?");
  return produceIndexDataForModuleFile(*ModFile, CI, IndexOpts, RecordOpts,
                                       ParentUnitWriter);
}
