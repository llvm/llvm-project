//===- IndexingAction.cpp - Frontend index action -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexingAction.h"
#include "IndexingContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"

using namespace clang;
using namespace clang::index;

void IndexDataConsumer::_anchor() {}

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
                                              SymbolRoleSet Roles,
                                              SourceLocation Loc) {
  return true;
}

namespace {

class IndexASTConsumer : public ASTConsumer {
  std::shared_ptr<Preprocessor> PP;
  IndexingContext &IndexCtx;

public:
  IndexASTConsumer(std::shared_ptr<Preprocessor> PP, IndexingContext &IndexCtx)
      : PP(std::move(PP)), IndexCtx(IndexCtx) {}

protected:
  void Initialize(ASTContext &Context) override {
    IndexCtx.setASTContext(Context);
    IndexCtx.getDataConsumer().initialize(Context);
    IndexCtx.getDataConsumer().setPreprocessor(PP);
  }

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    return IndexCtx.indexDeclGroupRef(DG);
  }

  void HandleInterestingDecl(DeclGroupRef DG) override {
    // Ignore deserialized decls.
  }

  void HandleTopLevelDeclInObjCContainer(DeclGroupRef DG) override {
    IndexCtx.indexDeclGroupRef(DG);
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
  }
};

class IndexActionBase {
protected:
  std::shared_ptr<IndexDataConsumer> DataConsumer;
  IndexingContext IndexCtx;

  IndexActionBase(std::shared_ptr<IndexDataConsumer> dataConsumer,
                  IndexingOptions Opts)
    : DataConsumer(std::move(dataConsumer)),
      IndexCtx(Opts, *DataConsumer) {}

  std::unique_ptr<IndexASTConsumer>
  createIndexASTConsumer(CompilerInstance &CI) {
    return llvm::make_unique<IndexASTConsumer>(CI.getPreprocessorPtr(),
                                               IndexCtx);
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

  void EndSourceFileAction() override {
    FrontendAction::EndSourceFileAction();
    finish();
  }
};

class WrappingIndexAction : public WrapperFrontendAction, IndexActionBase {
  bool IndexActionFailed = false;

public:
  WrappingIndexAction(std::unique_ptr<FrontendAction> WrappedAction,
                      std::shared_ptr<IndexDataConsumer> DataConsumer,
                      IndexingOptions Opts)
    : WrapperFrontendAction(std::move(WrappedAction)),
      IndexActionBase(std::move(DataConsumer), Opts) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
  void EndSourceFileAction() override;
};

} // anonymous namespace

void WrappingIndexAction::EndSourceFileAction() {
  // Invoke wrapped action's method.
  WrapperFrontendAction::EndSourceFileAction();
  if (!IndexActionFailed)
    finish();
}

std::unique_ptr<ASTConsumer>
WrappingIndexAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  auto OtherConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
  if (!OtherConsumer) {
    IndexActionFailed = true;
    return nullptr;
  }

  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  Consumers.push_back(std::move(OtherConsumer));
  Consumers.push_back(createIndexASTConsumer(CI));
  return llvm::make_unique<MultiplexConsumer>(std::move(Consumers));
}

std::unique_ptr<FrontendAction>
index::createIndexingAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
                            IndexingOptions Opts,
                            std::unique_ptr<FrontendAction> WrappedAction) {
  if (WrappedAction)
    return llvm::make_unique<WrappingIndexAction>(std::move(WrappedAction),
                                                  std::move(DataConsumer),
                                                  Opts);
  return llvm::make_unique<IndexAction>(std::move(DataConsumer), Opts);
}


static bool topLevelDeclVisitor(void *context, const Decl *D) {
  IndexingContext &IndexCtx = *static_cast<IndexingContext*>(context);
  return IndexCtx.indexTopLevelDecl(D);
}

static void indexTranslationUnit(ASTUnit &Unit, IndexingContext &IndexCtx) {
  Unit.visitLocalTopLevelDecls(&IndexCtx, topLevelDeclVisitor);
}

void index::indexASTUnit(ASTUnit &Unit, IndexDataConsumer &DataConsumer,
                         IndexingOptions Opts) {
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Unit.getASTContext());
  DataConsumer.initialize(Unit.getASTContext());
  DataConsumer.setPreprocessor(Unit.getPreprocessorPtr());
  indexTranslationUnit(Unit, IndexCtx);
  DataConsumer.finish();
}

void index::indexTopLevelDecls(ASTContext &Ctx, ArrayRef<const Decl *> Decls,
                               IndexDataConsumer &DataConsumer,
                               IndexingOptions Opts) {
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Ctx);

  DataConsumer.initialize(Ctx);
  for (const Decl *D : Decls)
    IndexCtx.indexTopLevelDecl(D);
  DataConsumer.finish();
}

void index::indexModuleFile(serialization::ModuleFile &Mod, ASTReader &Reader,
                            IndexDataConsumer &DataConsumer,
                            IndexingOptions Opts) {
  ASTContext &Ctx = Reader.getContext();
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Ctx);
  DataConsumer.initialize(Ctx);

  for (const Decl *D : Reader.getModuleFileLevelDecls(Mod)) {
    IndexCtx.indexTopLevelDecl(D);
  }
  DataConsumer.finish();
}
