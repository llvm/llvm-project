//===- IndexingAction.cpp - Frontend index action -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexDataConsumer.h"
#include "IndexingContext.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;
using namespace clang::index;

void IndexDataConsumer::_anchor() {}

bool IndexDataConsumer::handleDeclOccurence(const Decl *D, SymbolRoleSet Roles,
                                            ArrayRef<SymbolRelation> Relations,
                                            FileID FID, unsigned Offset,
                                            ASTNodeInfo ASTNode) {
  return true;
}

bool IndexDataConsumer::handleMacroOccurence(const IdentifierInfo *Name,
                                             const MacroInfo *MI, SymbolRoleSet Roles,
                                             FileID FID, unsigned Offset) {
  return true;
}

bool IndexDataConsumer::handleModuleOccurence(const ImportDecl *ImportD,
                                              SymbolRoleSet Roles,
                                              FileID FID, unsigned Offset) {
  return true;
}

namespace {

class IndexASTConsumer : public ASTConsumer {
  IndexingContext &IndexCtx;

public:
  IndexASTConsumer(IndexingContext &IndexCtx)
    : IndexCtx(IndexCtx) {}

protected:
  void Initialize(ASTContext &Context) override {
    IndexCtx.setASTContext(Context);
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

class IndexAction : public WrapperFrontendAction {
  IndexingOptions IndexOpts;
  std::shared_ptr<IndexDataConsumer> DataConsumer;
  std::unique_ptr<IndexingContext> IndexCtx;

public:
  IndexAction(std::unique_ptr<FrontendAction> WrappedAction,
              std::shared_ptr<IndexDataConsumer> DataConsumer,
              IndexingOptions Opts)
    : WrapperFrontendAction(std::move(WrappedAction)),
      IndexOpts(Opts),
      DataConsumer(std::move(DataConsumer)) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
  void EndSourceFileAction() override;
};

} // anonymous namespace

void IndexAction::EndSourceFileAction() {
  // Invoke wrapped action's method.
  WrapperFrontendAction::EndSourceFileAction();

  bool IndexActionFailed = !IndexCtx;
  if (!IndexActionFailed)
    DataConsumer->finish();
}

std::unique_ptr<ASTConsumer>
IndexAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  auto OtherConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
  if (!OtherConsumer)
    return nullptr;

  IndexCtx.reset(new IndexingContext(IndexOpts, *DataConsumer));

  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  Consumers.push_back(std::move(OtherConsumer));
  Consumers.push_back(llvm::make_unique<IndexASTConsumer>(*IndexCtx));
  return llvm::make_unique<MultiplexConsumer>(std::move(Consumers));
}

std::unique_ptr<FrontendAction>
index::createIndexingAction(std::unique_ptr<FrontendAction> WrappedAction,
                            std::shared_ptr<IndexDataConsumer> DataConsumer,
                            IndexingOptions Opts) {
  return llvm::make_unique<IndexAction>(std::move(WrappedAction),
                                        std::move(DataConsumer),
                                        Opts);
}


static bool topLevelDeclVisitor(void *context, const Decl *D) {
  IndexingContext &IndexCtx = *static_cast<IndexingContext*>(context);
  return IndexCtx.indexTopLevelDecl(D);
}

static void indexTranslationUnit(ASTUnit &Unit, IndexingContext &IndexCtx) {
  Unit.visitLocalTopLevelDecls(&IndexCtx, topLevelDeclVisitor);
}

void index::indexASTUnit(ASTUnit &Unit,
                         std::shared_ptr<IndexDataConsumer> DataConsumer,
                         IndexingOptions Opts) {
  IndexingContext IndexCtx(Opts, *DataConsumer);
  IndexCtx.setASTContext(Unit.getASTContext());
  indexTranslationUnit(Unit, IndexCtx);
}
