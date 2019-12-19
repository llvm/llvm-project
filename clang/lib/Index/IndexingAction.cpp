//===- IndexingAction.cpp - Frontend index action -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexingAction.h"
#include "IndexingContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/STLExtras.h"
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

std::unique_ptr<FrontendAction>
index::createIndexingAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
                            const IndexingOptions &Opts) {
  assert(DataConsumer != nullptr);
  return std::make_unique<IndexAction>(std::move(DataConsumer), Opts);
}

static bool topLevelDeclVisitor(void *context, const Decl *D) {
  IndexingContext &IndexCtx = *static_cast<IndexingContext*>(context);
  return IndexCtx.indexTopLevelDecl(D);
}

static void indexTranslationUnit(ASTUnit &Unit, IndexingContext &IndexCtx) {
  Unit.visitLocalTopLevelDecls(&IndexCtx, topLevelDeclVisitor);
}

static void indexPreprocessorMacros(const Preprocessor &PP,
                                    IndexDataConsumer &DataConsumer) {
  for (const auto &M : PP.macros())
    if (MacroDirective *MD = M.second.getLatest())
      DataConsumer.handleMacroOccurrence(
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

  if (Opts.IndexMacrosInPreprocessor)
    indexPreprocessorMacros(Reader.getPreprocessor(), DataConsumer);

  for (const Decl *D : Reader.getModuleFileLevelDecls(Mod)) {
    IndexCtx.indexTopLevelDecl(D);
  }
  DataConsumer.finish();
}
