//===--- IndexTests.cpp - Test indexing actions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace index {

struct TestSymbol {
  std::string QName;
  // FIXME: add more information.
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const TestSymbol &S) {
  return OS << S.QName;
}

namespace {
class Indexer : public IndexDataConsumer {
public:
  bool handleDeclOccurence(const Decl *D, SymbolRoleSet Roles,
                           ArrayRef<SymbolRelation>, SourceLocation,
                           ASTNodeInfo) override {
    const auto *ND = llvm::dyn_cast<NamedDecl>(D);
    if (!ND)
      return true;
    TestSymbol S;
    S.QName = ND->getQualifiedNameAsString();
    Symbols.push_back(std::move(S));
    return true;
  }

  bool handleMacroOccurence(const IdentifierInfo *Name, const MacroInfo *,
                            SymbolRoleSet, SourceLocation) override {
    TestSymbol S;
    S.QName = Name->getName();
    Symbols.push_back(std::move(S));
    return true;
  }

  std::vector<TestSymbol> Symbols;
};

class IndexAction : public ASTFrontendAction {
public:
  IndexAction(std::shared_ptr<Indexer> Index,
              IndexingOptions Opts = IndexingOptions())
      : Index(std::move(Index)), Opts(Opts) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    class Consumer : public ASTConsumer {
      std::shared_ptr<Indexer> Index;
      std::shared_ptr<Preprocessor> PP;
      IndexingOptions Opts;

    public:
      Consumer(std::shared_ptr<Indexer> Index, std::shared_ptr<Preprocessor> PP,
               IndexingOptions Opts)
          : Index(std::move(Index)), PP(std::move(PP)), Opts(Opts) {}

      void HandleTranslationUnit(ASTContext &Ctx) override {
        std::vector<Decl *> DeclsToIndex(
            Ctx.getTranslationUnitDecl()->decls().begin(),
            Ctx.getTranslationUnitDecl()->decls().end());
        indexTopLevelDecls(Ctx, *PP, DeclsToIndex, *Index, Opts);
      }
    };
    return llvm::make_unique<Consumer>(Index, CI.getPreprocessorPtr(), Opts);
  }

private:
  std::shared_ptr<Indexer> Index;
  IndexingOptions Opts;
};

using testing::Contains;
using testing::Not;
using testing::UnorderedElementsAre;

MATCHER_P(QName, Name, "") { return arg.QName == Name; }

TEST(IndexTest, Simple) {
  auto Index = std::make_shared<Indexer>();
  tooling::runToolOnCode(new IndexAction(Index), "class X {}; void f() {}");
  EXPECT_THAT(Index->Symbols, UnorderedElementsAre(QName("X"), QName("f")));
}

TEST(IndexTest, IndexPreprocessorMacros) {
  std::string Code = "#define INDEX_MAC 1";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  Opts.IndexMacrosInPreprocessor = true;
  tooling::runToolOnCode(new IndexAction(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols, Contains(QName("INDEX_MAC")));

  Opts.IndexMacrosInPreprocessor = false;
  Index->Symbols.clear();
  tooling::runToolOnCode(new IndexAction(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols, UnorderedElementsAre());
}

} // namespace
} // namespace index
} // namespace clang
