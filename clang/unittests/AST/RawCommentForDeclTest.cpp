//===- unittests/AST/RawCommentForDeclTestTest.cpp
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

namespace clang {

struct FoundComment {
  std::string DeclName;
  bool IsDefinition;
  std::string Comment;

  bool operator==(const FoundComment &RHS) const {
    return DeclName == RHS.DeclName && IsDefinition == RHS.IsDefinition &&
           Comment == RHS.Comment;
  }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &Stream,
                                       const FoundComment &C) {
    return Stream << "{Name: " << C.DeclName << ", Def: " << C.IsDefinition
                  << ", Comment: " << C.Comment << "}";
  }
};

class CollectCommentsAction : public ASTFrontendAction {
public:
  CollectCommentsAction(std::vector<FoundComment> &Comments)
      : Comments(Comments) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    CI.getLangOpts().CommentOpts.ParseAllComments = true;
    return std::make_unique<Consumer>(*this);
  }

  std::vector<FoundComment> &Comments;

private:
  class Consumer : public clang::ASTConsumer {
  private:
    CollectCommentsAction &Action;

  public:
    Consumer(CollectCommentsAction &Action) : Action(Action) {}

    bool HandleTopLevelDecl(DeclGroupRef DG) override {
      for (Decl *D : DG) {
        if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
          auto &Ctx = D->getASTContext();
          const auto *RC = Ctx.getRawCommentForAnyRedecl(D);
          Action.Comments.push_back(FoundComment{
              ND->getNameAsString(), IsDefinition(D),
              RC ? RC->getRawText(Ctx.getSourceManager()).str() : ""});
        }
      }

      return true;
    }

    static bool IsDefinition(const Decl *D) {
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        return FD->isThisDeclarationADefinition();
      }
      if (const TagDecl *TD = dyn_cast<TagDecl>(D)) {
        return TD->isThisDeclarationADefinition();
      }
      return false;
    }
  };
};

TEST(RawCommentForDecl, DefinitionComment) {
  std::vector<FoundComment> Comments;
  auto Action = std::make_unique<CollectCommentsAction>(Comments);
  ASSERT_TRUE(tooling::runToolOnCode(std::move(Action), R"cpp(
    void f();

    // f is the best
    void f() {}
  )cpp"));
  EXPECT_THAT(Comments, testing::ElementsAre(
                            FoundComment{"f", false, ""},
                            FoundComment{"f", true, "// f is the best"}));
}

} // namespace clang
