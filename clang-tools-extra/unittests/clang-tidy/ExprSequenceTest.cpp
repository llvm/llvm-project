//===---- UsingInserterTest.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/ExprSequence.h"

#include "ClangTidyTest.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace utils {

// Checks that expression `before` is sequenced before `after`.
// Check sthat expression `unseq1` is not sequenced before or sequenced
// after `unseq2`.
class ExprSequenceCheck : public clang::tidy::ClangTidyCheck {
public:
  ExprSequenceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(clang::ast_matchers::MatchFinder *Finder) override {
    using namespace clang::ast_matchers;
    const auto RefTo = [](StringRef name) {
      return declRefExpr(to(varDecl(hasName(name)))).bind(name);
    };
    Finder->addMatcher(functionDecl(hasDescendant(RefTo("before")),
                                    hasDescendant(RefTo("after")))
                           .bind("fn"),
                       this);
    Finder->addMatcher(functionDecl(hasDescendant(RefTo("unseq1")),
                                    hasDescendant(RefTo("unseq2")))
                           .bind("fn"),
                       this);
  }
  void
  check(const clang::ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Fn = Result.Nodes.getNodeAs<clang::FunctionDecl>("fn");
    const auto *Before = Result.Nodes.getNodeAs<clang::Expr>("before");
    const auto *After = Result.Nodes.getNodeAs<clang::Expr>("after");
    const auto *Unseq1 = Result.Nodes.getNodeAs<clang::Expr>("unseq1");
    const auto *Unseq2 = Result.Nodes.getNodeAs<clang::Expr>("unseq2");

    CFG::BuildOptions Options;
    Options.AddImplicitDtors = true;
    Options.AddTemporaryDtors = true;
    std::unique_ptr<CFG> TheCFG =
        CFG::buildCFG(nullptr, Fn->getBody(), Result.Context, Options);
    ASSERT_TRUE(TheCFG != nullptr);

    ExprSequence Seq(TheCFG.get(), Fn->getBody(), Result.Context);

    if (Before && !Seq.inSequence(Before, After)) {
      diag(Before->getBeginLoc(),
           "[seq]expected 'before' to be sequenced before 'after'");
    }
    if (Unseq1 &&
        (Seq.inSequence(Unseq1, Unseq2) || Seq.inSequence(Unseq2, Unseq1))) {
      diag(Before->getBeginLoc(),
           "[seq]expected 'unseq1' and 'unseq2' to not be sequenced");
    }
  }
};

void runChecker(StringRef Code) {
  std::vector<ClangTidyError> Errors;

  std::string result = test::runCheckOnCode<ExprSequenceCheck>(
      Code, &Errors, "foo.cc", {}, ClangTidyOptions());

  bool HasSeqError = false;
  for (const ClangTidyError &E : Errors) {
    // Ignore non-seq warnings.
    StringRef Msg(E.Message.Message);
    if (!Msg.consume_front("[seq]"))
      continue;
    llvm::errs() << Msg << "\nIn code:\n" << Code << "\n";
    HasSeqError = true;
  }
  EXPECT_FALSE(HasSeqError);
}

TEST(ExprSequenceTest, Unseq) {
  runChecker("int f(int unseq1, int unseq2) { return unseq1 + unseq2; }");
}

TEST(ExprSequenceTest, Comma) {
  runChecker("int f(int before, int after) { return before, after; }");
}

TEST(ExprSequenceTest, Ctor) {
  runChecker("struct S { S(int, int, int); };"
             "S f(int before, int after) { return S{before, 3, after}; }");
  runChecker("struct S { S(int, int, int); };"
             "S f(int unseq1, int unseq2) { return S(unseq1, 3, unseq2); }");
}

TEST(ExprSequenceTest, ConditionalOperator) {
  runChecker("int f(bool before, int after) { return before ? after : 3; }");
  runChecker("int f(bool before, int after) { return before ? 3 : after; }");
  runChecker(
      "int f(bool c, int unseq1, int unseq2) { return c ? unseq1 : unseq2; }");
}

} // namespace utils
} // namespace tidy
} // namespace clang
