//===---- OverlappingReplacementsTest.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyTest.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {
namespace {

const char BoundDecl[] = "decl";
const char BoundIf[] = "if";

// We define a reduced set of very small checks that allow to test different
// overlapping situations (no overlapping, replacements partially overlap, etc),
// as well as different kinds of diagnostics (one check produces several errors,
// several replacement ranges in an error, etc).
class UseCharCheck : public ClangTidyCheck {
public:
  UseCharCheck(StringRef CheckName, ClangTidyContext *Context)
      : ClangTidyCheck(CheckName, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    using namespace ast_matchers;
    Finder->addMatcher(varDecl(hasType(isInteger())).bind(BoundDecl), this);
  }
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    auto *VD = Result.Nodes.getNodeAs<VarDecl>(BoundDecl);
    diag(VD->getBeginLoc(), "use char") << FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(VD->getBeginLoc(), VD->getBeginLoc()),
        "char");
  }
};

class IfFalseCheck : public ClangTidyCheck {
public:
  IfFalseCheck(StringRef CheckName, ClangTidyContext *Context)
      : ClangTidyCheck(CheckName, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    using namespace ast_matchers;
    Finder->addMatcher(ifStmt().bind(BoundIf), this);
  }
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    auto *If = Result.Nodes.getNodeAs<IfStmt>(BoundIf);
    auto *Cond = If->getCond();
    SourceRange Range = Cond->getSourceRange();
    if (auto *D = If->getConditionVariable()) {
      Range = SourceRange(D->getBeginLoc(), D->getEndLoc());
    }
    diag(Range.getBegin(), "the cake is a lie") << FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(Range), "false");
  }
};

class RefactorCheck : public ClangTidyCheck {
public:
  RefactorCheck(StringRef CheckName, ClangTidyContext *Context)
      : ClangTidyCheck(CheckName, Context), NamePattern("::$") {}
  RefactorCheck(StringRef CheckName, ClangTidyContext *Context,
                StringRef NamePattern)
      : ClangTidyCheck(CheckName, Context), NamePattern(NamePattern) {}
  virtual std::string newName(StringRef OldName) = 0;

  void registerMatchers(ast_matchers::MatchFinder *Finder) final {
    using namespace ast_matchers;
    Finder->addMatcher(varDecl(matchesName(NamePattern)).bind(BoundDecl), this);
  }

  void check(const ast_matchers::MatchFinder::MatchResult &Result) final {
    auto *VD = Result.Nodes.getNodeAs<VarDecl>(BoundDecl);
    std::string NewName = newName(VD->getName());

    auto Diag = diag(VD->getLocation(), "refactor %0 into %1")
                << VD->getName() << NewName
                << FixItHint::CreateReplacement(
                       CharSourceRange::getTokenRange(VD->getLocation(),
                                                      VD->getLocation()),
                       NewName);

    class UsageVisitor : public RecursiveASTVisitor<UsageVisitor> {
    public:
      UsageVisitor(const ValueDecl *VD, StringRef NewName,
                   DiagnosticBuilder &Diag)
          : VD(VD), NewName(NewName), Diag(Diag) {}
      bool VisitDeclRefExpr(DeclRefExpr *E) {
        if (const ValueDecl *D = E->getDecl()) {
          if (VD->getCanonicalDecl() == D->getCanonicalDecl()) {
            Diag << FixItHint::CreateReplacement(
                CharSourceRange::getTokenRange(E->getSourceRange()), NewName);
          }
        }
        return RecursiveASTVisitor<UsageVisitor>::VisitDeclRefExpr(E);
      }

    private:
      const ValueDecl *VD;
      StringRef NewName;
      DiagnosticBuilder &Diag;
    };

    UsageVisitor(VD, NewName, Diag)
        .TraverseDecl(Result.Context->getTranslationUnitDecl());
  }

protected:
  const std::string NamePattern;
};

class StartsWithPotaCheck : public RefactorCheck {
public:
  StartsWithPotaCheck(StringRef CheckName, ClangTidyContext *Context)
      : RefactorCheck(CheckName, Context, "::pota") {}

  std::string newName(StringRef OldName) override {
    return "toma" + OldName.substr(4).str();
  }
};

class EndsWithTatoCheck : public RefactorCheck {
public:
  EndsWithTatoCheck(StringRef CheckName, ClangTidyContext *Context)
      : RefactorCheck(CheckName, Context, "tato$") {}

  std::string newName(StringRef OldName) override {
    return OldName.substr(0, OldName.size() - 4).str() + "melo";
  }
};

} // namespace

TEST(OverlappingReplacementsTest, UseCharCheckTest) {
  const char Code[] =
      R"(void f() {
  int a = 0;
  if (int b = 0) {
    int c = a;
  }
})";

  const char CharFix[] =
      R"(void f() {
  char a = 0;
  if (char b = 0) {
    char c = a;
  }
})";
  EXPECT_EQ(CharFix, runCheckOnCode<UseCharCheck>(Code));
}

TEST(OverlappingReplacementsTest, IfFalseCheckTest) {
  const char Code[] =
      R"(void f() {
  int potato = 0;
  if (int b = 0) {
    int c = potato;
  } else if (true) {
    int d = 0;
  }
})";

  const char IfFix[] =
      R"(void f() {
  int potato = 0;
  if (false) {
    int c = potato;
  } else if (false) {
    int d = 0;
  }
})";
  EXPECT_EQ(IfFix, runCheckOnCode<IfFalseCheck>(Code));
}

TEST(OverlappingReplacementsTest, StartsWithCheckTest) {
  const char Code[] =
      R"(void f() {
  int a = 0;
  int potato = 0;
  if (int b = 0) {
    int c = potato;
  } else if (true) {
    int d = 0;
  }
})";

  const char StartsFix[] =
      R"(void f() {
  int a = 0;
  int tomato = 0;
  if (int b = 0) {
    int c = tomato;
  } else if (true) {
    int d = 0;
  }
})";
  EXPECT_EQ(StartsFix, runCheckOnCode<StartsWithPotaCheck>(Code));
}

TEST(OverlappingReplacementsTest, EndsWithCheckTest) {
  const char Code[] =
      R"(void f() {
  int a = 0;
  int potato = 0;
  if (int b = 0) {
    int c = potato;
  } else if (true) {
    int d = 0;
  }
})";

  const char EndsFix[] =
      R"(void f() {
  int a = 0;
  int pomelo = 0;
  if (int b = 0) {
    int c = pomelo;
  } else if (true) {
    int d = 0;
  }
})";
  EXPECT_EQ(EndsFix, runCheckOnCode<EndsWithTatoCheck>(Code));
}

TEST(OverlappingReplacementTest, ReplacementsDoNotOverlap) {
  std::string Res;
  const char Code[] =
      R"(void f() {
  int potassium = 0;
  if (true) {
    int Potato = potassium;
  }
})";

  const char CharIfFix[] =
      R"(void f() {
  char potassium = 0;
  if (false) {
    char Potato = potassium;
  }
})";
  Res = runCheckOnCode<UseCharCheck, IfFalseCheck>(Code);
  EXPECT_EQ(CharIfFix, Res);

  const char StartsEndsFix[] =
      R"(void f() {
  int tomassium = 0;
  if (true) {
    int Pomelo = tomassium;
  }
})";
  Res = runCheckOnCode<StartsWithPotaCheck, EndsWithTatoCheck>(Code);
  EXPECT_EQ(StartsEndsFix, Res);

  const char CharIfStartsEndsFix[] =
      R"(void f() {
  char tomassium = 0;
  if (false) {
    char Pomelo = tomassium;
  }
})";
  Res = runCheckOnCode<UseCharCheck, IfFalseCheck, StartsWithPotaCheck,
                       EndsWithTatoCheck>(Code);
  EXPECT_EQ(CharIfStartsEndsFix, Res);
}

TEST(OverlappingReplacementsTest, ReplacementInsideOtherReplacement) {
  std::string Res;
  const char Code[] =
      R"(void f() {
  if (char potato = 0) {
  } else if (int a = 0) {
    char potato = 0;
    if (potato) potato;
  }
})";

  // Apply the UseCharCheck together with the IfFalseCheck.
  //
  // The 'If' fix contains the other, so that is the one that has to be applied.
  // } else if (int a = 0) {
  //            ^^^ -> char
  //            ~~~~~~~~~ -> false
  const char CharIfFix[] =
      R"(void f() {
  if (false) {
  } else if (false) {
    char potato = 0;
    if (false) potato;
  }
})";
  Res = runCheckOnCode<UseCharCheck, IfFalseCheck>(Code);
  EXPECT_EQ(CharIfFix, Res);
  Res = runCheckOnCode<IfFalseCheck, UseCharCheck>(Code);
  EXPECT_EQ(CharIfFix, Res);

  // Apply the IfFalseCheck with the StartsWithPotaCheck.
  //
  // The 'If' replacement is bigger here.
  // if (char potato = 0) {
  //          ^^^^^^ -> tomato
  //     ~~~~~~~~~~~~~~~ -> false
  //
  // But the refactoring is the one that contains the other here:
  // char potato = 0;
  //      ^^^^^^ -> tomato
  // if (potato) potato;
  //     ^^^^^^  ^^^^^^ -> tomato, tomato
  //     ~~~~~~ -> false
  const char IfStartsFix[] =
      R"(void f() {
  if (false) {
  } else if (false) {
    char tomato = 0;
    if (tomato) tomato;
  }
})";
  Res = runCheckOnCode<IfFalseCheck, StartsWithPotaCheck>(Code);
  EXPECT_EQ(IfStartsFix, Res);
  Res = runCheckOnCode<StartsWithPotaCheck, IfFalseCheck>(Code);
  EXPECT_EQ(IfStartsFix, Res);
}

TEST(OverlappingReplacements, TwoReplacementsInsideOne) {
  std::string Res;
  const char Code[] =
      R"(void f() {
  if (int potato = 0) {
    int a = 0;
  }
})";

  // The two smallest replacements should not be applied.
  // if (int potato = 0) {
  //         ^^^^^^ -> tomato
  //     *** -> char
  //     ~~~~~~~~~~~~~~ -> false
  // But other errors from the same checks should not be affected.
  //   int a = 0;
  //   *** -> char
  const char Fix[] =
      R"(void f() {
  if (false) {
    char a = 0;
  }
})";
  Res = runCheckOnCode<UseCharCheck, IfFalseCheck, StartsWithPotaCheck>(Code);
  EXPECT_EQ(Fix, Res);
  Res = runCheckOnCode<StartsWithPotaCheck, IfFalseCheck, UseCharCheck>(Code);
  EXPECT_EQ(Fix, Res);
}

TEST(OverlappingReplacementsTest,
     ApplyAtMostOneOfTheChangesWhenPartialOverlapping) {
  std::string Res;
  const char Code[] =
      R"(void f() {
  if (int potato = 0) {
    int a = potato;
  }
})";

  // These two replacements overlap, but none of them is completely contained
  // inside the other.
  // if (int potato = 0) {
  //         ^^^^^^ -> tomato
  //     ~~~~~~~~~~~~~~ -> false
  //   int a = potato;
  //           ^^^^^^ -> tomato
  //
  // The 'StartsWithPotaCheck' fix has endpoints inside the 'IfFalseCheck' fix,
  // so it is going to be set as inapplicable. The 'if' fix will be applied.
  const char IfFix[] =
      R"(void f() {
  if (false) {
    int a = potato;
  }
})";
  Res = runCheckOnCode<IfFalseCheck, StartsWithPotaCheck>(Code);
  EXPECT_EQ(IfFix, Res);
}

TEST(OverlappingReplacementsTest, TwoErrorsHavePerfectOverlapping) {
  std::string Res;
  const char Code[] =
      R"(void f() {
  int potato = 0;
  potato += potato * potato;
  if (char a = potato) potato;
})";

  // StartsWithPotaCheck will try to refactor 'potato' into 'tomato', and
  // EndsWithTatoCheck will try to use 'pomelo'. Both fixes have the same set of
  // ranges. This is a corner case of one error completely containing another:
  // the other completely contains the first one as well. Both errors are
  // discarded.

  Res = runCheckOnCode<StartsWithPotaCheck, EndsWithTatoCheck>(Code);
  EXPECT_EQ(Code, Res);
}

} // namespace test
} // namespace tidy
} // namespace clang
