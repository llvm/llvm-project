//===- unittest/Tooling/SourceCodeTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/SourceCode.h"
#include "TestVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace clang;
using namespace clang::ast_matchers;

using llvm::Failed;
using llvm::Succeeded;
using llvm::ValueIs;
using testing::Optional;
using tooling::getAssociatedRange;
using tooling::getExtendedRange;
using tooling::getExtendedText;
using tooling::getFileRangeForEdit;
using tooling::getText;
using tooling::maybeExtendRange;
using tooling::validateEditRange;

namespace {

struct IntLitVisitor : TestVisitor<IntLitVisitor> {
  bool VisitIntegerLiteral(IntegerLiteral *Expr) {
    OnIntLit(Expr, Context);
    return true;
  }

  std::function<void(IntegerLiteral *, ASTContext *Context)> OnIntLit;
};

struct CallsVisitor : TestVisitor<CallsVisitor> {
  bool VisitCallExpr(CallExpr *Expr) {
    OnCall(Expr, Context);
    return true;
  }

  std::function<void(CallExpr *, ASTContext *Context)> OnCall;
};

struct TypeLocVisitor : TestVisitor<TypeLocVisitor> {
  bool VisitTypeLoc(TypeLoc TL) {
    OnTypeLoc(TL, Context);
    return true;
  }

  std::function<void(TypeLoc, ASTContext *Context)> OnTypeLoc;
};

// Equality matcher for `clang::CharSourceRange`, which lacks `operator==`.
MATCHER_P(EqualsRange, R, "") {
  return arg.isTokenRange() == R.isTokenRange() &&
         arg.getBegin() == R.getBegin() && arg.getEnd() == R.getEnd();
}

MATCHER_P2(EqualsAnnotatedRange, Context, R, "") {
  if (arg.getBegin().isMacroID()) {
    *result_listener << "which starts in a macro";
    return false;
  }
  if (arg.getEnd().isMacroID()) {
    *result_listener << "which ends in a macro";
    return false;
  }

  CharSourceRange Range = Lexer::getAsCharRange(
      arg, Context->getSourceManager(), Context->getLangOpts());
  unsigned Begin = Context->getSourceManager().getFileOffset(Range.getBegin());
  unsigned End = Context->getSourceManager().getFileOffset(Range.getEnd());

  *result_listener << "which is a " << (arg.isTokenRange() ? "Token" : "Char")
                   << " range [" << Begin << "," << End << ")";
  return Begin == R.Begin && End == R.End;
}

static ::testing::Matcher<CharSourceRange> AsRange(const SourceManager &SM,
                                                   llvm::Annotations::Range R) {
  return EqualsRange(CharSourceRange::getCharRange(
      SM.getLocForStartOfFile(SM.getMainFileID()).getLocWithOffset(R.Begin),
      SM.getLocForStartOfFile(SM.getMainFileID()).getLocWithOffset(R.End)));
}

// Base class for visitors that expect a single match corresponding to a
// specific annotated range.
template <typename T> class AnnotatedCodeVisitor : public TestVisitor<T> {
protected:
  int MatchCount = 0;
  llvm::Annotations Code;

public:
  AnnotatedCodeVisitor() : Code("$r[[]]") {}
  // Helper for tests of `getAssociatedRange`.
  bool VisitDeclHelper(Decl *Decl) {
    // Only consider explicit declarations.
    if (Decl->isImplicit())
      return true;

    ++MatchCount;
    EXPECT_THAT(getAssociatedRange(*Decl, *this->Context),
                EqualsAnnotatedRange(this->Context, Code.range("r")))
        << Code.code();
    return true;
  }

  bool runOverAnnotated(llvm::StringRef AnnotatedCode,
                        std::vector<std::string> Args = {}) {
    Code = llvm::Annotations(AnnotatedCode);
    MatchCount = 0;
    Args.push_back("-std=c++11");
    Args.push_back("-fno-delayed-template-parsing");
    bool result = tooling::runToolOnCodeWithArgs(this->CreateTestAction(),
                                                 Code.code(), Args);
    EXPECT_EQ(MatchCount, 1) << AnnotatedCode;
    return result;
  }
};

TEST(SourceCodeTest, getText) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("foo(x, y)", getText(*CE, *Context));
  };
  Visitor.runOver("void foo(int x, int y) { foo(x, y); }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("APPLY(foo, x, y)", getText(*CE, *Context));
  };
  Visitor.runOver("#define APPLY(f, x, y) f(x, y)\n"
                  "void foo(int x, int y) { APPLY(foo, x, y); }");
}

TEST(SourceCodeTest, getTextWithMacro) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("F OO", getText(*CE, *Context));
    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    EXPECT_EQ("", getText(*P0, *Context));
    EXPECT_EQ("", getText(*P1, *Context));
  };
  Visitor.runOver("#define F foo(\n"
                  "#define OO x, y)\n"
                  "void foo(int x, int y) { F OO ; }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("", getText(*CE, *Context));
    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    EXPECT_EQ("x", getText(*P0, *Context));
    EXPECT_EQ("y", getText(*P1, *Context));
  };
  Visitor.runOver("#define FOO(x, y) (void)x; (void)y; foo(x, y);\n"
                  "void foo(int x, int y) { FOO(x,y) }");
}

TEST(SourceCodeTest, getExtendedText) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("foo(x, y);",
              getExtendedText(*CE, tok::TokenKind::semi, *Context));

    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    EXPECT_EQ("x", getExtendedText(*P0, tok::TokenKind::semi, *Context));
    EXPECT_EQ("x,", getExtendedText(*P0, tok::TokenKind::comma, *Context));
    EXPECT_EQ("y", getExtendedText(*P1, tok::TokenKind::semi, *Context));
  };
  Visitor.runOver("void foo(int x, int y) { foo(x, y); }");
  Visitor.runOver("void foo(int x, int y) { if (true) foo(x, y); }");
  Visitor.runOver("int foo(int x, int y) { if (true) return 3 + foo(x, y); }");
  Visitor.runOver("void foo(int x, int y) { for (foo(x, y);;) ++x; }");
  Visitor.runOver(
      "bool foo(int x, int y) { for (;foo(x, y);) x = 1; return true; }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("foo()", getExtendedText(*CE, tok::TokenKind::semi, *Context));
  };
  Visitor.runOver("bool foo() { if (foo()) return true; return false; }");
  Visitor.runOver("void foo() { int x; for (;; foo()) ++x; }");
  Visitor.runOver("int foo() { return foo() + 3; }");
}

TEST(SourceCodeTest, maybeExtendRange_TokenRange) {
  struct ExtendTokenRangeVisitor
      : AnnotatedCodeVisitor<ExtendTokenRangeVisitor> {
    bool VisitCallExpr(CallExpr *CE) {
      ++MatchCount;
      EXPECT_THAT(getExtendedRange(*CE, tok::TokenKind::semi, *Context),
                  EqualsAnnotatedRange(Context, Code.range("r")));
      return true;
    }
  };

  ExtendTokenRangeVisitor Visitor;
  // Extends to include semicolon.
  Visitor.runOverAnnotated("void f(int x, int y) { $r[[f(x, y);]] }");
  // Does not extend to include semicolon.
  Visitor.runOverAnnotated(
      "int f(int x, int y) { if (0) return $r[[f(x, y)]] + 3; }");
}

TEST(SourceCodeTest, maybeExtendRange_CharRange) {
  struct ExtendCharRangeVisitor : AnnotatedCodeVisitor<ExtendCharRangeVisitor> {
    bool VisitCallExpr(CallExpr *CE) {
      ++MatchCount;
      CharSourceRange Call = Lexer::getAsCharRange(CE->getSourceRange(),
                                                   Context->getSourceManager(),
                                                   Context->getLangOpts());
      EXPECT_THAT(maybeExtendRange(Call, tok::TokenKind::semi, *Context),
                  EqualsAnnotatedRange(Context, Code.range("r")));
      return true;
    }
  };
  ExtendCharRangeVisitor Visitor;
  // Extends to include semicolon.
  Visitor.runOverAnnotated("void f(int x, int y) { $r[[f(x, y);]] }");
  // Does not extend to include semicolon.
  Visitor.runOverAnnotated(
      "int f(int x, int y) { if (0) return $r[[f(x, y)]] + 3; }");
}

TEST(SourceCodeTest, getAssociatedRange) {
  struct VarDeclsVisitor : AnnotatedCodeVisitor<VarDeclsVisitor> {
    bool VisitVarDecl(VarDecl *Decl) { return VisitDeclHelper(Decl); }
  };
  VarDeclsVisitor Visitor;

  // Includes semicolon.
  Visitor.runOverAnnotated("$r[[int x = 4;]]");

  // Includes newline and semicolon.
  Visitor.runOverAnnotated("$r[[int x = 4;\n]]");

  // Includes trailing comments.
  Visitor.runOverAnnotated("$r[[int x = 4; // Comment\n]]");
  Visitor.runOverAnnotated("$r[[int x = 4; /* Comment */\n]]");

  // Does *not* include trailing comments when another entity appears between
  // the decl and the comment.
  Visitor.runOverAnnotated("$r[[int x = 4;]] class C {}; // Comment\n");

  // Includes attributes.
  Visitor.runOverAnnotated(R"cpp(
      $r[[__attribute__((deprecated("message")))
      int x;]])cpp");

  // Includes attributes and comments together.
  Visitor.runOverAnnotated(R"cpp(
      $r[[__attribute__((deprecated("message")))
      // Comment.
      int x;]])cpp");

  // Includes attributes through macro expansion.
  Visitor.runOverAnnotated(R"cpp(
      #define MACRO_EXPANSION __attribute__((deprecated("message")))
      $r[[MACRO_EXPANSION
      int x;]])cpp");

  // Includes attributes through macro expansion with comments.
  Visitor.runOverAnnotated(R"cpp(
      #define MACRO_EXPANSION __attribute__((deprecated("message")))
      $r[[MACRO_EXPANSION
      // Comment.
      int x;]])cpp");
}

TEST(SourceCodeTest, getAssociatedRangeClasses) {
  struct RecordDeclsVisitor : AnnotatedCodeVisitor<RecordDeclsVisitor> {
    bool VisitRecordDecl(RecordDecl *Decl) { return VisitDeclHelper(Decl); }
  };
  RecordDeclsVisitor Visitor;

  Visitor.runOverAnnotated("$r[[class A;]]");
  Visitor.runOverAnnotated("$r[[class A {};]]");

  // Includes leading template annotation.
  Visitor.runOverAnnotated("$r[[template <typename T> class A;]]");
  Visitor.runOverAnnotated("$r[[template <typename T> class A {};]]");
}

TEST(SourceCodeTest, getAssociatedRangeClassTemplateSpecializations) {
  struct CXXRecordDeclsVisitor : AnnotatedCodeVisitor<CXXRecordDeclsVisitor> {
    bool VisitCXXRecordDecl(CXXRecordDecl *Decl) {
      return Decl->getTemplateSpecializationKind() !=
                 TSK_ExplicitSpecialization ||
             VisitDeclHelper(Decl);
    }
  };
  CXXRecordDeclsVisitor Visitor;

  Visitor.runOverAnnotated(R"cpp(
      template <typename T> class A{};
      $r[[template <> class A<int>;]])cpp");
  Visitor.runOverAnnotated(R"cpp(
      template <typename T> class A{};
      $r[[template <> class A<int> {};]])cpp");
}

TEST(SourceCodeTest, getAssociatedRangeFunctions) {
  struct FunctionDeclsVisitor : AnnotatedCodeVisitor<FunctionDeclsVisitor> {
    bool VisitFunctionDecl(FunctionDecl *Decl) { return VisitDeclHelper(Decl); }
  };
  FunctionDeclsVisitor Visitor;

  Visitor.runOverAnnotated("$r[[int f();]]");
  Visitor.runOverAnnotated("$r[[int f() { return 0; }]]");
  // Includes leading template annotation.
  Visitor.runOverAnnotated("$r[[template <typename T> int f();]]");
  Visitor.runOverAnnotated("$r[[template <typename T> int f() { return 0; }]]");
}

TEST(SourceCodeTest, getAssociatedRangeMemberTemplates) {
  struct CXXMethodDeclsVisitor : AnnotatedCodeVisitor<CXXMethodDeclsVisitor> {
    bool VisitCXXMethodDecl(CXXMethodDecl *Decl) {
      // Only consider the definition of the template.
      return !Decl->doesThisDeclarationHaveABody() || VisitDeclHelper(Decl);
    }
  };
  CXXMethodDeclsVisitor Visitor;

  Visitor.runOverAnnotated(R"cpp(
      template <typename C>
      struct A { template <typename T> int member(T v); };

      $r[[template <typename C>
      template  <typename T>
      int A<C>::member(T v) { return 0; }]])cpp");
}

TEST(SourceCodeTest, getAssociatedRangeWithComments) {
  struct VarDeclsVisitor : AnnotatedCodeVisitor<VarDeclsVisitor> {
    bool VisitVarDecl(VarDecl *Decl) { return VisitDeclHelper(Decl); }
  };

  VarDeclsVisitor Visitor;
  auto Visit = [&](llvm::StringRef AnnotatedCode) {
    Visitor.runOverAnnotated(AnnotatedCode, {"-fparse-all-comments"});
  };

  // Includes leading comments.
  Visit("$r[[// Comment.\nint x = 4;]]");
  Visit("$r[[// Comment.\nint x = 4;\n]]");
  Visit("$r[[/* Comment.*/\nint x = 4;\n]]");
  // ... even if separated by (extra) horizontal whitespace.
  Visit("$r[[/* Comment.*/  \nint x = 4;\n]]");

  // Includes comments even in the presence of trailing whitespace.
  Visit("$r[[// Comment.\nint x = 4;]]  ");

  // Includes comments when the declaration is followed by the beginning or end
  // of a compound statement.
  Visit(R"cpp(
  void foo() {
    $r[[/* C */
    int x = 4;
  ]]};)cpp");
  Visit(R"cpp(
  void foo() {
    $r[[/* C */
    int x = 4;
   ]]{ class Foo {}; }
   })cpp");

  // Includes comments inside macros (when decl is in the same macro).
  Visit(R"cpp(
      #define DECL /* Comment */ int x
      $r[[DECL;]])cpp");

  Visit(R"cpp(
      #define DECL int x
      $r[[// Comment
      DECL;]])cpp");
  // Does not include comments when only the comment come from a macro.
  Visit(R"cpp(
      #define COMMENT /* Comment */
      COMMENT
      $r[[int x;]])cpp");

  // Includes multi-line comments.
  Visit(R"cpp(
      $r[[/* multi
       * line
       * comment
       */
      int x;]])cpp");
  Visit(R"cpp(
      $r[[// multi
      // line
      // comment
      int x;]])cpp");

  // Does not include comments separated by multiple empty lines.
  Visit("// Comment.\n\n\n$r[[int x = 4;\n]]");
  Visit("/* Comment.*/\n\n\n$r[[int x = 4;\n]]");

  // Does not include comments before a *series* of declarations.
  Visit(R"cpp(
      // Comment.
      $r[[int x = 4;
      ]]class foo {};)cpp");

  // Does not include IfThisThenThat comments
  Visit("// LINT.IfChange.\n$r[[int x = 4;]]");
  Visit("// LINT.ThenChange.\n$r[[int x = 4;]]");

  // Includes attributes.
  Visit(R"cpp(
      $r[[__attribute__((deprecated("message")))
      int x;]])cpp");

  // Includes attributes and comments together.
  Visit(R"cpp(
      $r[[__attribute__((deprecated("message")))
      // Comment.
      int x;]])cpp");

  // Includes attributes through macro expansion.
  Visitor.runOverAnnotated(R"cpp(
      #define MACRO_EXPANSION __attribute__((deprecated("message")))
      $r[[MACRO_EXPANSION
      int x;]])cpp");

  // Includes attributes through macro expansion with comments.
  Visitor.runOverAnnotated(R"cpp(
      #define MACRO_EXPANSION __attribute__((deprecated("message")))
      $r[[MACRO_EXPANSION
      // Comment.
      int x;]])cpp");
}

TEST(SourceCodeTest, getAssociatedRangeInvalidForPartialExpansions) {
  struct FailingVarDeclsVisitor : TestVisitor<FailingVarDeclsVisitor> {
    FailingVarDeclsVisitor() {}
    bool VisitVarDecl(VarDecl *Decl) {
      EXPECT_TRUE(getAssociatedRange(*Decl, *Context).isInvalid());
      return true;
    }
  };

  FailingVarDeclsVisitor Visitor;
  // Should fail because it only includes a part of the expansion.
  std::string Code = R"cpp(
      #define DECL class foo { }; int x
      DECL;)cpp";
  Visitor.runOver(Code);
}

class GetFileRangeForEditTest : public testing::TestWithParam<bool> {};
INSTANTIATE_TEST_SUITE_P(WithAndWithoutExpansions, GetFileRangeForEditTest,
                         testing::Bool());

TEST_P(GetFileRangeForEditTest, EditRangeWithMacroExpansionsShouldSucceed) {
  // The call expression, whose range we are extracting, includes two macro
  // expansions.
  llvm::Annotations Code(R"cpp(
#define M(a) a * 13
int foo(int x, int y);
int a = $r[[foo(M(1), M(2))]];
)cpp");

  CallsVisitor Visitor;
  Visitor.OnCall = [&Code](CallExpr *CE, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(CE->getSourceRange());
    EXPECT_THAT(getFileRangeForEdit(Range, *Context, GetParam()),
                ValueIs(AsRange(Context->getSourceManager(), Code.range("r"))));
  };
  Visitor.runOver(Code.code());
}

TEST(SourceCodeTest, EditWholeMacroExpansionShouldSucceed) {
  llvm::Annotations Code(R"cpp(
#define FOO 10
int a = $r[[FOO]];
)cpp");

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [&Code](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT(getFileRangeForEdit(Range, *Context),
                ValueIs(AsRange(Context->getSourceManager(), Code.range("r"))));
  };
  Visitor.runOver(Code.code());
}

TEST(SourceCodeTest, EditInvolvingExpansionIgnoringExpansionShouldFail) {
  // If we specify to ignore macro expansions, none of these call expressions
  // should have an editable range.
  llvm::Annotations Code(R"cpp(
#define M1(x) x(1)
#define M2(x, y) x ## y
#define M3(x) foobar(x)
int foobar(int);
int a = M1(foobar);
int b = M2(foo, bar(2));
int c = M3(3);
)cpp");

  CallsVisitor Visitor;
  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(CE->getSourceRange());
    EXPECT_FALSE(
        getFileRangeForEdit(Range, *Context, /*IncludeMacroExpansion=*/false));
  };
  Visitor.runOver(Code.code());
}

TEST(SourceCodeTest, InnerNestedTemplate) {
  llvm::Annotations Code(R"cpp(
    template <typename T>
    struct A {};
    template <typename T>
    struct B {};
    template <typename T>
    struct C {};

    void f(A<B<C<int>$r[[>>]]);
  )cpp");

  TypeLocVisitor Visitor;
  Visitor.OnTypeLoc = [&](TypeLoc TL, ASTContext *Context) {
    if (TL.getSourceRange().isInvalid())
      return;

    // There are no macros, so every TypeLoc's range should be valid.
    auto Range = CharSourceRange::getTokenRange(TL.getSourceRange());
    auto LastTokenRange = CharSourceRange::getTokenRange(TL.getEndLoc());
    EXPECT_TRUE(getFileRangeForEdit(Range, *Context,
                                    /*IncludeMacroExpansion=*/false))
        << TL.getSourceRange().printToString(Context->getSourceManager());
    EXPECT_TRUE(getFileRangeForEdit(LastTokenRange, *Context,
                                    /*IncludeMacroExpansion=*/false))
        << TL.getEndLoc().printToString(Context->getSourceManager());

    if (auto matches = match(
            templateSpecializationTypeLoc(
                loc(templateSpecializationType(
                    hasDeclaration(cxxRecordDecl(hasName("A"))))),
                hasTemplateArgumentLoc(
                    0, templateArgumentLoc(hasTypeLoc(typeLoc().bind("b"))))),
            TL, *Context);
        !matches.empty()) {
      // A range where the start token is split, but the end token is not.
      auto OuterTL = TL;
      auto MiddleTL = *matches[0].getNodeAs<TypeLoc>("b");
      EXPECT_THAT(
          getFileRangeForEdit(CharSourceRange::getTokenRange(
                                  MiddleTL.getEndLoc(), OuterTL.getEndLoc()),
                              *Context, /*IncludeMacroExpansion=*/false),
          Optional(EqualsAnnotatedRange(Context, Code.range("r"))));
    }
  };
  Visitor.runOver(Code.code(), TypeLocVisitor::Lang_CXX11);
}

TEST_P(GetFileRangeForEditTest, EditPartialMacroExpansionShouldFail) {
  std::string Code = R"cpp(
#define BAR 10+
int c = BAR 3.0;
)cpp";

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_FALSE(getFileRangeForEdit(Range, *Context, GetParam()));
  };
  Visitor.runOver(Code);
}

TEST_P(GetFileRangeForEditTest, EditWholeMacroArgShouldSucceed) {
  llvm::Annotations Code(R"cpp(
#define FOO(a) a + 7.0;
int a = FOO($r[[10]]);
)cpp");

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [&Code](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT(getFileRangeForEdit(Range, *Context, GetParam()),
                ValueIs(AsRange(Context->getSourceManager(), Code.range("r"))));
  };
  Visitor.runOver(Code.code());
}

TEST_P(GetFileRangeForEditTest, EditPartialMacroArgShouldSucceed) {
  llvm::Annotations Code(R"cpp(
#define FOO(a) a + 7.0;
int a = FOO($r[[10]] + 10.0);
)cpp");

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [&Code](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT(getFileRangeForEdit(Range, *Context, GetParam()),
                ValueIs(AsRange(Context->getSourceManager(), Code.range("r"))));
  };
  Visitor.runOver(Code.code());
}

TEST(SourceCodeTest, EditRangeWithMacroExpansionsIsValid) {
  // The call expression, whose range we are extracting, includes two macro
  // expansions.
  llvm::StringRef Code = R"cpp(
#define M(a) a * 13
int foo(int x, int y);
int a = foo(M(1), M(2));
)cpp";

  CallsVisitor Visitor;
  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(CE->getSourceRange());
    EXPECT_THAT_ERROR(validateEditRange(Range, Context->getSourceManager()),
                      Succeeded());
  };
  Visitor.runOver(Code);
}

TEST(SourceCodeTest, SpellingRangeOfMacroArgIsValid) {
  llvm::StringRef Code = R"cpp(
#define FOO(a) a + 7.0;
int a = FOO(10);
)cpp";

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [](IntegerLiteral *Expr, ASTContext *Context) {
    SourceLocation ArgLoc =
        Context->getSourceManager().getSpellingLoc(Expr->getBeginLoc());
    // The integer literal is a single token.
    auto ArgRange = CharSourceRange::getTokenRange(ArgLoc);
    EXPECT_THAT_ERROR(validateEditRange(ArgRange, Context->getSourceManager()),
                      Succeeded());
  };
  Visitor.runOver(Code);
}

TEST(SourceCodeTest, InvalidEditRangeIsInvalid) {
  llvm::StringRef Code = "int c = 10;";

  // We use the visitor just to get a valid context.
  IntLitVisitor Visitor;
  Visitor.OnIntLit = [](IntegerLiteral *, ASTContext *Context) {
    CharSourceRange Invalid;
    EXPECT_THAT_ERROR(validateEditRange(Invalid, Context->getSourceManager()),
                      Failed());
  };
  Visitor.runOver(Code);
}

TEST(SourceCodeTest, InvertedEditRangeIsInvalid) {
  llvm::StringRef Code = R"cpp(
int foo(int x);
int a = foo(2);
)cpp";

  CallsVisitor Visitor;
  Visitor.OnCall = [](CallExpr *Expr, ASTContext *Context) {
    auto InvertedRange = CharSourceRange::getTokenRange(
        SourceRange(Expr->getEndLoc(), Expr->getBeginLoc()));
    EXPECT_THAT_ERROR(
        validateEditRange(InvertedRange, Context->getSourceManager()),
        Failed());
  };
  Visitor.runOver(Code);
}

TEST(SourceCodeTest, MacroArgIsInvalid) {
  llvm::StringRef Code = R"cpp(
#define FOO(a) a + 7.0;
int a = FOO(10);
)cpp";

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT_ERROR(validateEditRange(Range, Context->getSourceManager()),
                      Failed());
  };
  Visitor.runOver(Code);
}

TEST(SourceCodeTest, EditWholeMacroExpansionIsInvalid) {
  llvm::StringRef Code = R"cpp(
#define FOO 10
int a = FOO;
)cpp";

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT_ERROR(validateEditRange(Range, Context->getSourceManager()),
                      Failed());

  };
  Visitor.runOver(Code);
}

TEST(SourceCodeTest, EditPartialMacroExpansionIsInvalid) {
  llvm::StringRef Code = R"cpp(
#define BAR 10+
int c = BAR 3.0;
)cpp";

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT_ERROR(validateEditRange(Range, Context->getSourceManager()),
                      Failed());
  };
  Visitor.runOver(Code);
}

TEST(SourceCodeTest, GetCallReturnType_Dependent) {
  llvm::Annotations Code{R"cpp(
template<class T, class F>
void templ(const T& t, F f) {}

template<class T, class F>
void templ1(const T& t, F f) {
  $test1[[f(t)]];
}

int f_overload(int) { return 1; }
int f_overload(double) { return 2; }

void f1() {
  int i = 0;
  templ(i, [](const auto &p) {
    $test2[[f_overload(p)]];
  });
}

struct A {
  void f_overload(int);
  void f_overload(double);
};

void f2() {
 int i = 0;
 templ(i, [](const auto &p) {
   A a;
   $test3[[a.f_overload(p)]];
 });
}
)cpp"};

  llvm::Annotations::Range R1 = Code.range("test1");
  llvm::Annotations::Range R2 = Code.range("test2");
  llvm::Annotations::Range R3 = Code.range("test3");

  CallsVisitor Visitor;
  Visitor.OnCall = [&R1, &R2, &R3](CallExpr *Expr, ASTContext *Context) {
    unsigned Begin = Context->getSourceManager().getFileOffset(
        Expr->getSourceRange().getBegin());
    unsigned End = Context->getSourceManager().getFileOffset(
        Expr->getSourceRange().getEnd());
    llvm::Annotations::Range R{Begin, End + 1};

    QualType CalleeType = Expr->getCallee()->getType();
    if (R == R1) {
      ASSERT_TRUE(CalleeType->isDependentType());
      EXPECT_EQ(Expr->getCallReturnType(*Context), Context->DependentTy);
    } else if (R == R2) {
      ASSERT_FALSE(CalleeType->isDependentType());
      ASSERT_TRUE(CalleeType->isSpecificPlaceholderType(BuiltinType::Overload));
      ASSERT_TRUE(isa<UnresolvedLookupExpr>(Expr->getCallee()));
      EXPECT_EQ(Expr->getCallReturnType(*Context), Context->DependentTy);
    } else if (R == R3) {
      ASSERT_FALSE(CalleeType->isDependentType());
      ASSERT_TRUE(
          CalleeType->isSpecificPlaceholderType(BuiltinType::BoundMember));
      ASSERT_TRUE(isa<UnresolvedMemberExpr>(Expr->getCallee()));
      EXPECT_EQ(Expr->getCallReturnType(*Context), Context->DependentTy);
    }
  };
  Visitor.runOver(Code.code(), CallsVisitor::Lang_CXX14);
}

} // end anonymous namespace
