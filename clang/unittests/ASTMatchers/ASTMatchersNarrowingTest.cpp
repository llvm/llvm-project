// unittests/ASTMatchers/ASTMatchersNarrowingTest.cpp - AST matcher unit tests//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTMatchersTest.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

TEST_P(ASTMatchersTest, IsExpandedFromMacro_MatchesInFile) {
  StringRef input = R"cc(
#define MY_MACRO(a) (4 + (a))
    void Test() { MY_MACRO(4); }
  )cc";
  EXPECT_TRUE(matches(input, binaryOperator(isExpandedFromMacro("MY_MACRO"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_MatchesNested) {
  StringRef input = R"cc(
#define MY_MACRO(a) (4 + (a))
#define WRAPPER(a) MY_MACRO(a)
    void Test() { WRAPPER(4); }
  )cc";
  EXPECT_TRUE(matches(input, binaryOperator(isExpandedFromMacro("MY_MACRO"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_MatchesIntermediate) {
  StringRef input = R"cc(
#define IMPL(a) (4 + (a))
#define MY_MACRO(a) IMPL(a)
#define WRAPPER(a) MY_MACRO(a)
    void Test() { WRAPPER(4); }
  )cc";
  EXPECT_TRUE(matches(input, binaryOperator(isExpandedFromMacro("MY_MACRO"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_MatchesTransitive) {
  StringRef input = R"cc(
#define MY_MACRO(a) (4 + (a))
#define WRAPPER(a) MY_MACRO(a)
    void Test() { WRAPPER(4); }
  )cc";
  EXPECT_TRUE(matches(input, binaryOperator(isExpandedFromMacro("WRAPPER"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_MatchesArgument) {
  StringRef input = R"cc(
#define MY_MACRO(a) (4 + (a))
    void Test() {
      int x = 5;
      MY_MACRO(x);
    }
  )cc";
  EXPECT_TRUE(matches(input, declRefExpr(isExpandedFromMacro("MY_MACRO"))));
}

// Like IsExpandedFromMacro_MatchesArgument, but the argument is itself a
// macro.
TEST_P(ASTMatchersTest, IsExpandedFromMacro_MatchesArgumentMacroExpansion) {
  StringRef input = R"cc(
#define MY_MACRO(a) (4 + (a))
#define IDENTITY(a) (a)
    void Test() {
      IDENTITY(MY_MACRO(2));
    }
  )cc";
  EXPECT_TRUE(matches(input, binaryOperator(isExpandedFromMacro("IDENTITY"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_MatchesWhenInArgument) {
  StringRef input = R"cc(
#define MY_MACRO(a) (4 + (a))
#define IDENTITY(a) (a)
    void Test() {
      IDENTITY(MY_MACRO(2));
    }
  )cc";
  EXPECT_TRUE(matches(input, binaryOperator(isExpandedFromMacro("MY_MACRO"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_MatchesObjectMacro) {
  StringRef input = R"cc(
#define PLUS (2 + 2)
    void Test() {
      PLUS;
    }
  )cc";
  EXPECT_TRUE(matches(input, binaryOperator(isExpandedFromMacro("PLUS"))));
}

TEST(IsExpandedFromMacro, MatchesFromCommandLine) {
  StringRef input = R"cc(
    void Test() { FOUR_PLUS_FOUR; }
  )cc";
  EXPECT_TRUE(matchesConditionally(
      input, binaryOperator(isExpandedFromMacro("FOUR_PLUS_FOUR")), true,
      {"-std=c++11", "-DFOUR_PLUS_FOUR=4+4"}));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_NotMatchesBeginOnly) {
  StringRef input = R"cc(
#define ONE_PLUS 1+
  void Test() { ONE_PLUS 4; }
  )cc";
  EXPECT_TRUE(
      notMatches(input, binaryOperator(isExpandedFromMacro("ONE_PLUS"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_NotMatchesEndOnly) {
  StringRef input = R"cc(
#define PLUS_ONE +1
  void Test() { 4 PLUS_ONE; }
  )cc";
  EXPECT_TRUE(
      notMatches(input, binaryOperator(isExpandedFromMacro("PLUS_ONE"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_NotMatchesDifferentMacro) {
  StringRef input = R"cc(
#define MY_MACRO(a) (4 + (a))
    void Test() { MY_MACRO(4); }
  )cc";
  EXPECT_TRUE(notMatches(input, binaryOperator(isExpandedFromMacro("OTHER"))));
}

TEST_P(ASTMatchersTest, IsExpandedFromMacro_NotMatchesDifferentInstances) {
  StringRef input = R"cc(
#define FOUR 4
    void Test() { FOUR + FOUR; }
  )cc";
  EXPECT_TRUE(notMatches(input, binaryOperator(isExpandedFromMacro("FOUR"))));
}

TEST(IsExpandedFromMacro, IsExpandedFromMacro_MatchesDecls) {
  StringRef input = R"cc(
#define MY_MACRO(a) int i = a;
    void Test() { MY_MACRO(4); }
  )cc";
  EXPECT_TRUE(matches(input, varDecl(isExpandedFromMacro("MY_MACRO"))));
}

TEST(IsExpandedFromMacro, IsExpandedFromMacro_MatchesTypelocs) {
  StringRef input = R"cc(
#define MY_TYPE int
    void Test() { MY_TYPE i = 4; }
  )cc";
  EXPECT_TRUE(matches(input, typeLoc(isExpandedFromMacro("MY_TYPE"))));
}

TEST_P(ASTMatchersTest, AllOf) {
  const char Program[] = "struct T { };"
                         "int f(int, struct T*, int, int);"
                         "void g(int x) { struct T t; f(x, &t, 3, 4); }";
  EXPECT_TRUE(matches(
      Program, callExpr(allOf(callee(functionDecl(hasName("f"))),
                              hasArgument(0, declRefExpr(to(varDecl())))))));
  EXPECT_TRUE(matches(
      Program,
      callExpr(
          allOf(callee(functionDecl(hasName("f"))),
                hasArgument(0, declRefExpr(to(varDecl()))),
                hasArgument(1, hasType(pointsTo(recordDecl(hasName("T")))))))));
  EXPECT_TRUE(matches(
      Program, callExpr(allOf(
                   callee(functionDecl(hasName("f"))),
                   hasArgument(0, declRefExpr(to(varDecl()))),
                   hasArgument(1, hasType(pointsTo(recordDecl(hasName("T"))))),
                   hasArgument(2, integerLiteral(equals(3)))))));
  EXPECT_TRUE(matches(
      Program, callExpr(allOf(
                   callee(functionDecl(hasName("f"))),
                   hasArgument(0, declRefExpr(to(varDecl()))),
                   hasArgument(1, hasType(pointsTo(recordDecl(hasName("T"))))),
                   hasArgument(2, integerLiteral(equals(3))),
                   hasArgument(3, integerLiteral(equals(4)))))));
}

TEST_P(ASTMatchersTest, Has) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `has()` that does not depend on C++.
    return;
  }

  DeclarationMatcher HasClassX = recordDecl(has(recordDecl(hasName("X"))));
  EXPECT_TRUE(matches("class Y { class X {}; };", HasClassX));
  EXPECT_TRUE(matches("class X {};", HasClassX));

  DeclarationMatcher YHasClassX =
      recordDecl(hasName("Y"), has(recordDecl(hasName("X"))));
  EXPECT_TRUE(matches("class Y { class X {}; };", YHasClassX));
  EXPECT_TRUE(notMatches("class X {};", YHasClassX));
  EXPECT_TRUE(notMatches("class Y { class Z { class X {}; }; };", YHasClassX));
}

TEST_P(ASTMatchersTest, Has_RecursiveAllOf) {
  if (!GetParam().isCXX()) {
    return;
  }

  DeclarationMatcher Recursive =
      recordDecl(has(recordDecl(has(recordDecl(hasName("X"))),
                                has(recordDecl(hasName("Y"))), hasName("Z"))),
                 has(recordDecl(has(recordDecl(hasName("A"))),
                                has(recordDecl(hasName("B"))), hasName("C"))),
                 hasName("F"));

  EXPECT_TRUE(matches("class F {"
                      "  class Z {"
                      "    class X {};"
                      "    class Y {};"
                      "  };"
                      "  class C {"
                      "    class A {};"
                      "    class B {};"
                      "  };"
                      "};",
                      Recursive));

  EXPECT_TRUE(matches("class F {"
                      "  class Z {"
                      "    class A {};"
                      "    class X {};"
                      "    class Y {};"
                      "  };"
                      "  class C {"
                      "    class X {};"
                      "    class A {};"
                      "    class B {};"
                      "  };"
                      "};",
                      Recursive));

  EXPECT_TRUE(matches("class O1 {"
                      "  class O2 {"
                      "    class F {"
                      "      class Z {"
                      "        class A {};"
                      "        class X {};"
                      "        class Y {};"
                      "      };"
                      "      class C {"
                      "        class X {};"
                      "        class A {};"
                      "        class B {};"
                      "      };"
                      "    };"
                      "  };"
                      "};",
                      Recursive));
}

TEST_P(ASTMatchersTest, Has_RecursiveAnyOf) {
  if (!GetParam().isCXX()) {
    return;
  }

  DeclarationMatcher Recursive = recordDecl(
      anyOf(has(recordDecl(anyOf(has(recordDecl(hasName("X"))),
                                 has(recordDecl(hasName("Y"))), hasName("Z")))),
            has(recordDecl(anyOf(hasName("C"), has(recordDecl(hasName("A"))),
                                 has(recordDecl(hasName("B")))))),
            hasName("F")));

  EXPECT_TRUE(matches("class F {};", Recursive));
  EXPECT_TRUE(matches("class Z {};", Recursive));
  EXPECT_TRUE(matches("class C {};", Recursive));
  EXPECT_TRUE(matches("class M { class N { class X {}; }; };", Recursive));
  EXPECT_TRUE(matches("class M { class N { class B {}; }; };", Recursive));
  EXPECT_TRUE(matches("class O1 { class O2 {"
                      "  class M { class N { class B {}; }; }; "
                      "}; };",
                      Recursive));
}

TEST_P(ASTMatchersTest, Unless) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `unless()` that does not depend on C++.
    return;
  }

  DeclarationMatcher NotClassX =
      cxxRecordDecl(isDerivedFrom("Y"), unless(hasName("X")));
  EXPECT_TRUE(notMatches("", NotClassX));
  EXPECT_TRUE(notMatches("class Y {};", NotClassX));
  EXPECT_TRUE(matches("class Y {}; class Z : public Y {};", NotClassX));
  EXPECT_TRUE(notMatches("class Y {}; class X : public Y {};", NotClassX));
  EXPECT_TRUE(
      notMatches("class Y {}; class Z {}; class X : public Y {};", NotClassX));

  DeclarationMatcher ClassXHasNotClassY =
      recordDecl(hasName("X"), has(recordDecl(hasName("Z"))),
                 unless(has(recordDecl(hasName("Y")))));
  EXPECT_TRUE(matches("class X { class Z {}; };", ClassXHasNotClassY));
  EXPECT_TRUE(
      notMatches("class X { class Y {}; class Z {}; };", ClassXHasNotClassY));

  DeclarationMatcher NamedNotRecord =
      namedDecl(hasName("Foo"), unless(recordDecl()));
  EXPECT_TRUE(matches("void Foo(){}", NamedNotRecord));
  EXPECT_TRUE(notMatches("struct Foo {};", NamedNotRecord));
}

TEST_P(ASTMatchersTest, HasCastKind) {
  EXPECT_TRUE(
      matches("char *p = 0;",
              traverse(TK_AsIs,
                       varDecl(has(castExpr(hasCastKind(CK_NullToPointer)))))));
  EXPECT_TRUE(notMatches(
      "char *p = 0;",
      traverse(TK_AsIs,
               varDecl(has(castExpr(hasCastKind(CK_DerivedToBase)))))));
  EXPECT_TRUE(matches("char *p = 0;",
                      traverse(TK_AsIs, varDecl(has(implicitCastExpr(
                                            hasCastKind(CK_NullToPointer)))))));
}

TEST_P(ASTMatchersTest, HasDescendant) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `hasDescendant()` that does not depend on C++.
    return;
  }

  DeclarationMatcher ZDescendantClassX =
      recordDecl(hasDescendant(recordDecl(hasName("X"))), hasName("Z"));
  EXPECT_TRUE(matches("class Z { class X {}; };", ZDescendantClassX));
  EXPECT_TRUE(
      matches("class Z { class Y { class X {}; }; };", ZDescendantClassX));
  EXPECT_TRUE(matches("class Z { class A { class Y { class X {}; }; }; };",
                      ZDescendantClassX));
  EXPECT_TRUE(
      matches("class Z { class A { class B { class Y { class X {}; }; }; }; };",
              ZDescendantClassX));
  EXPECT_TRUE(notMatches("class Z {};", ZDescendantClassX));

  DeclarationMatcher ZDescendantClassXHasClassY = recordDecl(
      hasDescendant(recordDecl(has(recordDecl(hasName("Y"))), hasName("X"))),
      hasName("Z"));
  EXPECT_TRUE(matches("class Z { class X { class Y {}; }; };",
                      ZDescendantClassXHasClassY));
  EXPECT_TRUE(
      matches("class Z { class A { class B { class X { class Y {}; }; }; }; };",
              ZDescendantClassXHasClassY));
  EXPECT_TRUE(notMatches("class Z {"
                         "  class A {"
                         "    class B {"
                         "      class X {"
                         "        class C {"
                         "          class Y {};"
                         "        };"
                         "      };"
                         "    }; "
                         "  };"
                         "};",
                         ZDescendantClassXHasClassY));

  DeclarationMatcher ZDescendantClassXDescendantClassY =
      recordDecl(hasDescendant(recordDecl(
                     hasDescendant(recordDecl(hasName("Y"))), hasName("X"))),
                 hasName("Z"));
  EXPECT_TRUE(
      matches("class Z { class A { class X { class B { class Y {}; }; }; }; };",
              ZDescendantClassXDescendantClassY));
  EXPECT_TRUE(matches("class Z {"
                      "  class A {"
                      "    class X {"
                      "      class B {"
                      "        class Y {};"
                      "      };"
                      "      class Y {};"
                      "    };"
                      "  };"
                      "};",
                      ZDescendantClassXDescendantClassY));
}

TEST_P(ASTMatchersTest, HasDescendant_Memoization) {
  DeclarationMatcher CannotMemoize =
      decl(hasDescendant(typeLoc().bind("x")), has(decl()));
  EXPECT_TRUE(matches("void f() { int i; }", CannotMemoize));
}

TEST_P(ASTMatchersTest, HasDescendant_MemoizationUsesRestrictKind) {
  auto Name = hasName("i");
  auto VD = internal::Matcher<VarDecl>(Name).dynCastTo<Decl>();
  auto RD = internal::Matcher<RecordDecl>(Name).dynCastTo<Decl>();
  // Matching VD first should not make a cache hit for RD.
  EXPECT_TRUE(notMatches("void f() { int i; }",
                         decl(hasDescendant(VD), hasDescendant(RD))));
  EXPECT_TRUE(notMatches("void f() { int i; }",
                         decl(hasDescendant(RD), hasDescendant(VD))));
  // Not matching RD first should not make a cache hit for VD either.
  EXPECT_TRUE(matches("void f() { int i; }",
                      decl(anyOf(hasDescendant(RD), hasDescendant(VD)))));
}

TEST_P(ASTMatchersTest, HasAncestor_Memoization) {
  if (!GetParam().isCXX()) {
    return;
  }

  // This triggers an hasAncestor with a TemplateArgument in the bound nodes.
  // That node can't be memoized so we have to check for it before trying to put
  // it on the cache.
  DeclarationMatcher CannotMemoize = classTemplateSpecializationDecl(
      hasAnyTemplateArgument(templateArgument().bind("targ")),
      forEach(fieldDecl(hasAncestor(forStmt()))));

  EXPECT_TRUE(notMatches("template <typename T> struct S;"
                         "template <> struct S<int>{ int i; int j; };",
                         CannotMemoize));
}

TEST_P(ASTMatchersTest, HasAttr) {
  EXPECT_TRUE(matches("struct __attribute__((warn_unused)) X {};",
                      decl(hasAttr(clang::attr::WarnUnused))));
  EXPECT_FALSE(matches("struct X {};", decl(hasAttr(clang::attr::WarnUnused))));
}

TEST_P(ASTMatchersTest, AnyOf) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `anyOf()` that does not depend on C++.
    return;
  }

  DeclarationMatcher YOrZDerivedFromX = cxxRecordDecl(
      anyOf(hasName("Y"), allOf(isDerivedFrom("X"), hasName("Z"))));
  EXPECT_TRUE(matches("class X {}; class Z : public X {};", YOrZDerivedFromX));
  EXPECT_TRUE(matches("class Y {};", YOrZDerivedFromX));
  EXPECT_TRUE(
      notMatches("class X {}; class W : public X {};", YOrZDerivedFromX));
  EXPECT_TRUE(notMatches("class Z {};", YOrZDerivedFromX));

  DeclarationMatcher XOrYOrZOrU =
      recordDecl(anyOf(hasName("X"), hasName("Y"), hasName("Z"), hasName("U")));
  EXPECT_TRUE(matches("class X {};", XOrYOrZOrU));
  EXPECT_TRUE(notMatches("class V {};", XOrYOrZOrU));

  DeclarationMatcher XOrYOrZOrUOrV = recordDecl(anyOf(
      hasName("X"), hasName("Y"), hasName("Z"), hasName("U"), hasName("V")));
  EXPECT_TRUE(matches("class X {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(matches("class Y {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(matches("class Z {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(matches("class U {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(matches("class V {};", XOrYOrZOrUOrV));
  EXPECT_TRUE(notMatches("class A {};", XOrYOrZOrUOrV));

  StatementMatcher MixedTypes = stmt(anyOf(ifStmt(), binaryOperator()));
  EXPECT_TRUE(matches("int F() { return 1 + 2; }", MixedTypes));
  EXPECT_TRUE(matches("int F() { if (true) return 1; }", MixedTypes));
  EXPECT_TRUE(notMatches("int F() { return 1; }", MixedTypes));

  EXPECT_TRUE(
      matches("void f() try { } catch (int) { } catch (...) { }",
              cxxCatchStmt(anyOf(hasDescendant(varDecl()), isCatchAll()))));
}

TEST_P(ASTMatchersTest, MapAnyOf) {
  if (!GetParam().isCXX()) {
    return;
  }

  if (GetParam().hasDelayedTemplateParsing()) {
    return;
  }

  StringRef Code = R"cpp(
void F() {
  if (true) {}
  for ( ; false; ) {}
}
)cpp";

  auto trueExpr = cxxBoolLiteral(equals(true));
  auto falseExpr = cxxBoolLiteral(equals(false));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(ifStmt, forStmt).with(hasCondition(trueExpr)))));
  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(ifStmt, forStmt).with(hasCondition(falseExpr)))));

  EXPECT_TRUE(
      matches(Code, cxxBoolLiteral(equals(true),
                                   hasAncestor(mapAnyOf(ifStmt, forStmt)))));

  EXPECT_TRUE(
      matches(Code, cxxBoolLiteral(equals(false),
                                   hasAncestor(mapAnyOf(ifStmt, forStmt)))));

  EXPECT_TRUE(
      notMatches(Code, floatLiteral(hasAncestor(mapAnyOf(ifStmt, forStmt)))));

  Code = R"cpp(
void func(bool b) {}
struct S {
  S(bool b) {}
};
void F() {
  func(false);
  S s(true);
}
)cpp";
  EXPECT_TRUE(matches(Code, traverse(TK_IgnoreUnlessSpelledInSource,
                                     mapAnyOf(callExpr, cxxConstructExpr)
                                         .with(hasArgument(0, trueExpr)))));
  EXPECT_TRUE(matches(Code, traverse(TK_IgnoreUnlessSpelledInSource,
                                     mapAnyOf(callExpr, cxxConstructExpr)
                                         .with(hasArgument(0, falseExpr)))));

  EXPECT_TRUE(
      matches(Code, traverse(TK_IgnoreUnlessSpelledInSource,
                             mapAnyOf(callExpr, cxxConstructExpr)
                                 .with(hasArgument(0, expr()),
                                       hasDeclaration(functionDecl())))));

  EXPECT_TRUE(matches(Code, traverse(TK_IgnoreUnlessSpelledInSource,
                                     mapAnyOf(callExpr, cxxConstructExpr))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(callExpr, cxxConstructExpr).bind("call"))));

  Code = R"cpp(
struct HasOpNeqMem
{
    bool operator!=(const HasOpNeqMem& other) const
    {
        return true;
    }
};
struct HasOpFree
{
};
bool operator!=(const HasOpFree& lhs, const HasOpFree& rhs)
{
    return true;
}

void binop()
{
    int s1;
    int s2;
    if (s1 != s2)
        return;
}

void opMem()
{
    HasOpNeqMem s1;
    HasOpNeqMem s2;
    if (s1 != s2)
        return;
}

void opFree()
{
    HasOpFree s1;
    HasOpFree s2;
    if (s1 != s2)
        return;
}

template<typename T>
void templ()
{
    T s1;
    T s2;
    if (s1 != s2)
        return;
}
)cpp";

  EXPECT_TRUE(matches(
      Code,
      traverse(TK_IgnoreUnlessSpelledInSource,
               mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                   .with(hasOperatorName("!="),
                         forFunction(functionDecl(hasName("binop"))),
                         hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                         hasRHS(declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code,
      traverse(TK_IgnoreUnlessSpelledInSource,
               mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                   .with(hasOperatorName("!="),
                         forFunction(functionDecl(hasName("opMem"))),
                         hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                         hasRHS(declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code,
      traverse(TK_IgnoreUnlessSpelledInSource,
               mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                   .with(hasOperatorName("!="),
                         forFunction(functionDecl(hasName("opFree"))),
                         hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                         hasRHS(declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                         .with(hasOperatorName("!="),
                               forFunction(functionDecl(hasName("binop"))),
                               hasEitherOperand(
                                   declRefExpr(to(varDecl(hasName("s1"))))),
                               hasEitherOperand(
                                   declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                         .with(hasOperatorName("!="),
                               forFunction(functionDecl(hasName("opMem"))),
                               hasEitherOperand(
                                   declRefExpr(to(varDecl(hasName("s1"))))),
                               hasEitherOperand(
                                   declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                         .with(hasOperatorName("!="),
                               forFunction(functionDecl(hasName("opFree"))),
                               hasEitherOperand(
                                   declRefExpr(to(varDecl(hasName("s1"))))),
                               hasEitherOperand(
                                   declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code,
      traverse(
          TK_IgnoreUnlessSpelledInSource,
          mapAnyOf(binaryOperator, cxxOperatorCallExpr)
              .with(hasOperatorName("!="),
                    forFunction(functionDecl(hasName("binop"))),
                    hasOperands(declRefExpr(to(varDecl(hasName("s1")))),
                                declRefExpr(to(varDecl(hasName("s2"))))),
                    hasOperands(declRefExpr(to(varDecl(hasName("s2")))),
                                declRefExpr(to(varDecl(hasName("s1")))))))));

  EXPECT_TRUE(matches(
      Code,
      traverse(
          TK_IgnoreUnlessSpelledInSource,
          mapAnyOf(binaryOperator, cxxOperatorCallExpr)
              .with(hasOperatorName("!="),
                    forFunction(functionDecl(hasName("opMem"))),
                    hasOperands(declRefExpr(to(varDecl(hasName("s1")))),
                                declRefExpr(to(varDecl(hasName("s2"))))),
                    hasOperands(declRefExpr(to(varDecl(hasName("s2")))),
                                declRefExpr(to(varDecl(hasName("s1")))))))));

  EXPECT_TRUE(matches(
      Code,
      traverse(
          TK_IgnoreUnlessSpelledInSource,
          mapAnyOf(binaryOperator, cxxOperatorCallExpr)
              .with(hasOperatorName("!="),
                    forFunction(functionDecl(hasName("opFree"))),
                    hasOperands(declRefExpr(to(varDecl(hasName("s1")))),
                                declRefExpr(to(varDecl(hasName("s2"))))),
                    hasOperands(declRefExpr(to(varDecl(hasName("s2")))),
                                declRefExpr(to(varDecl(hasName("s1")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                         .with(hasAnyOperatorName("==", "!="),
                               forFunction(functionDecl(hasName("binop")))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                         .with(hasAnyOperatorName("==", "!="),
                               forFunction(functionDecl(hasName("opMem")))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                         .with(hasAnyOperatorName("==", "!="),
                               forFunction(functionDecl(hasName("opFree")))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     binaryOperation(
                         hasOperatorName("!="),
                         forFunction(functionDecl(hasName("binop"))),
                         hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                         hasRHS(declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     binaryOperation(
                         hasOperatorName("!="),
                         forFunction(functionDecl(hasName("opMem"))),
                         hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                         hasRHS(declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     binaryOperation(
                         hasOperatorName("!="),
                         forFunction(functionDecl(hasName("opFree"))),
                         hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                         hasRHS(declRefExpr(to(varDecl(hasName("s2")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     binaryOperation(
                         hasOperatorName("!="),
                         forFunction(functionDecl(hasName("templ"))),
                         hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                         hasRHS(declRefExpr(to(varDecl(hasName("s2")))))))));

  Code = R"cpp(
struct HasOpEq
{
    bool operator==(const HasOpEq &) const;
};

void inverse()
{
    HasOpEq s1;
    HasOpEq s2;
    if (s1 != s2)
        return;
}

namespace std {
struct strong_ordering {
  int n;
  constexpr operator int() const { return n; }
  static const strong_ordering equal, greater, less;
};
constexpr strong_ordering strong_ordering::equal = {0};
constexpr strong_ordering strong_ordering::greater = {1};
constexpr strong_ordering strong_ordering::less = {-1};
}

struct HasSpaceshipMem {
  int a;
  constexpr auto operator<=>(const HasSpaceshipMem&) const = default;
};

void rewritten()
{
    HasSpaceshipMem s1;
    HasSpaceshipMem s2;
    if (s1 != s2)
        return;
}
)cpp";

  EXPECT_TRUE(matchesConditionally(
      Code,
      traverse(
          TK_IgnoreUnlessSpelledInSource,
          binaryOperation(hasOperatorName("!="),
                          forFunction(functionDecl(hasName("inverse"))),
                          hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                          hasRHS(declRefExpr(to(varDecl(hasName("s2"))))))),
      true, {"-std=c++20"}));

  EXPECT_TRUE(matchesConditionally(
      Code,
      traverse(
          TK_IgnoreUnlessSpelledInSource,
          binaryOperation(hasOperatorName("!="),
                          forFunction(functionDecl(hasName("rewritten"))),
                          hasLHS(declRefExpr(to(varDecl(hasName("s1"))))),
                          hasRHS(declRefExpr(to(varDecl(hasName("s2"))))))),
      true, {"-std=c++20"}));

  Code = R"cpp(
struct HasOpBangMem
{
    bool operator!() const
    {
        return false;
    }
};
struct HasOpBangFree
{
};
bool operator!(HasOpBangFree const&)
{
    return false;
}

void unop()
{
    int s1;
    if (!s1)
        return;
}

void opMem()
{
    HasOpBangMem s1;
    if (!s1)
        return;
}

void opFree()
{
    HasOpBangFree s1;
    if (!s1)
        return;
}
)cpp";

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(unaryOperator, cxxOperatorCallExpr)
                         .with(hasOperatorName("!"),
                               forFunction(functionDecl(hasName("unop"))),
                               hasUnaryOperand(
                                   declRefExpr(to(varDecl(hasName("s1")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(unaryOperator, cxxOperatorCallExpr)
                         .with(hasOperatorName("!"),
                               forFunction(functionDecl(hasName("opMem"))),
                               hasUnaryOperand(
                                   declRefExpr(to(varDecl(hasName("s1")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(unaryOperator, cxxOperatorCallExpr)
                         .with(hasOperatorName("!"),
                               forFunction(functionDecl(hasName("opFree"))),
                               hasUnaryOperand(
                                   declRefExpr(to(varDecl(hasName("s1")))))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(unaryOperator, cxxOperatorCallExpr)
                         .with(hasAnyOperatorName("+", "!"),
                               forFunction(functionDecl(hasName("unop")))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(unaryOperator, cxxOperatorCallExpr)
                         .with(hasAnyOperatorName("+", "!"),
                               forFunction(functionDecl(hasName("opMem")))))));

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     mapAnyOf(unaryOperator, cxxOperatorCallExpr)
                         .with(hasAnyOperatorName("+", "!"),
                               forFunction(functionDecl(hasName("opFree")))))));

  Code = R"cpp(
struct ConstructorTakesInt
{
  ConstructorTakesInt(int i) {}
};

void callTakesInt(int i)
{

}

void doCall()
{
  callTakesInt(42);
}

void doConstruct()
{
  ConstructorTakesInt cti(42);
}
)cpp";

  EXPECT_TRUE(matches(
      Code, traverse(TK_IgnoreUnlessSpelledInSource,
                     invocation(forFunction(functionDecl(hasName("doCall"))),
                                hasArgument(0, integerLiteral(equals(42))),
                                hasAnyArgument(integerLiteral(equals(42))),
                                forEachArgumentWithParam(
                                    integerLiteral(equals(42)),
                                    parmVarDecl(hasName("i")))))));

  EXPECT_TRUE(matches(
      Code,
      traverse(
          TK_IgnoreUnlessSpelledInSource,
          invocation(forFunction(functionDecl(hasName("doConstruct"))),
                     hasArgument(0, integerLiteral(equals(42))),
                     hasAnyArgument(integerLiteral(equals(42))),
                     forEachArgumentWithParam(integerLiteral(equals(42)),
                                              parmVarDecl(hasName("i")))))));
}

TEST_P(ASTMatchersTest, IsDerivedFrom) {
  if (!GetParam().isCXX()) {
    return;
  }

  DeclarationMatcher IsDerivedFromX = cxxRecordDecl(isDerivedFrom("X"));

  EXPECT_TRUE(matches("class X {}; class Y : public X {};", IsDerivedFromX));
  EXPECT_TRUE(notMatches("class X {};", IsDerivedFromX));
  EXPECT_TRUE(notMatches("class X;", IsDerivedFromX));
  EXPECT_TRUE(notMatches("class Y;", IsDerivedFromX));
  EXPECT_TRUE(notMatches("", IsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; template<int N> class Y : Y<N-1>, X {};",
                      IsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; template<int N> class Y : X, Y<N-1> {};",
                      IsDerivedFromX));

  DeclarationMatcher IsZDerivedFromX =
      cxxRecordDecl(hasName("Z"), isDerivedFrom("X"));
  EXPECT_TRUE(matches("class X {};"
                      "template<int N> class Y : Y<N-1> {};"
                      "template<> class Y<0> : X {};"
                      "class Z : Y<1> {};",
                      IsZDerivedFromX));

  DeclarationMatcher IsDirectlyDerivedFromX =
      cxxRecordDecl(isDirectlyDerivedFrom("X"));

  EXPECT_TRUE(
      matches("class X {}; class Y : public X {};", IsDirectlyDerivedFromX));
  EXPECT_TRUE(notMatches("class X {};", IsDirectlyDerivedFromX));
  EXPECT_TRUE(notMatches("class X;", IsDirectlyDerivedFromX));
  EXPECT_TRUE(notMatches("class Y;", IsDirectlyDerivedFromX));
  EXPECT_TRUE(notMatches("", IsDirectlyDerivedFromX));

  DeclarationMatcher IsAX = cxxRecordDecl(isSameOrDerivedFrom("X"));

  EXPECT_TRUE(matches("class X {}; class Y : public X {};", IsAX));
  EXPECT_TRUE(matches("class X {};", IsAX));
  EXPECT_TRUE(matches("class X;", IsAX));
  EXPECT_TRUE(notMatches("class Y;", IsAX));
  EXPECT_TRUE(notMatches("", IsAX));

  DeclarationMatcher ZIsDerivedFromX =
      cxxRecordDecl(hasName("Z"), isDerivedFrom("X"));
  DeclarationMatcher ZIsDirectlyDerivedFromX =
      cxxRecordDecl(hasName("Z"), isDirectlyDerivedFrom("X"));
  EXPECT_TRUE(
      matches("class X {}; class Y : public X {}; class Z : public Y {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("class X {}; class Y : public X {}; class Z : public Y {};",
                 ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(matches("class X {};"
                      "template<class T> class Y : public X {};"
                      "class Z : public Y<int> {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(notMatches("class X {};"
                         "template<class T> class Y : public X {};"
                         "class Z : public Y<int> {};",
                         ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(matches("class X {}; template<class T> class Z : public X {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("template<class T> class X {}; "
                      "template<class T> class Z : public X<T> {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("template<class T, class U=T> class X {}; "
                      "template<class T> class Z : public X<T> {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class X> class A { class Z : public X {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class X> class A { public: class Z : public X {}; }; "
              "class X{}; void y() { A<X>::Z z; }",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template <class T> class X {}; "
              "template<class Y> class A { class Z : public X<Y> {}; };",
              ZIsDerivedFromX));
  EXPECT_TRUE(notMatches("template<template<class T> class X> class A { "
                         "  class Z : public X<int> {}; };",
                         ZIsDerivedFromX));
  EXPECT_TRUE(matches("template<template<class T> class X> class A { "
                      "  public: class Z : public X<int> {}; }; "
                      "template<class T> class X {}; void y() { A<X>::Z z; }",
                      ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class X> class A { class Z : public X::D {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(matches("template<class X> class A { public: "
                      "  class Z : public X::D {}; }; "
                      "class Y { public: class X {}; typedef X D; }; "
                      "void y() { A<Y>::Z z; }",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; typedef X Y; class Z : public Y {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("template<class T> class Y { typedef typename T::U X; "
                      "  class Z : public X {}; };",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; class Z : public ::X {};", ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class T> class X {}; "
                 "template<class T> class A { class Z : public X<T>::D {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("template<class T> class X { public: typedef X<T> D; }; "
              "template<class T> class A { public: "
              "  class Z : public X<T>::D {}; }; void y() { A<int>::Z z; }",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class X> class A { class Z : public X::D::E {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(
      matches("class X {}; typedef X V; typedef V W; class Z : public W {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; class Y : public X {}; "
                      "typedef Y V; typedef V W; class Z : public W {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(notMatches("class X {}; class Y : public X {}; "
                         "typedef Y V; typedef V W; class Z : public W {};",
                         ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(
      matches("template<class T, class U> class X {}; "
              "template<class T> class A { class Z : public X<T, int> {}; };",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<class X> class D { typedef X A; typedef A B; "
                 "  typedef B C; class Z : public C {}; };",
                 ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; typedef X A; typedef A B; "
                      "class Z : public B {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; typedef X A; typedef A B; typedef B C; "
                      "class Z : public C {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("class U {}; typedef U X; typedef X V; "
                      "class Z : public V {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("class Base {}; typedef Base X; "
                      "class Z : public Base {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(matches("class Base {}; typedef Base Base2; typedef Base2 X; "
                      "class Z : public Base {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(notMatches("class Base {}; class Base2 {}; typedef Base2 X; "
                         "class Z : public Base {};",
                         ZIsDerivedFromX));
  EXPECT_TRUE(matches("class A {}; typedef A X; typedef A Y; "
                      "class Z : public Y {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(notMatches("template <typename T> class Z;"
                         "template <> class Z<void> {};"
                         "template <typename T> class Z : public Z<void> {};",
                         IsDerivedFromX));
  EXPECT_TRUE(matches("template <typename T> class X;"
                      "template <> class X<void> {};"
                      "template <typename T> class X : public X<void> {};",
                      IsDerivedFromX));
  EXPECT_TRUE(
      matches("class X {};"
              "template <typename T> class Z;"
              "template <> class Z<void> {};"
              "template <typename T> class Z : public Z<void>, public X {};",
              ZIsDerivedFromX));
  EXPECT_TRUE(
      notMatches("template<int> struct X;"
                 "template<int i> struct X : public X<i-1> {};",
                 cxxRecordDecl(isDerivedFrom(recordDecl(hasName("Some"))))));
  EXPECT_TRUE(matches(
      "struct A {};"
      "template<int> struct X;"
      "template<int i> struct X : public X<i-1> {};"
      "template<> struct X<0> : public A {};"
      "struct B : public X<42> {};",
      cxxRecordDecl(hasName("B"), isDerivedFrom(recordDecl(hasName("A"))))));
  EXPECT_TRUE(notMatches(
      "struct A {};"
      "template<int> struct X;"
      "template<int i> struct X : public X<i-1> {};"
      "template<> struct X<0> : public A {};"
      "struct B : public X<42> {};",
      cxxRecordDecl(hasName("B"),
                    isDirectlyDerivedFrom(recordDecl(hasName("A"))))));

  // FIXME: Once we have better matchers for template type matching,
  // get rid of the Variable(...) matching and match the right template
  // declarations directly.
  const char *RecursiveTemplateOneParameter =
      "class Base1 {}; class Base2 {};"
      "template <typename T> class Z;"
      "template <> class Z<void> : public Base1 {};"
      "template <> class Z<int> : public Base2 {};"
      "template <> class Z<float> : public Z<void> {};"
      "template <> class Z<double> : public Z<int> {};"
      "template <typename T> class Z : public Z<float>, public Z<double> {};"
      "void f() { Z<float> z_float; Z<double> z_double; Z<char> z_char; }";
  EXPECT_TRUE(matches(
      RecursiveTemplateOneParameter,
      varDecl(hasName("z_float"),
              hasInitializer(hasType(cxxRecordDecl(isDerivedFrom("Base1")))))));
  EXPECT_TRUE(notMatches(
      RecursiveTemplateOneParameter,
      varDecl(hasName("z_float"),
              hasInitializer(hasType(cxxRecordDecl(isDerivedFrom("Base2")))))));
  EXPECT_TRUE(
      matches(RecursiveTemplateOneParameter,
              varDecl(hasName("z_char"),
                      hasInitializer(hasType(cxxRecordDecl(
                          isDerivedFrom("Base1"), isDerivedFrom("Base2")))))));

  const char *RecursiveTemplateTwoParameters =
      "class Base1 {}; class Base2 {};"
      "template <typename T1, typename T2> class Z;"
      "template <typename T> class Z<void, T> : public Base1 {};"
      "template <typename T> class Z<int, T> : public Base2 {};"
      "template <typename T> class Z<float, T> : public Z<void, T> {};"
      "template <typename T> class Z<double, T> : public Z<int, T> {};"
      "template <typename T1, typename T2> class Z : "
      "    public Z<float, T2>, public Z<double, T2> {};"
      "void f() { Z<float, void> z_float; Z<double, void> z_double; "
      "           Z<char, void> z_char; }";
  EXPECT_TRUE(matches(
      RecursiveTemplateTwoParameters,
      varDecl(hasName("z_float"),
              hasInitializer(hasType(cxxRecordDecl(isDerivedFrom("Base1")))))));
  EXPECT_TRUE(notMatches(
      RecursiveTemplateTwoParameters,
      varDecl(hasName("z_float"),
              hasInitializer(hasType(cxxRecordDecl(isDerivedFrom("Base2")))))));
  EXPECT_TRUE(
      matches(RecursiveTemplateTwoParameters,
              varDecl(hasName("z_char"),
                      hasInitializer(hasType(cxxRecordDecl(
                          isDerivedFrom("Base1"), isDerivedFrom("Base2")))))));
  EXPECT_TRUE(matches("namespace ns { class X {}; class Y : public X {}; }",
                      cxxRecordDecl(isDerivedFrom("::ns::X"))));
  EXPECT_TRUE(notMatches("class X {}; class Y : public X {};",
                         cxxRecordDecl(isDerivedFrom("::ns::X"))));

  EXPECT_TRUE(matches(
      "class X {}; class Y : public X {};",
      cxxRecordDecl(isDerivedFrom(recordDecl(hasName("X")).bind("test")))));

  EXPECT_TRUE(matches("template<typename T> class X {};"
                      "template<typename T> using Z = X<T>;"
                      "template <typename T> class Y : Z<T> {};",
                      cxxRecordDecl(isDerivedFrom(namedDecl(hasName("X"))))));
}

TEST_P(ASTMatchersTest, IsDerivedFrom_EmptyName) {
  if (!GetParam().isCXX()) {
    return;
  }

  const char *const Code = "class X {}; class Y : public X {};";
  EXPECT_TRUE(notMatches(Code, cxxRecordDecl(isDerivedFrom(""))));
  EXPECT_TRUE(notMatches(Code, cxxRecordDecl(isDirectlyDerivedFrom(""))));
  EXPECT_TRUE(notMatches(Code, cxxRecordDecl(isSameOrDerivedFrom(""))));
}

TEST_P(ASTMatchersTest, IsDerivedFrom_ObjC) {
  DeclarationMatcher IsDerivedFromX = objcInterfaceDecl(isDerivedFrom("X"));
  EXPECT_TRUE(
      matchesObjC("@interface X @end @interface Y : X @end", IsDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end @interface Y<__covariant ObjectType> : X @end",
      IsDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end @compatibility_alias Y X; @interface Z : Y @end",
      IsDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end typedef X Y; @interface Z : Y @end", IsDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@interface X @end", IsDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@class X;", IsDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@class Y;", IsDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@interface X @end @compatibility_alias Y X;",
                             IsDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@interface X @end typedef X Y;", IsDerivedFromX));

  DeclarationMatcher IsDirectlyDerivedFromX =
      objcInterfaceDecl(isDirectlyDerivedFrom("X"));
  EXPECT_TRUE(matchesObjC("@interface X @end @interface Y : X @end",
                          IsDirectlyDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end @interface Y<__covariant ObjectType> : X @end",
      IsDirectlyDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end @compatibility_alias Y X; @interface Z : Y @end",
      IsDirectlyDerivedFromX));
  EXPECT_TRUE(
      matchesObjC("@interface X @end typedef X Y; @interface Z : Y @end",
                  IsDirectlyDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@interface X @end", IsDirectlyDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@class X;", IsDirectlyDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@class Y;", IsDirectlyDerivedFromX));
  EXPECT_TRUE(notMatchesObjC("@interface X @end @compatibility_alias Y X;",
                             IsDirectlyDerivedFromX));
  EXPECT_TRUE(
      notMatchesObjC("@interface X @end typedef X Y;", IsDirectlyDerivedFromX));

  DeclarationMatcher IsAX = objcInterfaceDecl(isSameOrDerivedFrom("X"));
  EXPECT_TRUE(matchesObjC("@interface X @end @interface Y : X @end", IsAX));
  EXPECT_TRUE(matchesObjC("@interface X @end", IsAX));
  EXPECT_TRUE(matchesObjC("@class X;", IsAX));
  EXPECT_TRUE(notMatchesObjC("@interface Y @end", IsAX));
  EXPECT_TRUE(notMatchesObjC("@class Y;", IsAX));

  DeclarationMatcher ZIsDerivedFromX =
      objcInterfaceDecl(hasName("Z"), isDerivedFrom("X"));
  DeclarationMatcher ZIsDirectlyDerivedFromX =
      objcInterfaceDecl(hasName("Z"), isDirectlyDerivedFrom("X"));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end @interface Y : X @end @interface Z : Y @end",
      ZIsDerivedFromX));
  EXPECT_TRUE(matchesObjC("@interface X @end @interface Y : X @end typedef Y "
                          "V; typedef V W; @interface Z : W @end",
                          ZIsDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end typedef X Y; @interface Z : Y @end", ZIsDerivedFromX));
  EXPECT_TRUE(
      matchesObjC("@interface X @end typedef X Y; @interface Z : Y @end",
                  ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface A @end typedef A X; typedef A Y; @interface Z : Y @end",
      ZIsDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface A @end typedef A X; typedef A Y; @interface Z : Y @end",
      ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end @compatibility_alias Y X; @interface Z : Y @end",
      ZIsDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface X @end @compatibility_alias Y X; @interface Z : Y @end",
      ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface Y @end @compatibility_alias X Y; @interface Z : Y @end",
      ZIsDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface Y @end @compatibility_alias X Y; @interface Z : Y @end",
      ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface A @end @compatibility_alias X A; @compatibility_alias Y A;"
      "@interface Z : Y @end",
      ZIsDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface A @end @compatibility_alias X A; @compatibility_alias Y A;"
      "@interface Z : Y @end",
      ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(matchesObjC(
      "@interface Y @end typedef Y X; @interface Z : X @end", ZIsDerivedFromX));
  EXPECT_TRUE(
      matchesObjC("@interface Y @end typedef Y X; @interface Z : X @end",
                  ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(
      matchesObjC("@interface A @end @compatibility_alias Y A; typedef Y X;"
                  "@interface Z : A @end",
                  ZIsDerivedFromX));
  EXPECT_TRUE(
      matchesObjC("@interface A @end @compatibility_alias Y A; typedef Y X;"
                  "@interface Z : A @end",
                  ZIsDirectlyDerivedFromX));
  EXPECT_TRUE(
      matchesObjC("@interface A @end typedef A Y; @compatibility_alias X Y;"
                  "@interface Z : A @end",
                  ZIsDerivedFromX));
  EXPECT_TRUE(
      matchesObjC("@interface A @end typedef A Y; @compatibility_alias X Y;"
                  "@interface Z : A @end",
                  ZIsDirectlyDerivedFromX));
}

TEST_P(ASTMatchersTest, IsLambda) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  const auto IsLambda = cxxMethodDecl(ofClass(cxxRecordDecl(isLambda())));
  EXPECT_TRUE(matches("auto x = []{};", IsLambda));
  EXPECT_TRUE(notMatches("struct S { void operator()() const; };", IsLambda));
}

TEST_P(ASTMatchersTest, Bind) {
  DeclarationMatcher ClassX = has(recordDecl(hasName("::X")).bind("x"));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X {};", ClassX,
      std::make_unique<VerifyIdIsBoundTo<CXXRecordDecl>>("x")));

  EXPECT_TRUE(matchAndVerifyResultFalse(
      "class X {};", ClassX,
      std::make_unique<VerifyIdIsBoundTo<CXXRecordDecl>>("other-id")));

  TypeMatcher TypeAHasClassB = hasDeclaration(
      recordDecl(hasName("A"), has(recordDecl(hasName("B")).bind("b"))));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { public: A *a; class B {}; };", TypeAHasClassB,
      std::make_unique<VerifyIdIsBoundTo<Decl>>("b")));

  StatementMatcher MethodX =
      callExpr(callee(cxxMethodDecl(hasName("x")))).bind("x");

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { void x() { x(); } };", MethodX,
      std::make_unique<VerifyIdIsBoundTo<CXXMemberCallExpr>>("x")));
}

TEST_P(ASTMatchersTest, Bind_SameNameInAlternatives) {
  StatementMatcher matcher = anyOf(
      binaryOperator(hasOperatorName("+"), hasLHS(expr().bind("x")),
                     hasRHS(integerLiteral(equals(0)))),
      binaryOperator(hasOperatorName("+"), hasLHS(integerLiteral(equals(0))),
                     hasRHS(expr().bind("x"))));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      // The first branch of the matcher binds x to 0 but then fails.
      // The second branch binds x to f() and succeeds.
      "int f() { return 0 + f(); }", matcher,
      std::make_unique<VerifyIdIsBoundTo<CallExpr>>("x")));
}

TEST_P(ASTMatchersTest, Bind_BindsIDForMemoizedResults) {
  // Using the same matcher in two match expressions will make memoization
  // kick in.
  DeclarationMatcher ClassX = recordDecl(hasName("X")).bind("x");
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { class B { class X {}; }; };",
      DeclarationMatcher(
          anyOf(recordDecl(hasName("A"), hasDescendant(ClassX)),
                recordDecl(hasName("B"), hasDescendant(ClassX)))),
      std::make_unique<VerifyIdIsBoundTo<Decl>>("x", 2)));
}

TEST_P(ASTMatchersTest, HasType_MatchesAsString) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `hasType()` that does not depend on C++.
    return;
  }

  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() {Y* y; y->x(); }",
              cxxMemberCallExpr(on(hasType(asString("class Y *"))))));
  EXPECT_TRUE(
      matches("class X { void x(int x) {} };",
              cxxMethodDecl(hasParameter(0, hasType(asString("int"))))));
  EXPECT_TRUE(matches("namespace ns { struct A {}; }  struct B { ns::A a; };",
                      fieldDecl(hasType(asString("ns::A")))));
  EXPECT_TRUE(
      matches("namespace { struct A {}; }  struct B { A a; };",
              fieldDecl(hasType(asString("struct (anonymous namespace)::A")))));
}

TEST_P(ASTMatchersTest, HasOverloadedOperatorName) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher OpCallAndAnd =
      cxxOperatorCallExpr(hasOverloadedOperatorName("&&"));
  EXPECT_TRUE(matches("class Y { }; "
                      "bool operator&&(Y x, Y y) { return true; }; "
                      "Y a; Y b; bool c = a && b;",
                      OpCallAndAnd));
  StatementMatcher OpCallLessLess =
      cxxOperatorCallExpr(hasOverloadedOperatorName("<<"));
  EXPECT_TRUE(notMatches("class Y { }; "
                         "bool operator&&(Y x, Y y) { return true; }; "
                         "Y a; Y b; bool c = a && b;",
                         OpCallLessLess));
  StatementMatcher OpStarCall =
      cxxOperatorCallExpr(hasOverloadedOperatorName("*"));
  EXPECT_TRUE(
      matches("class Y; int operator*(Y &); void f(Y &y) { *y; }", OpStarCall));
  DeclarationMatcher ClassWithOpStar =
      cxxRecordDecl(hasMethod(hasOverloadedOperatorName("*")));
  EXPECT_TRUE(matches("class Y { int operator*(); };", ClassWithOpStar));
  EXPECT_TRUE(notMatches("class Y { void myOperator(); };", ClassWithOpStar));
  DeclarationMatcher AnyOpStar = functionDecl(hasOverloadedOperatorName("*"));
  EXPECT_TRUE(matches("class Y; int operator*(Y &);", AnyOpStar));
  EXPECT_TRUE(matches("class Y { int operator*(); };", AnyOpStar));
  DeclarationMatcher AnyAndOp =
      functionDecl(hasAnyOverloadedOperatorName("&", "&&"));
  EXPECT_TRUE(matches("class Y; Y operator&(Y &, Y &);", AnyAndOp));
  EXPECT_TRUE(matches("class Y; Y operator&&(Y &, Y &);", AnyAndOp));
  EXPECT_TRUE(matches("class Y { Y operator&(Y &); };", AnyAndOp));
  EXPECT_TRUE(matches("class Y { Y operator&&(Y &); };", AnyAndOp));
}

TEST_P(ASTMatchersTest, HasOverloadedOperatorName_MatchesNestedCalls) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class Y { }; "
      "Y& operator&&(Y& x, Y& y) { return x; }; "
      "Y a; Y b; Y c; Y d = a && b && c;",
      cxxOperatorCallExpr(hasOverloadedOperatorName("&&")).bind("x"),
      std::make_unique<VerifyIdIsBoundTo<CXXOperatorCallExpr>>("x", 2)));
  EXPECT_TRUE(matches("class Y { }; "
                      "Y& operator&&(Y& x, Y& y) { return x; }; "
                      "Y a; Y b; Y c; Y d = a && b && c;",
                      cxxOperatorCallExpr(hasParent(cxxOperatorCallExpr()))));
  EXPECT_TRUE(
      matches("class Y { }; "
              "Y& operator&&(Y& x, Y& y) { return x; }; "
              "Y a; Y b; Y c; Y d = a && b && c;",
              cxxOperatorCallExpr(hasDescendant(cxxOperatorCallExpr()))));
}

TEST_P(ASTMatchersTest, HasLocalStorage) {
  auto M = varDecl(hasName("X"), hasLocalStorage());
  EXPECT_TRUE(matches("void f() { int X; }", M));
  EXPECT_TRUE(notMatches("int X;", M));
  EXPECT_TRUE(notMatches("void f() { static int X; }", M));
}

TEST_P(ASTMatchersTest, HasGlobalStorage) {
  auto M = varDecl(hasName("X"), hasGlobalStorage());
  EXPECT_TRUE(notMatches("void f() { int X; }", M));
  EXPECT_TRUE(matches("int X;", M));
  EXPECT_TRUE(matches("void f() { static int X; }", M));
}

TEST_P(ASTMatchersTest, IsStaticLocal) {
  auto M = varDecl(isStaticLocal());
  EXPECT_TRUE(matches("void f() { static int X; }", M));
  EXPECT_TRUE(notMatches("static int X;", M));
  EXPECT_TRUE(notMatches("void f() { int X; }", M));
  EXPECT_TRUE(notMatches("int X;", M));
}

TEST_P(ASTMatchersTest, StorageDuration) {
  StringRef T =
      "void f() { int x; static int y; } int a;static int b;extern int c;";

  EXPECT_TRUE(matches(T, varDecl(hasName("x"), hasAutomaticStorageDuration())));
  EXPECT_TRUE(
      notMatches(T, varDecl(hasName("y"), hasAutomaticStorageDuration())));
  EXPECT_TRUE(
      notMatches(T, varDecl(hasName("a"), hasAutomaticStorageDuration())));

  EXPECT_TRUE(matches(T, varDecl(hasName("y"), hasStaticStorageDuration())));
  EXPECT_TRUE(matches(T, varDecl(hasName("a"), hasStaticStorageDuration())));
  EXPECT_TRUE(matches(T, varDecl(hasName("b"), hasStaticStorageDuration())));
  EXPECT_TRUE(matches(T, varDecl(hasName("c"), hasStaticStorageDuration())));
  EXPECT_TRUE(notMatches(T, varDecl(hasName("x"), hasStaticStorageDuration())));

  // FIXME: Add thread_local variables to the source code snippet.
  EXPECT_TRUE(notMatches(T, varDecl(hasName("x"), hasThreadStorageDuration())));
  EXPECT_TRUE(notMatches(T, varDecl(hasName("y"), hasThreadStorageDuration())));
  EXPECT_TRUE(notMatches(T, varDecl(hasName("a"), hasThreadStorageDuration())));
}

TEST_P(ASTMatchersTest, VarDecl_MatchesFunctionParameter) {
  EXPECT_TRUE(matches("void f(int i) {}", varDecl(hasName("i"))));
}

TEST_P(ASTMatchersTest, SizeOfExpr_MatchesCorrectType) {
  EXPECT_TRUE(matches("void x() { int a = sizeof(a); }",
                      sizeOfExpr(hasArgumentOfType(asString("int")))));
  EXPECT_TRUE(notMatches("void x() { int a = sizeof(a); }",
                         sizeOfExpr(hasArgumentOfType(asString("float")))));
  EXPECT_TRUE(matches(
      "struct A {}; void x() { struct A a; int b = sizeof(a); }",
      sizeOfExpr(hasArgumentOfType(hasDeclaration(recordDecl(hasName("A")))))));
  EXPECT_TRUE(notMatches("void x() { int a = sizeof(a); }",
                         sizeOfExpr(hasArgumentOfType(
                             hasDeclaration(recordDecl(hasName("string")))))));
}

TEST_P(ASTMatchersTest, IsInteger_MatchesIntegers) {
  EXPECT_TRUE(matches("int i = 0;", varDecl(hasType(isInteger()))));
  EXPECT_TRUE(
      matches("long long i = 0; void f(long long) { }; void g() {f(i);}",
              callExpr(hasArgument(
                  0, declRefExpr(to(varDecl(hasType(isInteger()))))))));
}

TEST_P(ASTMatchersTest, IsInteger_ReportsNoFalsePositives) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a similar negative test for `isInteger()` that does not depend
    // on C++.
    return;
  }

  EXPECT_TRUE(notMatches("int *i;", varDecl(hasType(isInteger()))));
  EXPECT_TRUE(
      notMatches("struct T {}; T t; void f(T *) { }; void g() {f(&t);}",
                 callExpr(hasArgument(
                     0, declRefExpr(to(varDecl(hasType(isInteger()))))))));
}

TEST_P(ASTMatchersTest, IsSignedInteger_MatchesSignedIntegers) {
  EXPECT_TRUE(matches("int i = 0;", varDecl(hasType(isSignedInteger()))));
  EXPECT_TRUE(
      notMatches("unsigned i = 0;", varDecl(hasType(isSignedInteger()))));
}

TEST_P(ASTMatchersTest, IsUnsignedInteger_MatchesUnsignedIntegers) {
  EXPECT_TRUE(notMatches("int i = 0;", varDecl(hasType(isUnsignedInteger()))));
  EXPECT_TRUE(
      matches("unsigned i = 0;", varDecl(hasType(isUnsignedInteger()))));
}

TEST_P(ASTMatchersTest, IsAnyPointer_MatchesPointers) {
  if (!GetParam().isCXX11OrLater()) {
    // FIXME: Add a test for `isAnyPointer()` that does not depend on C++.
    return;
  }

  EXPECT_TRUE(matches("int* i = nullptr;", varDecl(hasType(isAnyPointer()))));
}

TEST_P(ASTMatchersTest, IsAnyPointer_MatchesObjcPointer) {
  EXPECT_TRUE(matchesObjC("@interface Foo @end Foo *f;",
                          varDecl(hasType(isAnyPointer()))));
}

TEST_P(ASTMatchersTest, IsAnyPointer_ReportsNoFalsePositives) {
  EXPECT_TRUE(notMatches("int i = 0;", varDecl(hasType(isAnyPointer()))));
}

TEST_P(ASTMatchersTest, IsAnyCharacter_MatchesCharacters) {
  EXPECT_TRUE(matches("char i = 0;", varDecl(hasType(isAnyCharacter()))));
}

TEST_P(ASTMatchersTest, IsAnyCharacter_ReportsNoFalsePositives) {
  EXPECT_TRUE(notMatches("int i;", varDecl(hasType(isAnyCharacter()))));
}

TEST_P(ASTMatchersTest, IsArrow_MatchesMemberVariablesViaArrow) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `isArrow()` that does not depend on C++.
    return;
  }
  if (GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("class Y { void x() { this->y; } int y; };",
                      memberExpr(isArrow())));
  EXPECT_TRUE(
      matches("class Y { void x() { y; } int y; };", memberExpr(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { (*this).y; } int y; };",
                         memberExpr(isArrow())));
  EXPECT_TRUE(matches("template <class T> class Y { void x() { this->m; } };",
                      cxxDependentScopeMemberExpr(isArrow())));
  EXPECT_TRUE(
      notMatches("template <class T> class Y { void x() { (*this).m; } };",
                 cxxDependentScopeMemberExpr(isArrow())));
}

TEST_P(ASTMatchersTest, IsArrow_MatchesStaticMemberVariablesViaArrow) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `isArrow()` that does not depend on C++.
    return;
  }

  EXPECT_TRUE(matches("class Y { void x() { this->y; } static int y; };",
                      memberExpr(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { y; } static int y; };",
                         memberExpr(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { (*this).y; } static int y; };",
                         memberExpr(isArrow())));
}

TEST_P(ASTMatchersTest, IsArrow_MatchesMemberCallsViaArrow) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `isArrow()` that does not depend on C++.
    return;
  }
  if (GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(
      matches("class Y { void x() { this->x(); } };", memberExpr(isArrow())));
  EXPECT_TRUE(matches("class Y { void x() { x(); } };", memberExpr(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { Y y; y.x(); } };",
                         memberExpr(isArrow())));
  EXPECT_TRUE(
      matches("class Y { template <class T> void x() { this->x<T>(); } };",
              unresolvedMemberExpr(isArrow())));
  EXPECT_TRUE(matches("class Y { template <class T> void x() { x<T>(); } };",
                      unresolvedMemberExpr(isArrow())));
  EXPECT_TRUE(
      notMatches("class Y { template <class T> void x() { (*this).x<T>(); } };",
                 unresolvedMemberExpr(isArrow())));
}

TEST_P(ASTMatchersTest, IsExplicit_CXXConversionDecl) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("struct S { explicit operator int(); };",
                      cxxConversionDecl(isExplicit())));
  EXPECT_TRUE(notMatches("struct S { operator int(); };",
                         cxxConversionDecl(isExplicit())));
}

TEST_P(ASTMatchersTest, IsExplicit_CXXConversionDecl_CXX20) {
  if (!GetParam().isCXX20OrLater()) {
    return;
  }

  EXPECT_TRUE(
      notMatches("template<bool b> struct S { explicit(b) operator int(); };",
                 cxxConversionDecl(isExplicit())));
  EXPECT_TRUE(matches("struct S { explicit(true) operator int(); };",
                      cxxConversionDecl(isExplicit())));
  EXPECT_TRUE(notMatches("struct S { explicit(false) operator int(); };",
                         cxxConversionDecl(isExplicit())));
}

TEST_P(ASTMatchersTest, ArgumentCountIs_CallExpr) {
  StatementMatcher Call1Arg = callExpr(argumentCountIs(1));

  EXPECT_TRUE(matches("void x(int) { x(0); }", Call1Arg));
  EXPECT_TRUE(notMatches("void x(int, int) { x(0, 0); }", Call1Arg));
}

TEST_P(ASTMatchersTest, ArgumentCountIs_CallExpr_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher Call1Arg = callExpr(argumentCountIs(1));
  EXPECT_TRUE(matches("class X { void x(int) { x(0); } };", Call1Arg));
}

TEST_P(ASTMatchersTest, ParameterCountIs) {
  DeclarationMatcher Function1Arg = functionDecl(parameterCountIs(1));
  EXPECT_TRUE(matches("void f(int i) {}", Function1Arg));
  EXPECT_TRUE(notMatches("void f() {}", Function1Arg));
  EXPECT_TRUE(notMatches("void f(int i, int j, int k) {}", Function1Arg));
  EXPECT_TRUE(matches("void f(int i, ...) {};", Function1Arg));
}

TEST_P(ASTMatchersTest, ParameterCountIs_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }

  DeclarationMatcher Function1Arg = functionDecl(parameterCountIs(1));
  EXPECT_TRUE(matches("class X { void f(int i) {} };", Function1Arg));
}

TEST_P(ASTMatchersTest, References) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `references()` that does not depend on C++.
    return;
  }

  DeclarationMatcher ReferenceClassX =
      varDecl(hasType(references(recordDecl(hasName("X")))));
  EXPECT_TRUE(
      matches("class X {}; void y(X y) { X &x = y; }", ReferenceClassX));
  EXPECT_TRUE(
      matches("class X {}; void y(X y) { const X &x = y; }", ReferenceClassX));
  // The match here is on the implicit copy constructor code for
  // class X, not on code 'X x = y'.
  EXPECT_TRUE(matches("class X {}; void y(X y) { X x = y; }", ReferenceClassX));
  EXPECT_TRUE(notMatches("class X {}; extern X x;", ReferenceClassX));
  EXPECT_TRUE(
      notMatches("class X {}; void y(X *y) { X *&x = y; }", ReferenceClassX));
}

TEST_P(ASTMatchersTest, HasLocalQualifiers) {
  if (!GetParam().isCXX11OrLater()) {
    // FIXME: Add a test for `hasLocalQualifiers()` that does not depend on C++.
    return;
  }

  EXPECT_TRUE(notMatches("typedef const int const_int; const_int i = 1;",
                         varDecl(hasType(hasLocalQualifiers()))));
  EXPECT_TRUE(matches("int *const j = nullptr;",
                      varDecl(hasType(hasLocalQualifiers()))));
  EXPECT_TRUE(
      matches("int *volatile k;", varDecl(hasType(hasLocalQualifiers()))));
  EXPECT_TRUE(notMatches("int m;", varDecl(hasType(hasLocalQualifiers()))));
}

TEST_P(ASTMatchersTest, IsExternC_MatchesExternCFunctionDeclarations) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("extern \"C\" void f() {}", functionDecl(isExternC())));
  EXPECT_TRUE(
      matches("extern \"C\" { void f() {} }", functionDecl(isExternC())));
  EXPECT_TRUE(notMatches("void f() {}", functionDecl(isExternC())));
}

TEST_P(ASTMatchersTest, IsExternC_MatchesExternCVariableDeclarations) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("extern \"C\" int i;", varDecl(isExternC())));
  EXPECT_TRUE(matches("extern \"C\" { int i; }", varDecl(isExternC())));
  EXPECT_TRUE(notMatches("int i;", varDecl(isExternC())));
}

TEST_P(ASTMatchersTest, IsStaticStorageClass) {
  EXPECT_TRUE(
      matches("static void f() {}", functionDecl(isStaticStorageClass())));
  EXPECT_TRUE(matches("static int i = 1;", varDecl(isStaticStorageClass())));
  EXPECT_TRUE(notMatches("int i = 1;", varDecl(isStaticStorageClass())));
  EXPECT_TRUE(notMatches("extern int i;", varDecl(isStaticStorageClass())));
  EXPECT_TRUE(notMatches("void f() {}", functionDecl(isStaticStorageClass())));
}

TEST_P(ASTMatchersTest, IsDefaulted) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("class A { ~A(); };",
                         functionDecl(hasName("~A"), isDefaulted())));
  EXPECT_TRUE(matches("class B { ~B() = default; };",
                      functionDecl(hasName("~B"), isDefaulted())));
}

TEST_P(ASTMatchersTest, IsDeleted) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      notMatches("void Func();", functionDecl(hasName("Func"), isDeleted())));
  EXPECT_TRUE(matches("void Func() = delete;",
                      functionDecl(hasName("Func"), isDeleted())));
}

TEST_P(ASTMatchersTest, IsNoThrow_DynamicExceptionSpec) {
  if (!GetParam().supportsCXXDynamicExceptionSpecification()) {
    return;
  }

  EXPECT_TRUE(notMatches("void f();", functionDecl(isNoThrow())));
  EXPECT_TRUE(notMatches("void f() throw(int);", functionDecl(isNoThrow())));
  EXPECT_TRUE(matches("void f() throw();", functionDecl(isNoThrow())));

  EXPECT_TRUE(notMatches("void f();", functionProtoType(isNoThrow())));
  EXPECT_TRUE(
      notMatches("void f() throw(int);", functionProtoType(isNoThrow())));
  EXPECT_TRUE(matches("void f() throw();", functionProtoType(isNoThrow())));
}

TEST_P(ASTMatchersTest, IsNoThrow_CXX11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(
      notMatches("void f() noexcept(false);", functionDecl(isNoThrow())));
  EXPECT_TRUE(matches("void f() noexcept;", functionDecl(isNoThrow())));

  EXPECT_TRUE(
      notMatches("void f() noexcept(false);", functionProtoType(isNoThrow())));
  EXPECT_TRUE(matches("void f() noexcept;", functionProtoType(isNoThrow())));
}

TEST_P(ASTMatchersTest, IsConstexpr) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("constexpr int foo = 42;",
                      varDecl(hasName("foo"), isConstexpr())));
  EXPECT_TRUE(matches("constexpr int bar();",
                      functionDecl(hasName("bar"), isConstexpr())));
}

TEST_P(ASTMatchersTest, IsConstexpr_MatchesIfConstexpr) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }

  EXPECT_TRUE(
      matches("void baz() { if constexpr(1 > 0) {} }", ifStmt(isConstexpr())));
  EXPECT_TRUE(
      notMatches("void baz() { if (1 > 0) {} }", ifStmt(isConstexpr())));
}

TEST_P(ASTMatchersTest, HasInitStatement_MatchesSelectionInitializers) {
  EXPECT_TRUE(notMatches("void baz() { if (1 > 0) {} }",
                         ifStmt(hasInitStatement(anything()))));
  EXPECT_TRUE(notMatches("void baz(int i) { switch (i) { default: break; } }",
                         switchStmt(hasInitStatement(anything()))));
}

TEST_P(ASTMatchersTest, HasInitStatement_MatchesSelectionInitializers_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("void baz() { if (int i = 1) {} }",
                         ifStmt(hasInitStatement(anything()))));
}

TEST_P(ASTMatchersTest, HasInitStatement_MatchesSelectionInitializers_CXX17) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("void baz() { if (int i = 1; i > 0) {} }",
                      ifStmt(hasInitStatement(anything()))));
  EXPECT_TRUE(
      matches("void baz(int i) { switch (int j = i; j) { default: break; } }",
              switchStmt(hasInitStatement(anything()))));
}

TEST_P(ASTMatchersTest, HasInitStatement_MatchesRangeForInitializers) {
  if (!GetParam().isCXX20OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("void baz() {"
                      "int items[] = {};"
                      "for (auto &arr = items; auto &item : arr) {}"
                      "}",
                      cxxForRangeStmt(hasInitStatement(anything()))));
  EXPECT_TRUE(notMatches("void baz() {"
                         "int items[] = {};"
                         "for (auto &item : items) {}"
                         "}",
                         cxxForRangeStmt(hasInitStatement(anything()))));
}

TEST_P(ASTMatchersTest, TemplateArgumentCountIs) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("template<typename T> struct C {}; C<int> c;",
              classTemplateSpecializationDecl(templateArgumentCountIs(1))));
  EXPECT_TRUE(
      notMatches("template<typename T> struct C {}; C<int> c;",
                 classTemplateSpecializationDecl(templateArgumentCountIs(2))));

  EXPECT_TRUE(matches("template<typename T> struct C {}; C<int> c;",
                      templateSpecializationType(templateArgumentCountIs(1))));
  EXPECT_TRUE(
      notMatches("template<typename T> struct C {}; C<int> c;",
                 templateSpecializationType(templateArgumentCountIs(2))));
}

TEST_P(ASTMatchersTest, IsIntegral) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches(
      "template<int T> struct C {}; C<42> c;",
      classTemplateSpecializationDecl(hasAnyTemplateArgument(isIntegral()))));
  EXPECT_TRUE(notMatches("template<typename T> struct C {}; C<int> c;",
                         classTemplateSpecializationDecl(hasAnyTemplateArgument(
                             templateArgument(isIntegral())))));
}

TEST_P(ASTMatchersTest, EqualsIntegralValue) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("template<int T> struct C {}; C<42> c;",
                      classTemplateSpecializationDecl(
                          hasAnyTemplateArgument(equalsIntegralValue("42")))));
  EXPECT_TRUE(matches("template<int T> struct C {}; C<-42> c;",
                      classTemplateSpecializationDecl(
                          hasAnyTemplateArgument(equalsIntegralValue("-42")))));
  EXPECT_TRUE(matches("template<int T> struct C {}; C<-0042> c;",
                      classTemplateSpecializationDecl(
                          hasAnyTemplateArgument(equalsIntegralValue("-34")))));
  EXPECT_TRUE(notMatches("template<int T> struct C {}; C<42> c;",
                         classTemplateSpecializationDecl(hasAnyTemplateArgument(
                             equalsIntegralValue("0042")))));
}

TEST_P(ASTMatchersTest, AccessSpecDecl) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class C { public: int i; };", accessSpecDecl()));
  EXPECT_TRUE(
      matches("class C { public: int i; };", accessSpecDecl(isPublic())));
  EXPECT_TRUE(
      notMatches("class C { public: int i; };", accessSpecDecl(isProtected())));
  EXPECT_TRUE(
      notMatches("class C { public: int i; };", accessSpecDecl(isPrivate())));

  EXPECT_TRUE(notMatches("class C { int i; };", accessSpecDecl()));
}

TEST_P(ASTMatchersTest, IsFinal) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("class X final {};", cxxRecordDecl(isFinal())));
  EXPECT_TRUE(matches("class X { virtual void f() final; };",
                      cxxMethodDecl(isFinal())));
  EXPECT_TRUE(notMatches("class X {};", cxxRecordDecl(isFinal())));
  EXPECT_TRUE(
      notMatches("class X { virtual void f(); };", cxxMethodDecl(isFinal())));
}

TEST_P(ASTMatchersTest, IsVirtual) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class X { virtual int f(); };",
                      cxxMethodDecl(isVirtual(), hasName("::X::f"))));
  EXPECT_TRUE(notMatches("class X { int f(); };", cxxMethodDecl(isVirtual())));
}

TEST_P(ASTMatchersTest, IsVirtualAsWritten) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class A { virtual int f(); };"
                      "class B : public A { int f(); };",
                      cxxMethodDecl(isVirtualAsWritten(), hasName("::A::f"))));
  EXPECT_TRUE(
      notMatches("class A { virtual int f(); };"
                 "class B : public A { int f(); };",
                 cxxMethodDecl(isVirtualAsWritten(), hasName("::B::f"))));
}

TEST_P(ASTMatchersTest, IsPure) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class X { virtual int f() = 0; };",
                      cxxMethodDecl(isPure(), hasName("::X::f"))));
  EXPECT_TRUE(notMatches("class X { int f(); };", cxxMethodDecl(isPure())));
}

TEST_P(ASTMatchersTest, IsCopyAssignmentOperator) {
  if (!GetParam().isCXX()) {
    return;
  }

  auto CopyAssignment =
      cxxMethodDecl(isCopyAssignmentOperator(), unless(isImplicit()));
  EXPECT_TRUE(matches("class X { X &operator=(X); };", CopyAssignment));
  EXPECT_TRUE(matches("class X { X &operator=(X &); };", CopyAssignment));
  EXPECT_TRUE(matches("class X { X &operator=(const X &); };", CopyAssignment));
  EXPECT_TRUE(matches("class X { X &operator=(volatile X &); };", //
                      CopyAssignment));
  EXPECT_TRUE(matches("class X { X &operator=(const volatile X &); };",
                      CopyAssignment));
  EXPECT_TRUE(notMatches("class X { X &operator=(X &&); };", CopyAssignment));
}

TEST_P(ASTMatchersTest, IsMoveAssignmentOperator) {
  if (!GetParam().isCXX()) {
    return;
  }

  auto MoveAssignment =
      cxxMethodDecl(isMoveAssignmentOperator(), unless(isImplicit()));
  EXPECT_TRUE(notMatches("class X { X &operator=(X); };", MoveAssignment));
  EXPECT_TRUE(matches("class X { X &operator=(X &&); };", MoveAssignment));
  EXPECT_TRUE(matches("class X { X &operator=(const X &&); };", //
                      MoveAssignment));
  EXPECT_TRUE(matches("class X { X &operator=(volatile X &&); };", //
                      MoveAssignment));
  EXPECT_TRUE(matches("class X { X &operator=(const volatile X &&); };",
                      MoveAssignment));
  EXPECT_TRUE(notMatches("class X { X &operator=(X &); };", MoveAssignment));
}

TEST_P(ASTMatchersTest, IsConst) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("struct A { void foo() const; };", cxxMethodDecl(isConst())));
  EXPECT_TRUE(
      notMatches("struct A { void foo(); };", cxxMethodDecl(isConst())));
}

TEST_P(ASTMatchersTest, IsOverride) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class X { virtual int f(); }; "
                      "class Y : public X { int f(); };",
                      cxxMethodDecl(isOverride(), hasName("::Y::f"))));
  EXPECT_TRUE(notMatches("class X { virtual int f(); }; "
                         "class Y : public X { int f(); };",
                         cxxMethodDecl(isOverride(), hasName("::X::f"))));
  EXPECT_TRUE(notMatches("class X { int f(); }; "
                         "class Y : public X { int f(); };",
                         cxxMethodDecl(isOverride())));
  EXPECT_TRUE(notMatches("class X { int f(); int f(int); }; ",
                         cxxMethodDecl(isOverride())));
  EXPECT_TRUE(
      matches("template <typename Base> struct Y : Base { void f() override;};",
              cxxMethodDecl(isOverride(), hasName("::Y::f"))));
}

TEST_P(ASTMatchersTest, HasArgument_CXXConstructorDecl) {
  if (!GetParam().isCXX()) {
    return;
  }

  auto Constructor = traverse(
      TK_AsIs,
      cxxConstructExpr(hasArgument(0, declRefExpr(to(varDecl(hasName("y")))))));

  EXPECT_TRUE(matches(
      "class X { public: X(int); }; void x() { int y; X x(y); }", Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; X x = X(y); }",
              Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; X x = y; }",
              Constructor));
  EXPECT_TRUE(notMatches(
      "class X { public: X(int); }; void x() { int z; X x(z); }", Constructor));

  StatementMatcher WrongIndex =
      traverse(TK_AsIs, cxxConstructExpr(hasArgument(
                            42, declRefExpr(to(varDecl(hasName("y")))))));
  EXPECT_TRUE(notMatches(
      "class X { public: X(int); }; void x() { int y; X x(y); }", WrongIndex));
}

TEST_P(ASTMatchersTest, ArgumentCountIs_CXXConstructExpr) {
  if (!GetParam().isCXX()) {
    return;
  }

  auto Constructor1Arg =
      traverse(TK_AsIs, cxxConstructExpr(argumentCountIs(1)));

  EXPECT_TRUE(matches("class X { public: X(int); }; void x() { X x(0); }",
                      Constructor1Arg));
  EXPECT_TRUE(matches("class X { public: X(int); }; void x() { X x = X(0); }",
                      Constructor1Arg));
  EXPECT_TRUE(matches("class X { public: X(int); }; void x() { X x = 0; }",
                      Constructor1Arg));
  EXPECT_TRUE(
      notMatches("class X { public: X(int, int); }; void x() { X x(0, 0); }",
                 Constructor1Arg));
}

TEST(ASTMatchersTest, NamesMember_CXXDependentScopeMemberExpr) {

  // Member functions:
  {
    auto Code = "template <typename T> struct S{ void mem(); }; template "
                "<typename T> void x() { S<T> s; s.mem(); }";

    EXPECT_TRUE(matches(
        Code,
        cxxDependentScopeMemberExpr(
            hasObjectExpression(declRefExpr(hasType(templateSpecializationType(
                hasDeclaration(classTemplateDecl(has(cxxRecordDecl(
                    has(cxxMethodDecl(hasName("mem")).bind("templMem")))))))))),
            memberHasSameNameAsBoundNode("templMem"))));

    EXPECT_TRUE(
        matches(Code, cxxDependentScopeMemberExpr(hasMemberName("mem"))));
  }

  // Member variables:
  {
    auto Code = "template <typename T> struct S{ int mem; }; template "
                "<typename T> void x() { S<T> s; s.mem; }";

    EXPECT_TRUE(
        matches(Code, cxxDependentScopeMemberExpr(hasMemberName("mem"))));

    EXPECT_TRUE(matches(
        Code,
        cxxDependentScopeMemberExpr(
            hasObjectExpression(declRefExpr(hasType(templateSpecializationType(
                hasDeclaration(classTemplateDecl(has(cxxRecordDecl(
                    has(fieldDecl(hasName("mem")).bind("templMem")))))))))),
            memberHasSameNameAsBoundNode("templMem"))));
  }

  // static member variables:
  {
    auto Code = "template <typename T> struct S{ static int mem; }; template "
                "<typename T> void x() { S<T> s; s.mem; }";

    EXPECT_TRUE(
        matches(Code, cxxDependentScopeMemberExpr(hasMemberName("mem"))));

    EXPECT_TRUE(matches(
        Code,
        cxxDependentScopeMemberExpr(
            hasObjectExpression(declRefExpr(hasType(templateSpecializationType(
                hasDeclaration(classTemplateDecl(has(cxxRecordDecl(
                    has(varDecl(hasName("mem")).bind("templMem")))))))))),
            memberHasSameNameAsBoundNode("templMem"))));
  }
  {
    auto Code = R"cpp(
template <typename T>
struct S {
  bool operator==(int) const { return true; }
};

template <typename T>
void func(T t) {
  S<T> s;
  s.operator==(1);
}
)cpp";

    EXPECT_TRUE(matches(
        Code, cxxDependentScopeMemberExpr(hasMemberName("operator=="))));
  }

  // other named decl:
  {
    auto Code = "template <typename T> struct S{ static int mem; }; struct "
                "mem{}; template "
                "<typename T> void x() { S<T> s; s.mem; }";

    EXPECT_TRUE(matches(
        Code,
        translationUnitDecl(has(cxxRecordDecl(hasName("mem"))),
                            hasDescendant(cxxDependentScopeMemberExpr()))));

    EXPECT_FALSE(matches(
        Code,
        translationUnitDecl(has(cxxRecordDecl(hasName("mem")).bind("templMem")),
                            hasDescendant(cxxDependentScopeMemberExpr(
                                memberHasSameNameAsBoundNode("templMem"))))));
  }
}

TEST(ASTMatchersTest, ArgumentCountIs_CXXUnresolvedConstructExpr) {
  const auto *Code =
      "template <typename T> struct S{}; template <typename T> void "
      "x() { auto s = S<T>(); }";

  EXPECT_TRUE(matches(Code, cxxUnresolvedConstructExpr(argumentCountIs(0))));
  EXPECT_TRUE(notMatches(Code, cxxUnresolvedConstructExpr(argumentCountIs(1))));
}

TEST(ASTMatchersTest, HasArgument_CXXUnresolvedConstructExpr) {
  const auto *Code =
      "template <typename T> struct S{ S(int){} }; template <typename "
      "T> void x() { int y; auto s = S<T>(y); }";
  EXPECT_TRUE(matches(Code, cxxUnresolvedConstructExpr(hasArgument(
                                0, declRefExpr(to(varDecl(hasName("y"))))))));
  EXPECT_TRUE(
      notMatches(Code, cxxUnresolvedConstructExpr(hasArgument(
                           0, declRefExpr(to(varDecl(hasName("x"))))))));
}

TEST_P(ASTMatchersTest, IsListInitialization) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  auto ConstructorListInit =
      traverse(TK_AsIs, varDecl(has(cxxConstructExpr(isListInitialization()))));

  EXPECT_TRUE(matches("class X { public: X(int); }; void x() { X x{0}; }",
                      ConstructorListInit));
  EXPECT_FALSE(matches("class X { public: X(int); }; void x() { X x(0); }",
                       ConstructorListInit));
}

TEST_P(ASTMatchersTest, IsImplicit_CXXConstructorDecl) {
  if (!GetParam().isCXX()) {
    return;
  }

  // This one doesn't match because the constructor is not added by the
  // compiler (it is not needed).
  EXPECT_TRUE(notMatches("class Foo { };", cxxConstructorDecl(isImplicit())));
  // The compiler added the implicit default constructor.
  EXPECT_TRUE(matches("class Foo { }; Foo* f = new Foo();",
                      cxxConstructorDecl(isImplicit())));
  EXPECT_TRUE(matches("class Foo { Foo(){} };",
                      cxxConstructorDecl(unless(isImplicit()))));
  // The compiler added an implicit assignment operator.
  EXPECT_TRUE(matches("struct A { int x; } a = {0}, b = a; void f() { a = b; }",
                      cxxMethodDecl(isImplicit(), hasName("operator="))));
}

TEST_P(ASTMatchersTest, IsExplicit_CXXConstructorDecl) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("struct S { explicit S(int); };",
                      cxxConstructorDecl(isExplicit())));
  EXPECT_TRUE(
      notMatches("struct S { S(int); };", cxxConstructorDecl(isExplicit())));
}

TEST_P(ASTMatchersTest, IsExplicit_CXXConstructorDecl_CXX20) {
  if (!GetParam().isCXX20OrLater()) {
    return;
  }

  EXPECT_TRUE(notMatches("template<bool b> struct S { explicit(b) S(int);};",
                         cxxConstructorDecl(isExplicit())));
  EXPECT_TRUE(matches("struct S { explicit(true) S(int);};",
                      cxxConstructorDecl(isExplicit())));
  EXPECT_TRUE(notMatches("struct S { explicit(false) S(int);};",
                         cxxConstructorDecl(isExplicit())));
}

TEST_P(ASTMatchersTest, IsExplicit_CXXDeductionGuideDecl) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }

  EXPECT_TRUE(notMatches("template<typename T> struct S { S(int);};"
                         "S(int) -> S<int>;",
                         cxxDeductionGuideDecl(isExplicit())));
  EXPECT_TRUE(matches("template<typename T> struct S { S(int);};"
                      "explicit S(int) -> S<int>;",
                      cxxDeductionGuideDecl(isExplicit())));
}

TEST_P(ASTMatchersTest, IsExplicit_CXXDeductionGuideDecl_CXX20) {
  if (!GetParam().isCXX20OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("template<typename T> struct S { S(int);};"
                      "explicit(true) S(int) -> S<int>;",
                      cxxDeductionGuideDecl(isExplicit())));
  EXPECT_TRUE(notMatches("template<typename T> struct S { S(int);};"
                         "explicit(false) S(int) -> S<int>;",
                         cxxDeductionGuideDecl(isExplicit())));
  EXPECT_TRUE(
      notMatches("template<typename T> struct S { S(int);};"
                 "template<bool b = true> explicit(b) S(int) -> S<int>;",
                 cxxDeductionGuideDecl(isExplicit())));
}

TEST_P(ASTMatchersTest, CXXConstructorDecl_Kinds) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("struct S { S(); };", cxxConstructorDecl(isDefaultConstructor(),
                                                       unless(isImplicit()))));
  EXPECT_TRUE(notMatches(
      "struct S { S(); };",
      cxxConstructorDecl(isCopyConstructor(), unless(isImplicit()))));
  EXPECT_TRUE(notMatches(
      "struct S { S(); };",
      cxxConstructorDecl(isMoveConstructor(), unless(isImplicit()))));

  EXPECT_TRUE(notMatches(
      "struct S { S(const S&); };",
      cxxConstructorDecl(isDefaultConstructor(), unless(isImplicit()))));
  EXPECT_TRUE(
      matches("struct S { S(const S&); };",
              cxxConstructorDecl(isCopyConstructor(), unless(isImplicit()))));
  EXPECT_TRUE(notMatches(
      "struct S { S(const S&); };",
      cxxConstructorDecl(isMoveConstructor(), unless(isImplicit()))));

  EXPECT_TRUE(notMatches(
      "struct S { S(S&&); };",
      cxxConstructorDecl(isDefaultConstructor(), unless(isImplicit()))));
  EXPECT_TRUE(notMatches(
      "struct S { S(S&&); };",
      cxxConstructorDecl(isCopyConstructor(), unless(isImplicit()))));
  EXPECT_TRUE(
      matches("struct S { S(S&&); };",
              cxxConstructorDecl(isMoveConstructor(), unless(isImplicit()))));
}

TEST_P(ASTMatchersTest, IsUserProvided) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(notMatches("struct S { int X = 0; };",
                         cxxConstructorDecl(isUserProvided())));
  EXPECT_TRUE(notMatches("struct S { S() = default; };",
                         cxxConstructorDecl(isUserProvided())));
  EXPECT_TRUE(notMatches("struct S { S() = delete; };",
                         cxxConstructorDecl(isUserProvided())));
  EXPECT_TRUE(
      matches("struct S { S(); };", cxxConstructorDecl(isUserProvided())));
  EXPECT_TRUE(matches("struct S { S(); }; S::S(){}",
                      cxxConstructorDecl(isUserProvided())));
}

TEST_P(ASTMatchersTest, IsDelegatingConstructor) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(notMatches("struct S { S(); S(int); int X; };",
                         cxxConstructorDecl(isDelegatingConstructor())));
  EXPECT_TRUE(notMatches("struct S { S(){} S(int X) : X(X) {} int X; };",
                         cxxConstructorDecl(isDelegatingConstructor())));
  EXPECT_TRUE(matches(
      "struct S { S() : S(0) {} S(int X) : X(X) {} int X; };",
      cxxConstructorDecl(isDelegatingConstructor(), parameterCountIs(0))));
  EXPECT_TRUE(matches(
      "struct S { S(); S(int X); int X; }; S::S(int X) : S() {}",
      cxxConstructorDecl(isDelegatingConstructor(), parameterCountIs(1))));
}

TEST_P(ASTMatchersTest, HasSize) {
  StatementMatcher Literal = stringLiteral(hasSize(4));
  EXPECT_TRUE(matches("const char *s = \"abcd\";", Literal));
  // with escaped characters
  EXPECT_TRUE(matches("const char *s = \"\x05\x06\x07\x08\";", Literal));
  // no matching, too small
  EXPECT_TRUE(notMatches("const char *s = \"ab\";", Literal));
}

TEST_P(ASTMatchersTest, HasSize_CXX) {
  if (!GetParam().isCXX()) {
    // FIXME: Fix this test to also work in non-C++ language modes.
    return;
  }

  StatementMatcher Literal = stringLiteral(hasSize(4));
  // wide string
  EXPECT_TRUE(matches("const wchar_t *s = L\"abcd\";", Literal));
}

TEST_P(ASTMatchersTest, HasName_MatchesNamespaces) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("namespace a { namespace b { class C; } }",
                      recordDecl(hasName("a::b::C"))));
  EXPECT_TRUE(matches("namespace a { namespace b { class C; } }",
                      recordDecl(hasName("::a::b::C"))));
  EXPECT_TRUE(matches("namespace a { namespace b { class C; } }",
                      recordDecl(hasName("b::C"))));
  EXPECT_TRUE(matches("namespace a { namespace b { class C; } }",
                      recordDecl(hasName("C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
                         recordDecl(hasName("c::b::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
                         recordDecl(hasName("a::c::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
                         recordDecl(hasName("a::b::A"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
                         recordDecl(hasName("::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
                         recordDecl(hasName("::b::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
                         recordDecl(hasName("z::a::b::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class C; } }",
                         recordDecl(hasName("a+b::C"))));
  EXPECT_TRUE(notMatches("namespace a { namespace b { class AC; } }",
                         recordDecl(hasName("C"))));
}

TEST_P(ASTMatchersTest, HasName_MatchesOuterClasses) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class A { class B { class C; }; };",
                      recordDecl(hasName("A::B::C"))));
  EXPECT_TRUE(matches("class A { class B { class C; }; };",
                      recordDecl(hasName("::A::B::C"))));
  EXPECT_TRUE(matches("class A { class B { class C; }; };",
                      recordDecl(hasName("B::C"))));
  EXPECT_TRUE(
      matches("class A { class B { class C; }; };", recordDecl(hasName("C"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
                         recordDecl(hasName("c::B::C"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
                         recordDecl(hasName("A::c::C"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
                         recordDecl(hasName("A::B::A"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
                         recordDecl(hasName("::C"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
                         recordDecl(hasName("::B::C"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
                         recordDecl(hasName("z::A::B::C"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
                         recordDecl(hasName("A+B::C"))));
}

TEST_P(ASTMatchersTest, HasName_MatchesInlinedNamespaces) {
  if (!GetParam().isCXX()) {
    return;
  }

  StringRef code = "namespace a { inline namespace b { class C; } }";
  EXPECT_TRUE(matches(code, recordDecl(hasName("a::b::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("a::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("::a::b::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("::a::C"))));
}

TEST_P(ASTMatchersTest, HasName_MatchesAnonymousNamespaces) {
  if (!GetParam().isCXX()) {
    return;
  }

  StringRef code = "namespace a { namespace { class C; } }";
  EXPECT_TRUE(
      matches(code, recordDecl(hasName("a::(anonymous namespace)::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("a::C"))));
  EXPECT_TRUE(
      matches(code, recordDecl(hasName("::a::(anonymous namespace)::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("::a::C"))));
}

TEST_P(ASTMatchersTest, HasName_MatchesAnonymousOuterClasses) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class A { class { class C; } x; };",
                      recordDecl(hasName("A::(anonymous class)::C"))));
  EXPECT_TRUE(matches("class A { class { class C; } x; };",
                      recordDecl(hasName("::A::(anonymous class)::C"))));
  EXPECT_FALSE(matches("class A { class { class C; } x; };",
                       recordDecl(hasName("::A::C"))));
  EXPECT_TRUE(matches("class A { struct { class C; } x; };",
                      recordDecl(hasName("A::(anonymous struct)::C"))));
  EXPECT_TRUE(matches("class A { struct { class C; } x; };",
                      recordDecl(hasName("::A::(anonymous struct)::C"))));
  EXPECT_FALSE(matches("class A { struct { class C; } x; };",
                       recordDecl(hasName("::A::C"))));
}

TEST_P(ASTMatchersTest, HasName_MatchesFunctionScope) {
  if (!GetParam().isCXX()) {
    return;
  }

  StringRef code =
      "namespace a { void F(int a) { struct S { int m; }; int i; } }";
  EXPECT_TRUE(matches(code, varDecl(hasName("i"))));
  EXPECT_FALSE(matches(code, varDecl(hasName("F()::i"))));

  EXPECT_TRUE(matches(code, fieldDecl(hasName("m"))));
  EXPECT_TRUE(matches(code, fieldDecl(hasName("S::m"))));
  EXPECT_TRUE(matches(code, fieldDecl(hasName("F(int)::S::m"))));
  EXPECT_TRUE(matches(code, fieldDecl(hasName("a::F(int)::S::m"))));
  EXPECT_TRUE(matches(code, fieldDecl(hasName("::a::F(int)::S::m"))));
}

TEST_P(ASTMatchersTest, HasName_QualifiedStringMatchesThroughLinkage) {
  if (!GetParam().isCXX()) {
    return;
  }

  // https://bugs.llvm.org/show_bug.cgi?id=42193
  StringRef code = R"cpp(namespace foo { extern "C" void test(); })cpp";
  EXPECT_TRUE(matches(code, functionDecl(hasName("test"))));
  EXPECT_TRUE(matches(code, functionDecl(hasName("foo::test"))));
  EXPECT_TRUE(matches(code, functionDecl(hasName("::foo::test"))));
  EXPECT_TRUE(notMatches(code, functionDecl(hasName("::test"))));

  code = R"cpp(namespace foo { extern "C" { void test(); } })cpp";
  EXPECT_TRUE(matches(code, functionDecl(hasName("test"))));
  EXPECT_TRUE(matches(code, functionDecl(hasName("foo::test"))));
  EXPECT_TRUE(matches(code, functionDecl(hasName("::foo::test"))));
  EXPECT_TRUE(notMatches(code, functionDecl(hasName("::test"))));
}

TEST_P(ASTMatchersTest, HasAnyName) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `hasAnyName()` that does not depend on C++.
    return;
  }

  StringRef Code = "namespace a { namespace b { class C; } }";

  EXPECT_TRUE(matches(Code, recordDecl(hasAnyName("XX", "a::b::C"))));
  EXPECT_TRUE(matches(Code, recordDecl(hasAnyName("a::b::C", "XX"))));
  EXPECT_TRUE(matches(Code, recordDecl(hasAnyName("XX::C", "a::b::C"))));
  EXPECT_TRUE(matches(Code, recordDecl(hasAnyName("XX", "C"))));

  EXPECT_TRUE(notMatches(Code, recordDecl(hasAnyName("::C", "::b::C"))));
  EXPECT_TRUE(
      matches(Code, recordDecl(hasAnyName("::C", "::b::C", "::a::b::C"))));

  std::vector<StringRef> Names = {"::C", "::b::C", "::a::b::C"};
  EXPECT_TRUE(matches(Code, recordDecl(hasAnyName(Names))));
}

TEST_P(ASTMatchersTest, IsDefinition) {
  DeclarationMatcher DefinitionOfClassA =
      recordDecl(hasName("A"), isDefinition());
  EXPECT_TRUE(matches("struct A {};", DefinitionOfClassA));
  EXPECT_TRUE(notMatches("struct A;", DefinitionOfClassA));

  DeclarationMatcher DefinitionOfVariableA =
      varDecl(hasName("a"), isDefinition());
  EXPECT_TRUE(matches("int a;", DefinitionOfVariableA));
  EXPECT_TRUE(notMatches("extern int a;", DefinitionOfVariableA));
}

TEST_P(ASTMatchersTest, IsDefinition_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }

  DeclarationMatcher DefinitionOfMethodA =
      cxxMethodDecl(hasName("a"), isDefinition());
  EXPECT_TRUE(matches("class A { void a() {} };", DefinitionOfMethodA));
  EXPECT_TRUE(notMatches("class A { void a(); };", DefinitionOfMethodA));

  DeclarationMatcher DefinitionOfObjCMethodA =
      objcMethodDecl(hasName("a"), isDefinition());
  EXPECT_TRUE(matchesObjC("@interface A @end "
                          "@implementation A; -(void)a {} @end",
                          DefinitionOfObjCMethodA));
  EXPECT_TRUE(
      notMatchesObjC("@interface A; - (void)a; @end", DefinitionOfObjCMethodA));
}

TEST_P(ASTMatchersTest, HandlesNullQualTypes) {
  if (!GetParam().isCXX()) {
    // FIXME: Add an equivalent test that does not depend on C++.
    return;
  }

  // FIXME: Add a Type matcher so we can replace uses of this
  // variable with Type(True())
  const TypeMatcher AnyType = anything();

  // We don't really care whether this matcher succeeds; we're testing that
  // it completes without crashing.
  EXPECT_TRUE(matches(
      "struct A { };"
      "template <typename T>"
      "void f(T t) {"
      "  T local_t(t /* this becomes a null QualType in the AST */);"
      "}"
      "void g() {"
      "  f(0);"
      "}",
      expr(hasType(TypeMatcher(anyOf(TypeMatcher(hasDeclaration(anything())),
                                     pointsTo(AnyType), references(AnyType)
                                     // Other QualType matchers should go here.
                                     ))))));
}

TEST_P(ASTMatchersTest, ObjCIvarRefExpr) {
  StringRef ObjCString =
      "@interface A @end "
      "@implementation A { A *x; } - (void) func { x = 0; } @end";
  EXPECT_TRUE(matchesObjC(ObjCString, objcIvarRefExpr()));
  EXPECT_TRUE(matchesObjC(
      ObjCString, objcIvarRefExpr(hasDeclaration(namedDecl(hasName("x"))))));
  EXPECT_FALSE(matchesObjC(
      ObjCString, objcIvarRefExpr(hasDeclaration(namedDecl(hasName("y"))))));
}

TEST_P(ASTMatchersTest, BlockExpr) {
  EXPECT_TRUE(matchesObjC("void f() { ^{}(); }", blockExpr()));
}

TEST_P(ASTMatchersTest,
       StatementCountIs_FindsNoStatementsInAnEmptyCompoundStatement) {
  EXPECT_TRUE(matches("void f() { }", compoundStmt(statementCountIs(0))));
  EXPECT_TRUE(notMatches("void f() {}", compoundStmt(statementCountIs(1))));
}

TEST_P(ASTMatchersTest, StatementCountIs_AppearsToMatchOnlyOneCount) {
  EXPECT_TRUE(matches("void f() { 1; }", compoundStmt(statementCountIs(1))));
  EXPECT_TRUE(notMatches("void f() { 1; }", compoundStmt(statementCountIs(0))));
  EXPECT_TRUE(notMatches("void f() { 1; }", compoundStmt(statementCountIs(2))));
}

TEST_P(ASTMatchersTest, StatementCountIs_WorksWithMultipleStatements) {
  EXPECT_TRUE(
      matches("void f() { 1; 2; 3; }", compoundStmt(statementCountIs(3))));
}

TEST_P(ASTMatchersTest, StatementCountIs_WorksWithNestedCompoundStatements) {
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
                      compoundStmt(statementCountIs(1))));
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
                      compoundStmt(statementCountIs(2))));
  EXPECT_TRUE(notMatches("void f() { { 1; } { 1; 2; 3; 4; } }",
                         compoundStmt(statementCountIs(3))));
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
                      compoundStmt(statementCountIs(4))));
}

TEST_P(ASTMatchersTest, Member_WorksInSimplestCase) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `member()` that does not depend on C++.
    return;
  }
  EXPECT_TRUE(matches("struct { int first; } s; int i(s.first);",
                      memberExpr(member(hasName("first")))));
}

TEST_P(ASTMatchersTest, Member_DoesNotMatchTheBaseExpression) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `member()` that does not depend on C++.
    return;
  }

  // Don't pick out the wrong part of the member expression, this should
  // be checking the member (name) only.
  EXPECT_TRUE(notMatches("struct { int i; } first; int i(first.i);",
                         memberExpr(member(hasName("first")))));
}

TEST_P(ASTMatchersTest, Member_MatchesInMemberFunctionCall) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("void f() {"
                      "  struct { void first() {}; } s;"
                      "  s.first();"
                      "};",
                      memberExpr(member(hasName("first")))));
}

TEST_P(ASTMatchersTest, FieldDecl) {
  EXPECT_TRUE(
      matches("struct A { int i; }; void f() { struct A a; a.i = 2; }",
              memberExpr(hasDeclaration(fieldDecl(hasType(isInteger()))))));
  EXPECT_TRUE(
      notMatches("struct A { float f; }; void f() { struct A a; a.f = 2.0f; }",
                 memberExpr(hasDeclaration(fieldDecl(hasType(isInteger()))))));
}

TEST_P(ASTMatchersTest, IsBitField) {
  EXPECT_TRUE(matches("struct C { int a : 2; int b; };",
                      fieldDecl(isBitField(), hasName("a"))));
  EXPECT_TRUE(notMatches("struct C { int a : 2; int b; };",
                         fieldDecl(isBitField(), hasName("b"))));
  EXPECT_TRUE(matches("struct C { int a : 2; int b : 4; };",
                      fieldDecl(isBitField(), hasBitWidth(2), hasName("a"))));
}

TEST_P(ASTMatchersTest, HasInClassInitializer) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("class C { int a = 2; int b; };",
              fieldDecl(hasInClassInitializer(integerLiteral(equals(2))),
                        hasName("a"))));
  EXPECT_TRUE(
      notMatches("class C { int a = 2; int b; };",
                 fieldDecl(hasInClassInitializer(anything()), hasName("b"))));
}

TEST_P(ASTMatchersTest, IsPublic_IsProtected_IsPrivate) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("struct A { int i; };", fieldDecl(isPublic(), hasName("i"))));
  EXPECT_TRUE(notMatches("struct A { int i; };",
                         fieldDecl(isProtected(), hasName("i"))));
  EXPECT_TRUE(
      notMatches("struct A { int i; };", fieldDecl(isPrivate(), hasName("i"))));

  EXPECT_TRUE(
      notMatches("class A { int i; };", fieldDecl(isPublic(), hasName("i"))));
  EXPECT_TRUE(notMatches("class A { int i; };",
                         fieldDecl(isProtected(), hasName("i"))));
  EXPECT_TRUE(
      matches("class A { int i; };", fieldDecl(isPrivate(), hasName("i"))));

  EXPECT_TRUE(notMatches("class A { protected: int i; };",
                         fieldDecl(isPublic(), hasName("i"))));
  EXPECT_TRUE(matches("class A { protected: int i; };",
                      fieldDecl(isProtected(), hasName("i"))));
  EXPECT_TRUE(notMatches("class A { protected: int i; };",
                         fieldDecl(isPrivate(), hasName("i"))));

  // Non-member decls have the AccessSpecifier AS_none and thus aren't matched.
  EXPECT_TRUE(notMatches("int i;", varDecl(isPublic(), hasName("i"))));
  EXPECT_TRUE(notMatches("int i;", varDecl(isProtected(), hasName("i"))));
  EXPECT_TRUE(notMatches("int i;", varDecl(isPrivate(), hasName("i"))));
}

TEST_P(ASTMatchersTest,
       HasDynamicExceptionSpec_MatchesDynamicExceptionSpecifications) {
  if (!GetParam().supportsCXXDynamicExceptionSpecification()) {
    return;
  }

  EXPECT_TRUE(notMatches("void f();", functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void j() throw();", functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void k() throw(int);", functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void l() throw(...);", functionDecl(hasDynamicExceptionSpec())));

  EXPECT_TRUE(
      notMatches("void f();", functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(matches("void j() throw();",
                      functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(matches("void k() throw(int);",
                      functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(matches("void l() throw(...);",
                      functionProtoType(hasDynamicExceptionSpec())));
}

TEST_P(ASTMatchersTest,
       HasDynamicExceptionSpec_MatchesDynamicExceptionSpecifications_CXX11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(notMatches("void g() noexcept;",
                         functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void h() noexcept(true);",
                         functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void i() noexcept(false);",
                         functionDecl(hasDynamicExceptionSpec())));

  EXPECT_TRUE(notMatches("void g() noexcept;",
                         functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void h() noexcept(true);",
                         functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void i() noexcept(false);",
                         functionProtoType(hasDynamicExceptionSpec())));
}

TEST_P(ASTMatchersTest, HasObjectExpression_DoesNotMatchMember) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches(
      "class X {}; struct Z { X m; }; void f(Z z) { z.m; }",
      memberExpr(hasObjectExpression(hasType(recordDecl(hasName("X")))))));
}

TEST_P(ASTMatchersTest, HasObjectExpression_MatchesBaseOfVariable) {
  EXPECT_TRUE(matches(
      "struct X { int m; }; void f(struct X x) { x.m; }",
      memberExpr(hasObjectExpression(hasType(recordDecl(hasName("X")))))));
  EXPECT_TRUE(matches("struct X { int m; }; void f(struct X* x) { x->m; }",
                      memberExpr(hasObjectExpression(
                          hasType(pointsTo(recordDecl(hasName("X"))))))));
}

TEST_P(ASTMatchersTest, HasObjectExpression_MatchesBaseOfVariable_CXX) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("template <class T> struct X { void f() { T t; t.m; } };",
                      cxxDependentScopeMemberExpr(hasObjectExpression(
                          declRefExpr(to(namedDecl(hasName("t"))))))));
  EXPECT_TRUE(
      matches("template <class T> struct X { void f() { T t; t->m; } };",
              cxxDependentScopeMemberExpr(hasObjectExpression(
                  declRefExpr(to(namedDecl(hasName("t"))))))));
}

TEST_P(ASTMatchersTest, HasObjectExpression_MatchesBaseOfMemberFunc) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches(
      "struct X { void f(); }; void g(X x) { x.f(); }",
      memberExpr(hasObjectExpression(hasType(recordDecl(hasName("X")))))));
}

TEST_P(ASTMatchersTest, HasObjectExpression_MatchesBaseOfMemberFunc_Template) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("struct X { template <class T> void f(); };"
                      "template <class T> void g(X x) { x.f<T>(); }",
                      unresolvedMemberExpr(hasObjectExpression(
                          hasType(recordDecl(hasName("X")))))));
  EXPECT_TRUE(matches("template <class T> void f(T t) { t.g(); }",
                      cxxDependentScopeMemberExpr(hasObjectExpression(
                          declRefExpr(to(namedDecl(hasName("t"))))))));
}

TEST_P(ASTMatchersTest, HasObjectExpression_ImplicitlyFormedMemberExpression) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class X {}; struct S { X m; void f() { this->m; } };",
                      memberExpr(hasObjectExpression(
                          hasType(pointsTo(recordDecl(hasName("S"))))))));
  EXPECT_TRUE(matches("class X {}; struct S { X m; void f() { m; } };",
                      memberExpr(hasObjectExpression(
                          hasType(pointsTo(recordDecl(hasName("S"))))))));
}

TEST_P(ASTMatchersTest, FieldDecl_DoesNotMatchNonFieldMembers) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("class X { void m(); };", fieldDecl(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { class m {}; };", fieldDecl(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { enum { m }; };", fieldDecl(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { enum m {}; };", fieldDecl(hasName("m"))));
}

TEST_P(ASTMatchersTest, FieldDecl_MatchesField) {
  EXPECT_TRUE(matches("struct X { int m; };", fieldDecl(hasName("m"))));
}

TEST_P(ASTMatchersTest, IsVolatileQualified) {
  EXPECT_TRUE(
      matches("volatile int i = 42;", varDecl(hasType(isVolatileQualified()))));
  EXPECT_TRUE(
      notMatches("volatile int *i;", varDecl(hasType(isVolatileQualified()))));
  EXPECT_TRUE(matches("typedef volatile int v_int; v_int i = 42;",
                      varDecl(hasType(isVolatileQualified()))));
}

TEST_P(ASTMatchersTest, IsConstQualified_MatchesConstInt) {
  EXPECT_TRUE(
      matches("const int i = 42;", varDecl(hasType(isConstQualified()))));
}

TEST_P(ASTMatchersTest, IsConstQualified_MatchesConstPointer) {
  EXPECT_TRUE(matches("int i = 42; int* const p = &i;",
                      varDecl(hasType(isConstQualified()))));
}

TEST_P(ASTMatchersTest, IsConstQualified_MatchesThroughTypedef) {
  EXPECT_TRUE(matches("typedef const int const_int; const_int i = 42;",
                      varDecl(hasType(isConstQualified()))));
  EXPECT_TRUE(matches("typedef int* int_ptr; const int_ptr p = ((int*)0);",
                      varDecl(hasType(isConstQualified()))));
}

TEST_P(ASTMatchersTest, IsConstQualified_DoesNotMatchInappropriately) {
  EXPECT_TRUE(notMatches("typedef int nonconst_int; nonconst_int i = 42;",
                         varDecl(hasType(isConstQualified()))));
  EXPECT_TRUE(
      notMatches("int const* p;", varDecl(hasType(isConstQualified()))));
}

TEST_P(ASTMatchersTest, DeclCountIs_DeclCountIsCorrect) {
  EXPECT_TRUE(matches("void f() {int i,j;}", declStmt(declCountIs(2))));
  EXPECT_TRUE(
      notMatches("void f() {int i,j; int k;}", declStmt(declCountIs(3))));
  EXPECT_TRUE(
      notMatches("void f() {int i,j, k, l;}", declStmt(declCountIs(3))));
}

TEST_P(ASTMatchersTest, EachOf_TriggersForEachMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { int a; int b; };",
      recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                        has(fieldDecl(hasName("b")).bind("v")))),
      std::make_unique<VerifyIdIsBoundTo<FieldDecl>>("v", 2)));
}

TEST_P(ASTMatchersTest, EachOf_BehavesLikeAnyOfUnlessBothMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "struct A { int a; int c; };",
      recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                        has(fieldDecl(hasName("b")).bind("v")))),
      std::make_unique<VerifyIdIsBoundTo<FieldDecl>>("v", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "struct A { int c; int b; };",
      recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                        has(fieldDecl(hasName("b")).bind("v")))),
      std::make_unique<VerifyIdIsBoundTo<FieldDecl>>("v", 1)));
  EXPECT_TRUE(
      notMatches("struct A { int c; int d; };",
                 recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                                   has(fieldDecl(hasName("b")).bind("v"))))));
}

TEST_P(ASTMatchersTest, Optionally_SubmatchersDoNotMatch) {
  EXPECT_TRUE(matchAndVerifyResultFalse(
      "class A { int a; int b; };",
      recordDecl(optionally(has(fieldDecl(hasName("c")).bind("c")))),
      std::make_unique<VerifyIdIsBoundTo<FieldDecl>>("c")));
}

// Regression test.
TEST_P(ASTMatchersTest, Optionally_SubmatchersDoNotMatchButPreserveBindings) {
  StringRef Code = "class A { int a; int b; };";
  auto Matcher = recordDecl(decl().bind("decl"),
                            optionally(has(fieldDecl(hasName("c")).bind("v"))));
  // "decl" is still bound.
  EXPECT_TRUE(matchAndVerifyResultTrue(
      Code, Matcher, std::make_unique<VerifyIdIsBoundTo<RecordDecl>>("decl")));
  // "v" is not bound, but the match still suceeded.
  EXPECT_TRUE(matchAndVerifyResultFalse(
      Code, Matcher, std::make_unique<VerifyIdIsBoundTo<FieldDecl>>("v")));
}

TEST_P(ASTMatchersTest, Optionally_SubmatchersMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { int a; int c; };",
      recordDecl(optionally(has(fieldDecl(hasName("a")).bind("v")))),
      std::make_unique<VerifyIdIsBoundTo<FieldDecl>>("v")));
}

TEST_P(ASTMatchersTest,
       IsTemplateInstantiation_MatchesImplicitClassTemplateInstantiation) {
  if (!GetParam().isCXX()) {
    return;
  }

  // Make sure that we can both match the class by name (::X) and by the type
  // the template was instantiated with (via a field).

  EXPECT_TRUE(
      matches("template <typename T> class X {}; class A {}; X<A> x;",
              cxxRecordDecl(hasName("::X"), isTemplateInstantiation())));

  EXPECT_TRUE(matches(
      "template <typename T> class X { T t; }; class A {}; X<A> x;",
      cxxRecordDecl(
          isTemplateInstantiation(),
          hasDescendant(fieldDecl(hasType(recordDecl(hasName("A"))))))));
}

TEST_P(ASTMatchersTest,
       IsTemplateInstantiation_MatchesImplicitFunctionTemplateInstantiation) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches(
      "template <typename T> void f(T t) {} class A {}; void g() { f(A()); }",
      functionDecl(hasParameter(0, hasType(recordDecl(hasName("A")))),
                   isTemplateInstantiation())));
}

TEST_P(ASTMatchersTest,
       IsTemplateInstantiation_MatchesExplicitClassTemplateInstantiation) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("template <typename T> class X { T t; }; class A {};"
                      "template class X<A>;",
                      cxxRecordDecl(isTemplateInstantiation(),
                                    hasDescendant(fieldDecl(
                                        hasType(recordDecl(hasName("A"))))))));

  // Make sure that we match the instantiation instead of the template
  // definition by checking whether the member function is present.
  EXPECT_TRUE(
      matches("template <typename T> class X { void f() { T t; } };"
              "extern template class X<int>;",
              cxxRecordDecl(isTemplateInstantiation(),
                            unless(hasDescendant(varDecl(hasName("t")))))));
}

TEST_P(
    ASTMatchersTest,
    IsTemplateInstantiation_MatchesInstantiationOfPartiallySpecializedClassTemplate) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("template <typename T> class X {};"
              "template <typename T> class X<T*> {}; class A {}; X<A*> x;",
              cxxRecordDecl(hasName("::X"), isTemplateInstantiation())));
}

TEST_P(
    ASTMatchersTest,
    IsTemplateInstantiation_MatchesInstantiationOfClassTemplateNestedInNonTemplate) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("class A {};"
              "class X {"
              "  template <typename U> class Y { U u; };"
              "  Y<A> y;"
              "};",
              cxxRecordDecl(hasName("::X::Y"), isTemplateInstantiation())));
}

TEST_P(
    ASTMatchersTest,
    IsTemplateInstantiation_DoesNotMatchInstantiationsInsideOfInstantiation) {
  if (!GetParam().isCXX()) {
    return;
  }

  // FIXME: Figure out whether this makes sense. It doesn't affect the
  // normal use case as long as the uppermost instantiation always is marked
  // as template instantiation, but it might be confusing as a predicate.
  EXPECT_TRUE(matches(
      "class A {};"
      "template <typename T> class X {"
      "  template <typename U> class Y { U u; };"
      "  Y<T> y;"
      "}; X<A> x;",
      cxxRecordDecl(hasName("::X<A>::Y"), unless(isTemplateInstantiation()))));
}

TEST_P(
    ASTMatchersTest,
    IsTemplateInstantiation_DoesNotMatchExplicitClassTemplateSpecialization) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      notMatches("template <typename T> class X {}; class A {};"
                 "template <> class X<A> {}; X<A> x;",
                 cxxRecordDecl(hasName("::X"), isTemplateInstantiation())));
}

TEST_P(ASTMatchersTest, IsTemplateInstantiation_DoesNotMatchNonTemplate) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("class A {}; class Y { A a; };",
                         cxxRecordDecl(isTemplateInstantiation())));
}

TEST_P(ASTMatchersTest, IsInstantiated_MatchesInstantiation) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("template<typename T> class A { T i; }; class Y { A<int> a; };",
              cxxRecordDecl(isInstantiated())));
}

TEST_P(ASTMatchersTest, IsInstantiated_NotMatchesDefinition) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("template<typename T> class A { T i; };",
                         cxxRecordDecl(isInstantiated())));
}

TEST_P(ASTMatchersTest, IsInTemplateInstantiation_MatchesInstantiationStmt) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("template<typename T> struct A { A() { T i; } };"
                      "class Y { A<int> a; }; Y y;",
                      declStmt(isInTemplateInstantiation())));
}

TEST_P(ASTMatchersTest, IsInTemplateInstantiation_NotMatchesDefinitionStmt) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("template<typename T> struct A { void x() { T i; } };",
                         declStmt(isInTemplateInstantiation())));
}

TEST_P(ASTMatchersTest, IsInstantiated_MatchesFunctionInstantiation) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("template<typename T> void A(T t) { T i; } void x() { A(0); }",
              functionDecl(isInstantiated())));
}

TEST_P(ASTMatchersTest, IsInstantiated_NotMatchesFunctionDefinition) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("template<typename T> void A(T t) { T i; }",
                         varDecl(isInstantiated())));
}

TEST_P(ASTMatchersTest,
       IsInTemplateInstantiation_MatchesFunctionInstantiationStmt) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("template<typename T> void A(T t) { T i; } void x() { A(0); }",
              declStmt(isInTemplateInstantiation())));
}

TEST_P(ASTMatchersTest,
       IsInTemplateInstantiation_NotMatchesFunctionDefinitionStmt) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("template<typename T> void A(T t) { T i; }",
                         declStmt(isInTemplateInstantiation())));
}

TEST_P(ASTMatchersTest, IsInTemplateInstantiation_Sharing) {
  if (!GetParam().isCXX()) {
    return;
  }

  auto Matcher = binaryOperator(unless(isInTemplateInstantiation()));
  // FIXME: Node sharing is an implementation detail, exposing it is ugly
  // and makes the matcher behave in non-obvious ways.
  EXPECT_TRUE(notMatches(
      "int j; template<typename T> void A(T t) { j += 42; } void x() { A(0); }",
      Matcher));
  EXPECT_TRUE(matches(
      "int j; template<typename T> void A(T t) { j += t; } void x() { A(0); }",
      Matcher));
}

TEST_P(ASTMatchersTest, IsInstantiationDependent_MatchesNonValueTypeDependent) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches(
      "template<typename T> void f() { (void) sizeof(sizeof(T() + T())); }",
      expr(isInstantiationDependent())));
}

TEST_P(ASTMatchersTest, IsInstantiationDependent_MatchesValueDependent) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("template<int T> int f() { return T; }",
                      expr(isInstantiationDependent())));
}

TEST_P(ASTMatchersTest, IsInstantiationDependent_MatchesTypeDependent) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("template<typename T> T f() { return T(); }",
                      expr(isInstantiationDependent())));
}

TEST_P(ASTMatchersTest, IsTypeDependent_MatchesTypeDependent) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("template<typename T> T f() { return T(); }",
                      expr(isTypeDependent())));
}

TEST_P(ASTMatchersTest, IsTypeDependent_NotMatchesValueDependent) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("template<int T> int f() { return T; }",
                         expr(isTypeDependent())));
}

TEST_P(ASTMatchersTest, IsValueDependent_MatchesValueDependent) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("template<int T> int f() { return T; }",
                      expr(isValueDependent())));
}

TEST_P(ASTMatchersTest, IsValueDependent_MatchesTypeDependent) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("template<typename T> T f() { return T(); }",
                      expr(isValueDependent())));
}

TEST_P(ASTMatchersTest, IsValueDependent_MatchesInstantiationDependent) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches(
      "template<typename T> void f() { (void) sizeof(sizeof(T() + T())); }",
      expr(isValueDependent())));
}

TEST_P(ASTMatchersTest,
       IsExplicitTemplateSpecialization_DoesNotMatchPrimaryTemplate) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("template <typename T> class X {};",
                         cxxRecordDecl(isExplicitTemplateSpecialization())));
  EXPECT_TRUE(notMatches("template <typename T> void f(T t);",
                         functionDecl(isExplicitTemplateSpecialization())));
}

TEST_P(
    ASTMatchersTest,
    IsExplicitTemplateSpecialization_DoesNotMatchExplicitTemplateInstantiations) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      notMatches("template <typename T> class X {};"
                 "template class X<int>; extern template class X<long>;",
                 cxxRecordDecl(isExplicitTemplateSpecialization())));
  EXPECT_TRUE(
      notMatches("template <typename T> void f(T t) {}"
                 "template void f(int t); extern template void f(long t);",
                 functionDecl(isExplicitTemplateSpecialization())));
}

TEST_P(
    ASTMatchersTest,
    IsExplicitTemplateSpecialization_DoesNotMatchImplicitTemplateInstantiations) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("template <typename T> class X {}; X<int> x;",
                         cxxRecordDecl(isExplicitTemplateSpecialization())));
  EXPECT_TRUE(
      notMatches("template <typename T> void f(T t); void g() { f(10); }",
                 functionDecl(isExplicitTemplateSpecialization())));
}

TEST_P(
    ASTMatchersTest,
    IsExplicitTemplateSpecialization_MatchesExplicitTemplateSpecializations) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("template <typename T> class X {};"
                      "template<> class X<int> {};",
                      cxxRecordDecl(isExplicitTemplateSpecialization())));
  EXPECT_TRUE(matches("template <typename T> void f(T t) {}"
                      "template<> void f(int t) {}",
                      functionDecl(isExplicitTemplateSpecialization())));
}

TEST_P(ASTMatchersTest, IsNoReturn) {
  EXPECT_TRUE(notMatches("void func();", functionDecl(isNoReturn())));
  EXPECT_TRUE(notMatches("void func() {}", functionDecl(isNoReturn())));

  EXPECT_TRUE(matches("__attribute__((noreturn)) void func();",
                      functionDecl(isNoReturn())));
  EXPECT_TRUE(matches("__attribute__((noreturn)) void func() {}",
                      functionDecl(isNoReturn())));

  EXPECT_TRUE(matches("_Noreturn void func();", functionDecl(isNoReturn())));
  EXPECT_TRUE(matches("_Noreturn void func() {}", functionDecl(isNoReturn())));
}

TEST_P(ASTMatchersTest, IsNoReturn_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      notMatches("struct S { void func(); };", functionDecl(isNoReturn())));
  EXPECT_TRUE(
      notMatches("struct S { void func() {} };", functionDecl(isNoReturn())));

  EXPECT_TRUE(notMatches("struct S { static void func(); };",
                         functionDecl(isNoReturn())));
  EXPECT_TRUE(notMatches("struct S { static void func() {} };",
                         functionDecl(isNoReturn())));

  EXPECT_TRUE(notMatches("struct S { S(); };", functionDecl(isNoReturn())));
  EXPECT_TRUE(notMatches("struct S { S() {} };", functionDecl(isNoReturn())));

  // ---

  EXPECT_TRUE(matches("struct S { __attribute__((noreturn)) void func(); };",
                      functionDecl(isNoReturn())));
  EXPECT_TRUE(matches("struct S { __attribute__((noreturn)) void func() {} };",
                      functionDecl(isNoReturn())));

  EXPECT_TRUE(
      matches("struct S { __attribute__((noreturn)) static void func(); };",
              functionDecl(isNoReturn())));
  EXPECT_TRUE(
      matches("struct S { __attribute__((noreturn)) static void func() {} };",
              functionDecl(isNoReturn())));

  EXPECT_TRUE(matches("struct S { __attribute__((noreturn)) S(); };",
                      functionDecl(isNoReturn())));
  EXPECT_TRUE(matches("struct S { __attribute__((noreturn)) S() {} };",
                      functionDecl(isNoReturn())));
}

TEST_P(ASTMatchersTest, IsNoReturn_CXX11Attribute) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("[[noreturn]] void func();", functionDecl(isNoReturn())));
  EXPECT_TRUE(
      matches("[[noreturn]] void func() {}", functionDecl(isNoReturn())));

  EXPECT_TRUE(matches("struct S { [[noreturn]] void func(); };",
                      functionDecl(isNoReturn())));
  EXPECT_TRUE(matches("struct S { [[noreturn]] void func() {} };",
                      functionDecl(isNoReturn())));

  EXPECT_TRUE(matches("struct S { [[noreturn]] static void func(); };",
                      functionDecl(isNoReturn())));
  EXPECT_TRUE(matches("struct S { [[noreturn]] static void func() {} };",
                      functionDecl(isNoReturn())));

  EXPECT_TRUE(
      matches("struct S { [[noreturn]] S(); };", functionDecl(isNoReturn())));
  EXPECT_TRUE(
      matches("struct S { [[noreturn]] S() {} };", functionDecl(isNoReturn())));
}

TEST_P(ASTMatchersTest, BooleanType) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `booleanType()` that does not depend on C++.
    return;
  }

  EXPECT_TRUE(matches("struct S { bool func(); };",
                      cxxMethodDecl(returns(booleanType()))));
  EXPECT_TRUE(notMatches("struct S { void func(); };",
                         cxxMethodDecl(returns(booleanType()))));
}

TEST_P(ASTMatchersTest, VoidType) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `voidType()` that does not depend on C++.
    return;
  }

  EXPECT_TRUE(matches("struct S { void func(); };",
                      cxxMethodDecl(returns(voidType()))));
}

TEST_P(ASTMatchersTest, RealFloatingPointType) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `realFloatingPointType()` that does not depend on
    // C++.
    return;
  }

  EXPECT_TRUE(matches("struct S { float func(); };",
                      cxxMethodDecl(returns(realFloatingPointType()))));
  EXPECT_TRUE(notMatches("struct S { int func(); };",
                         cxxMethodDecl(returns(realFloatingPointType()))));
  EXPECT_TRUE(matches("struct S { long double func(); };",
                      cxxMethodDecl(returns(realFloatingPointType()))));
}

TEST_P(ASTMatchersTest, ArrayType) {
  EXPECT_TRUE(matches("int a[] = {2,3};", arrayType()));
  EXPECT_TRUE(matches("int a[42];", arrayType()));
  EXPECT_TRUE(matches("void f(int b) { int a[b]; }", arrayType()));

  EXPECT_TRUE(notMatches("struct A {}; struct A a[7];",
                         arrayType(hasElementType(builtinType()))));

  EXPECT_TRUE(matches("int const a[] = { 2, 3 };",
                      qualType(arrayType(hasElementType(builtinType())))));
  EXPECT_TRUE(matches(
      "int const a[] = { 2, 3 };",
      qualType(isConstQualified(), arrayType(hasElementType(builtinType())))));
  EXPECT_TRUE(matches("typedef const int T; T x[] = { 1, 2 };",
                      qualType(isConstQualified(), arrayType())));

  EXPECT_TRUE(notMatches(
      "int a[] = { 2, 3 };",
      qualType(isConstQualified(), arrayType(hasElementType(builtinType())))));
  EXPECT_TRUE(notMatches(
      "int a[] = { 2, 3 };",
      qualType(arrayType(hasElementType(isConstQualified(), builtinType())))));
  EXPECT_TRUE(notMatches("int const a[] = { 2, 3 };",
                         qualType(arrayType(hasElementType(builtinType())),
                                  unless(isConstQualified()))));

  EXPECT_TRUE(
      matches("int a[2];", constantArrayType(hasElementType(builtinType()))));
  EXPECT_TRUE(matches("const int a = 0;", qualType(isInteger())));
}

TEST_P(ASTMatchersTest, DecayedType) {
  EXPECT_TRUE(
      matches("void f(int i[]);",
              valueDecl(hasType(decayedType(hasDecayedType(pointerType()))))));
  EXPECT_TRUE(notMatches("int i[7];", decayedType()));
}

TEST_P(ASTMatchersTest, ComplexType) {
  EXPECT_TRUE(matches("_Complex float f;", complexType()));
  EXPECT_TRUE(
      matches("_Complex float f;", complexType(hasElementType(builtinType()))));
  EXPECT_TRUE(notMatches("_Complex float f;",
                         complexType(hasElementType(isInteger()))));
}

TEST_P(ASTMatchersTest, IsAnonymous) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("namespace N {}", namespaceDecl(isAnonymous())));
  EXPECT_TRUE(matches("namespace {}", namespaceDecl(isAnonymous())));
}

TEST_P(ASTMatchersTest, InStdNamespace) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("class vector {};"
                         "namespace foo {"
                         "  class vector {};"
                         "}"
                         "namespace foo {"
                         "  namespace std {"
                         "    class vector {};"
                         "  }"
                         "}",
                         cxxRecordDecl(hasName("vector"), isInStdNamespace())));

  EXPECT_TRUE(matches("namespace std {"
                      "  class vector {};"
                      "}",
                      cxxRecordDecl(hasName("vector"), isInStdNamespace())));
}

TEST_P(ASTMatchersTest, InStdNamespace_CXX11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("namespace std {"
                      "  inline namespace __1 {"
                      "    class vector {};"
                      "  }"
                      "}",
                      cxxRecordDecl(hasName("vector"), isInStdNamespace())));
  EXPECT_TRUE(notMatches("namespace std {"
                         "  inline namespace __1 {"
                         "    inline namespace __fs {"
                         "      namespace filesystem {"
                         "        inline namespace v1 {"
                         "          class path {};"
                         "        }"
                         "      }"
                         "    }"
                         "  }"
                         "}",
                         cxxRecordDecl(hasName("path"), isInStdNamespace())));
  EXPECT_TRUE(
      matches("namespace std {"
              "  inline namespace __1 {"
              "    inline namespace __fs {"
              "      namespace filesystem {"
              "        inline namespace v1 {"
              "          class path {};"
              "        }"
              "      }"
              "    }"
              "  }"
              "}",
              cxxRecordDecl(hasName("path"),
                            hasAncestor(namespaceDecl(hasName("filesystem"),
                                                      isInStdNamespace())))));
}

TEST_P(ASTMatchersTest, EqualsBoundNodeMatcher_QualType) {
  EXPECT_TRUE(matches(
      "int i = 1;", varDecl(hasType(qualType().bind("type")),
                            hasInitializer(ignoringParenImpCasts(
                                hasType(qualType(equalsBoundNode("type"))))))));
  EXPECT_TRUE(notMatches("int i = 1.f;",
                         varDecl(hasType(qualType().bind("type")),
                                 hasInitializer(ignoringParenImpCasts(hasType(
                                     qualType(equalsBoundNode("type"))))))));
}

TEST_P(ASTMatchersTest, EqualsBoundNodeMatcher_NonMatchingTypes) {
  EXPECT_TRUE(notMatches(
      "int i = 1;", varDecl(namedDecl(hasName("i")).bind("name"),
                            hasInitializer(ignoringParenImpCasts(
                                hasType(qualType(equalsBoundNode("type"))))))));
}

TEST_P(ASTMatchersTest, EqualsBoundNodeMatcher_Stmt) {
  EXPECT_TRUE(
      matches("void f() { if(1) {} }",
              stmt(allOf(ifStmt().bind("if"),
                         hasParent(stmt(has(stmt(equalsBoundNode("if")))))))));

  EXPECT_TRUE(notMatches(
      "void f() { if(1) { if (1) {} } }",
      stmt(allOf(ifStmt().bind("if"), has(stmt(equalsBoundNode("if")))))));
}

TEST_P(ASTMatchersTest, EqualsBoundNodeMatcher_Decl) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `equalsBoundNode()` for declarations that does not
    // depend on C++.
    return;
  }

  EXPECT_TRUE(matches(
      "class X { class Y {}; };",
      decl(allOf(recordDecl(hasName("::X::Y")).bind("record"),
                 hasParent(decl(has(decl(equalsBoundNode("record")))))))));

  EXPECT_TRUE(notMatches("class X { class Y {}; };",
                         decl(allOf(recordDecl(hasName("::X")).bind("record"),
                                    has(decl(equalsBoundNode("record")))))));
}

TEST_P(ASTMatchersTest, EqualsBoundNodeMatcher_Type) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `equalsBoundNode()` for types that does not depend
    // on C++.
    return;
  }
  EXPECT_TRUE(matches(
      "class X { int a; int b; };",
      recordDecl(
          has(fieldDecl(hasName("a"), hasType(type().bind("t")))),
          has(fieldDecl(hasName("b"), hasType(type(equalsBoundNode("t"))))))));

  EXPECT_TRUE(notMatches(
      "class X { int a; double b; };",
      recordDecl(
          has(fieldDecl(hasName("a"), hasType(type().bind("t")))),
          has(fieldDecl(hasName("b"), hasType(type(equalsBoundNode("t"))))))));
}

TEST_P(ASTMatchersTest, EqualsBoundNodeMatcher_UsingForEachDescendant) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "int f() {"
      "  if (1) {"
      "    int i = 9;"
      "  }"
      "  int j = 10;"
      "  {"
      "    float k = 9.0;"
      "  }"
      "  return 0;"
      "}",
      // Look for variable declarations within functions whose type is the same
      // as the function return type.
      functionDecl(
          returns(qualType().bind("type")),
          forEachDescendant(varDecl(hasType(qualType(equalsBoundNode("type"))))
                                .bind("decl"))),
      // Only i and j should match, not k.
      std::make_unique<VerifyIdIsBoundTo<VarDecl>>("decl", 2)));
}

TEST_P(ASTMatchersTest, EqualsBoundNodeMatcher_FiltersMatchedCombinations) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f() {"
      "  int x;"
      "  double d;"
      "  x = d + x - d + x;"
      "}",
      functionDecl(
          hasName("f"), forEachDescendant(varDecl().bind("d")),
          forEachDescendant(declRefExpr(to(decl(equalsBoundNode("d")))))),
      std::make_unique<VerifyIdIsBoundTo<VarDecl>>("d", 5)));
}

TEST_P(ASTMatchersTest,
       EqualsBoundNodeMatcher_UnlessDescendantsOfAncestorsMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "struct StringRef { int size() const; const char* data() const; };"
      "void f(StringRef v) {"
      "  v.data();"
      "}",
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasName("data"))),
          on(declRefExpr(to(
              varDecl(hasType(recordDecl(hasName("StringRef")))).bind("var")))),
          unless(hasAncestor(stmt(hasDescendant(cxxMemberCallExpr(
              callee(cxxMethodDecl(anyOf(hasName("size"), hasName("length")))),
              on(declRefExpr(to(varDecl(equalsBoundNode("var")))))))))))
          .bind("data"),
      std::make_unique<VerifyIdIsBoundTo<Expr>>("data", 1)));

  EXPECT_FALSE(matches(
      "struct StringRef { int size() const; const char* data() const; };"
      "void f(StringRef v) {"
      "  v.data();"
      "  v.size();"
      "}",
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasName("data"))),
          on(declRefExpr(to(
              varDecl(hasType(recordDecl(hasName("StringRef")))).bind("var")))),
          unless(hasAncestor(stmt(hasDescendant(cxxMemberCallExpr(
              callee(cxxMethodDecl(anyOf(hasName("size"), hasName("length")))),
              on(declRefExpr(to(varDecl(equalsBoundNode("var")))))))))))
          .bind("data")));
}

TEST_P(ASTMatchersTest, NullPointerConstant) {
  EXPECT_TRUE(matches("#define NULL ((void *)0)\n"
                      "void *v1 = NULL;",
                      expr(nullPointerConstant())));
  EXPECT_TRUE(matches("char *cp = (char *)0;", expr(nullPointerConstant())));
  EXPECT_TRUE(matches("int *ip = 0;", expr(nullPointerConstant())));
  EXPECT_FALSE(matches("int i = 0;", expr(nullPointerConstant())));
}

TEST_P(ASTMatchersTest, NullPointerConstant_GNUNull) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("void *p = __null;", expr(nullPointerConstant())));
}

TEST_P(ASTMatchersTest, NullPointerConstant_GNUNullInTemplate) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  const char kTest[] = R"(
    template <typename T>
    struct MyTemplate {
      MyTemplate() : field_(__null) {}
      T* field_;
    };
  )";
  EXPECT_TRUE(matches(kTest, expr(nullPointerConstant())));
}

TEST_P(ASTMatchersTest, NullPointerConstant_CXX11Nullptr) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("void *p = nullptr;", expr(nullPointerConstant())));
}

TEST_P(ASTMatchersTest, HasExternalFormalLinkage) {
  EXPECT_TRUE(matches("int a = 0;",
                      namedDecl(hasName("a"), hasExternalFormalLinkage())));
  EXPECT_TRUE(notMatches("static int a = 0;",
                         namedDecl(hasName("a"), hasExternalFormalLinkage())));
  EXPECT_TRUE(notMatches("static void f(void) { int a = 0; }",
                         namedDecl(hasName("a"), hasExternalFormalLinkage())));
  EXPECT_TRUE(notMatches("void f(void) { int a = 0; }",
                         namedDecl(hasName("a"), hasExternalFormalLinkage())));
}

TEST_P(ASTMatchersTest, HasExternalFormalLinkage_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(notMatches("namespace { int a = 0; }",
                         namedDecl(hasName("a"), hasExternalFormalLinkage())));
}

TEST_P(ASTMatchersTest, HasDefaultArgument) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(
      matches("void x(int val = 0) {}", parmVarDecl(hasDefaultArgument())));
  EXPECT_TRUE(
      notMatches("void x(int val) {}", parmVarDecl(hasDefaultArgument())));
}

TEST_P(ASTMatchersTest, IsAtPosition) {
  EXPECT_TRUE(matches("void x(int a, int b) {}", parmVarDecl(isAtPosition(1))));
  EXPECT_TRUE(matches("void x(int a, int b) {}", parmVarDecl(isAtPosition(0))));
  EXPECT_TRUE(matches("void x(int a, int b) {}", parmVarDecl(isAtPosition(1))));
  EXPECT_TRUE(notMatches("void x(int val) {}", parmVarDecl(isAtPosition(1))));
}

TEST_P(ASTMatchersTest, IsAtPosition_FunctionDecl) {
  EXPECT_TRUE(matches("void x(int a);", parmVarDecl(isAtPosition(0))));
  EXPECT_TRUE(matches("void x(int a, int b);", parmVarDecl(isAtPosition(0))));
  EXPECT_TRUE(matches("void x(int a, int b);", parmVarDecl(isAtPosition(1))));
  EXPECT_TRUE(notMatches("void x(int val);", parmVarDecl(isAtPosition(1))));
}

TEST_P(ASTMatchersTest, IsAtPosition_Lambda) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(
      matches("void x() { [](int a) {};  }", parmVarDecl(isAtPosition(0))));
  EXPECT_TRUE(matches("void x() { [](int a, int b) {}; }",
                      parmVarDecl(isAtPosition(0))));
  EXPECT_TRUE(matches("void x() { [](int a, int b) {}; }",
                      parmVarDecl(isAtPosition(1))));
  EXPECT_TRUE(
      notMatches("void x() { [](int val) {}; }", parmVarDecl(isAtPosition(1))));
}

TEST_P(ASTMatchersTest, IsAtPosition_BlockDecl) {
  EXPECT_TRUE(matchesObjC(
      "void func()  { void (^my_block)(int arg) = ^void(int arg) {}; } ",
      parmVarDecl(isAtPosition(0))));

  EXPECT_TRUE(matchesObjC("void func()  { void (^my_block)(int x, int y) = "
                          "^void(int x, int y) {}; } ",
                          parmVarDecl(isAtPosition(1))));

  EXPECT_TRUE(notMatchesObjC(
      "void func()  { void (^my_block)(int arg) = ^void(int arg) {}; } ",
      parmVarDecl(isAtPosition(1))));
}

TEST_P(ASTMatchersTest, IsArray) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("struct MyClass {}; MyClass *p1 = new MyClass[10];",
                      cxxNewExpr(isArray())));
}

TEST_P(ASTMatchersTest, HasArraySize) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("struct MyClass {}; MyClass *p1 = new MyClass[10];",
                      cxxNewExpr(hasArraySize(
                          ignoringParenImpCasts(integerLiteral(equals(10)))))));
}

TEST_P(ASTMatchersTest, HasDefinition_MatchesStructDefinition) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("struct x {};", cxxRecordDecl(hasDefinition())));
  EXPECT_TRUE(notMatches("struct x;", cxxRecordDecl(hasDefinition())));
}

TEST_P(ASTMatchersTest, HasDefinition_MatchesClassDefinition) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class x {};", cxxRecordDecl(hasDefinition())));
  EXPECT_TRUE(notMatches("class x;", cxxRecordDecl(hasDefinition())));
}

TEST_P(ASTMatchersTest, HasDefinition_MatchesUnionDefinition) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("union x {};", cxxRecordDecl(hasDefinition())));
  EXPECT_TRUE(notMatches("union x;", cxxRecordDecl(hasDefinition())));
}

TEST_P(ASTMatchersTest, IsScoped_MatchesScopedEnum) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("enum class X {};", enumDecl(isScoped())));
}

TEST_P(ASTMatchersTest, IsScoped_NotMatchesRegularEnum) {
  EXPECT_TRUE(notMatches("enum E { E1 };", enumDecl(isScoped())));
}

TEST_P(ASTMatchersTest, IsStruct) {
  EXPECT_TRUE(matches("struct S {};", tagDecl(isStruct())));
}

TEST_P(ASTMatchersTest, IsUnion) {
  EXPECT_TRUE(matches("union U {};", tagDecl(isUnion())));
}

TEST_P(ASTMatchersTest, IsEnum) {
  EXPECT_TRUE(matches("enum E { E1 };", tagDecl(isEnum())));
}

TEST_P(ASTMatchersTest, IsClass) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches("class C {};", tagDecl(isClass())));
}

TEST_P(ASTMatchersTest, HasTrailingReturn_MatchesTrailingReturn) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("auto Y() -> int { return 0; }",
                      functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(matches("auto X() -> int;", functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(
      notMatches("int X() { return 0; }", functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(notMatches("int X();", functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(notMatches("void X();", functionDecl(hasTrailingReturn())));
}

TEST_P(ASTMatchersTest, HasTrailingReturn_MatchesLambdaTrailingReturn) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(matches(
      "auto lambda2 = [](double x, double y) -> double {return x + y;};",
      functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(
      notMatches("auto lambda2 = [](double x, double y) {return x + y;};",
                 functionDecl(hasTrailingReturn())));
}

TEST_P(ASTMatchersTest, IsAssignmentOperator) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher BinAsgmtOperator = binaryOperator(isAssignmentOperator());
  StatementMatcher CXXAsgmtOperator =
      cxxOperatorCallExpr(isAssignmentOperator());

  EXPECT_TRUE(matches("void x() { int a; a += 1; }", BinAsgmtOperator));
  EXPECT_TRUE(matches("void x() { int a; a = 2; }", BinAsgmtOperator));
  EXPECT_TRUE(matches("void x() { int a; a &= 3; }", BinAsgmtOperator));
  EXPECT_TRUE(matches("struct S { S& operator=(const S&); };"
                      "void x() { S s1, s2; s1 = s2; }",
                      CXXAsgmtOperator));
  EXPECT_TRUE(
      notMatches("void x() { int a; if(a == 0) return; }", BinAsgmtOperator));
}

TEST_P(ASTMatchersTest, IsComparisonOperator) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher BinCompOperator = binaryOperator(isComparisonOperator());
  StatementMatcher CXXCompOperator =
      cxxOperatorCallExpr(isComparisonOperator());

  EXPECT_TRUE(matches("void x() { int a; a == 1; }", BinCompOperator));
  EXPECT_TRUE(matches("void x() { int a; a > 2; }", BinCompOperator));
  EXPECT_TRUE(matches("struct S { bool operator==(const S&); };"
                      "void x() { S s1, s2; bool b1 = s1 == s2; }",
                      CXXCompOperator));
  EXPECT_TRUE(
      notMatches("void x() { int a; if(a = 0) return; }", BinCompOperator));
}

TEST_P(ASTMatchersTest, HasInit) {
  if (!GetParam().isCXX11OrLater()) {
    // FIXME: Add a test for `hasInit()` that does not depend on C++.
    return;
  }

  EXPECT_TRUE(matches("int x{0};", initListExpr(hasInit(0, expr()))));
  EXPECT_FALSE(matches("int x{0};", initListExpr(hasInit(1, expr()))));
  EXPECT_FALSE(matches("int x;", initListExpr(hasInit(0, expr()))));
}

TEST_P(ASTMatchersTest, IsMain) {
  EXPECT_TRUE(matches("int main() {}", functionDecl(isMain())));

  EXPECT_TRUE(notMatches("int main2() {}", functionDecl(isMain())));
}

TEST_P(ASTMatchersTest, OMPExecutableDirective_IsStandaloneDirective) {
  auto Matcher = ompExecutableDirective(isStandaloneDirective());

  StringRef Source0 = R"(
void x() {
#pragma omp parallel
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source0, Matcher));

  StringRef Source1 = R"(
void x() {
#pragma omp taskyield
})";
  EXPECT_TRUE(matchesWithOpenMP(Source1, Matcher));
}

TEST_P(ASTMatchersTest, OMPExecutableDirective_HasStructuredBlock) {
  StringRef Source0 = R"(
void x() {
#pragma omp parallel
;
})";
  EXPECT_TRUE(matchesWithOpenMP(
      Source0, ompExecutableDirective(hasStructuredBlock(nullStmt()))));

  StringRef Source1 = R"(
void x() {
#pragma omp parallel
{;}
})";
  EXPECT_TRUE(notMatchesWithOpenMP(
      Source1, ompExecutableDirective(hasStructuredBlock(nullStmt()))));
  EXPECT_TRUE(matchesWithOpenMP(
      Source1, ompExecutableDirective(hasStructuredBlock(compoundStmt()))));

  StringRef Source2 = R"(
void x() {
#pragma omp taskyield
{;}
})";
  EXPECT_TRUE(notMatchesWithOpenMP(
      Source2, ompExecutableDirective(hasStructuredBlock(anything()))));
}

TEST_P(ASTMatchersTest, OMPExecutableDirective_HasClause) {
  auto Matcher = ompExecutableDirective(hasAnyClause(anything()));

  StringRef Source0 = R"(
void x() {
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source0, Matcher));

  StringRef Source1 = R"(
void x() {
#pragma omp parallel
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source1, Matcher));

  StringRef Source2 = R"(
void x() {
#pragma omp parallel default(none)
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source2, Matcher));

  StringRef Source3 = R"(
void x() {
#pragma omp parallel default(shared)
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source3, Matcher));

  StringRef Source4 = R"(
void x() {
#pragma omp parallel default(firstprivate)
;
})";
  EXPECT_TRUE(matchesWithOpenMP51(Source4, Matcher));

  StringRef Source5 = R"(
void x(int x) {
#pragma omp parallel num_threads(x)
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source5, Matcher));
}

TEST_P(ASTMatchersTest, OMPDefaultClause_IsNoneKind) {
  auto Matcher =
      ompExecutableDirective(hasAnyClause(ompDefaultClause(isNoneKind())));

  StringRef Source0 = R"(
void x() {
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source0, Matcher));

  StringRef Source1 = R"(
void x() {
#pragma omp parallel
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source1, Matcher));

  StringRef Source2 = R"(
void x() {
#pragma omp parallel default(none)
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source2, Matcher));

  StringRef Source3 = R"(
void x() {
#pragma omp parallel default(shared)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source3, Matcher));

  StringRef Source4 = R"(
void x(int x) {
#pragma omp parallel default(firstprivate)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP51(Source4, Matcher));

  const std::string Source5 = R"(
void x(int x) {
#pragma omp parallel num_threads(x)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source5, Matcher));
}

TEST_P(ASTMatchersTest, OMPDefaultClause_IsSharedKind) {
  auto Matcher =
      ompExecutableDirective(hasAnyClause(ompDefaultClause(isSharedKind())));

  StringRef Source0 = R"(
void x() {
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source0, Matcher));

  StringRef Source1 = R"(
void x() {
#pragma omp parallel
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source1, Matcher));

  StringRef Source2 = R"(
void x() {
#pragma omp parallel default(shared)
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source2, Matcher));

  StringRef Source3 = R"(
void x() {
#pragma omp parallel default(none)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source3, Matcher));

  StringRef Source4 = R"(
void x(int x) {
#pragma omp parallel default(firstprivate)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP51(Source4, Matcher));

  const std::string Source5 = R"(
void x(int x) {
#pragma omp parallel num_threads(x)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source5, Matcher));
}

TEST(OMPDefaultClause, isFirstPrivateKind) {
  auto Matcher = ompExecutableDirective(
      hasAnyClause(ompDefaultClause(isFirstPrivateKind())));

  const std::string Source0 = R"(
void x() {
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source0, Matcher));

  const std::string Source1 = R"(
void x() {
#pragma omp parallel
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source1, Matcher));

  const std::string Source2 = R"(
void x() {
#pragma omp parallel default(shared)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source2, Matcher));

  const std::string Source3 = R"(
void x() {
#pragma omp parallel default(none)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source3, Matcher));

  const std::string Source4 = R"(
void x(int x) {
#pragma omp parallel default(firstprivate)
;
})";
  EXPECT_TRUE(matchesWithOpenMP51(Source4, Matcher));

  const std::string Source5 = R"(
void x(int x) {
#pragma omp parallel num_threads(x)
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source5, Matcher));
}

TEST_P(ASTMatchersTest, OMPExecutableDirective_IsAllowedToContainClauseKind) {
  auto Matcher = ompExecutableDirective(
      isAllowedToContainClauseKind(llvm::omp::OMPC_default));

  StringRef Source0 = R"(
void x() {
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source0, Matcher));

  StringRef Source1 = R"(
void x() {
#pragma omp parallel
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source1, Matcher));

  StringRef Source2 = R"(
void x() {
#pragma omp parallel default(none)
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source2, Matcher));

  StringRef Source3 = R"(
void x() {
#pragma omp parallel default(shared)
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source3, Matcher));

  StringRef Source4 = R"(
void x() {
#pragma omp parallel default(firstprivate)
;
})";
  EXPECT_TRUE(matchesWithOpenMP51(Source4, Matcher));

  StringRef Source5 = R"(
void x(int x) {
#pragma omp parallel num_threads(x)
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source5, Matcher));

  StringRef Source6 = R"(
void x() {
#pragma omp taskyield
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source6, Matcher));

  StringRef Source7 = R"(
void x() {
#pragma omp task
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source7, Matcher));
}

TEST_P(ASTMatchersTest, HasAnyBase_DirectBase) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches(
      "struct Base {};"
      "struct ExpectedMatch : Base {};",
      cxxRecordDecl(hasName("ExpectedMatch"),
                    hasAnyBase(hasType(cxxRecordDecl(hasName("Base")))))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IndirectBase) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches(
      "struct Base {};"
      "struct Intermediate : Base {};"
      "struct ExpectedMatch : Intermediate {};",
      cxxRecordDecl(hasName("ExpectedMatch"),
                    hasAnyBase(hasType(cxxRecordDecl(hasName("Base")))))));
}

TEST_P(ASTMatchersTest, HasAnyBase_NoBase) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("struct Foo {};"
                         "struct Bar {};",
                         cxxRecordDecl(hasAnyBase(hasType(cxxRecordDecl())))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPublic_Public) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class Base {};"
                      "class Derived : public Base {};",
                      cxxRecordDecl(hasAnyBase(isPublic()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPublic_DefaultAccessSpecifierPublic) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class Base {};"
                      "struct Derived : Base {};",
                      cxxRecordDecl(hasAnyBase(isPublic()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPublic_Private) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "class Derived : private Base {};",
                         cxxRecordDecl(hasAnyBase(isPublic()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPublic_DefaultAccessSpecifierPrivate) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "class Derived : Base {};",
                         cxxRecordDecl(hasAnyBase(isPublic()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPublic_Protected) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "class Derived : protected Base {};",
                         cxxRecordDecl(hasAnyBase(isPublic()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPrivate_Private) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class Base {};"
                      "class Derived : private Base {};",
                      cxxRecordDecl(hasAnyBase(isPrivate()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPrivate_DefaultAccessSpecifierPrivate) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("struct Base {};"
                      "class Derived : Base {};",
                      cxxRecordDecl(hasAnyBase(isPrivate()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPrivate_Public) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "class Derived : public Base {};",
                         cxxRecordDecl(hasAnyBase(isPrivate()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPrivate_DefaultAccessSpecifierPublic) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "struct Derived : Base {};",
                         cxxRecordDecl(hasAnyBase(isPrivate()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsPrivate_Protected) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "class Derived : protected Base {};",
                         cxxRecordDecl(hasAnyBase(isPrivate()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsProtected_Protected) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class Base {};"
                      "class Derived : protected Base {};",
                      cxxRecordDecl(hasAnyBase(isProtected()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsProtected_Public) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "class Derived : public Base {};",
                         cxxRecordDecl(hasAnyBase(isProtected()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsProtected_Private) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "class Derived : private Base {};",
                         cxxRecordDecl(hasAnyBase(isProtected()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsVirtual_Directly) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class Base {};"
                      "class Derived : virtual Base {};",
                      cxxRecordDecl(hasAnyBase(isVirtual()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsVirtual_Indirectly) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(
      matches("class Base {};"
              "class Intermediate : virtual Base {};"
              "class Derived : Intermediate {};",
              cxxRecordDecl(hasName("Derived"), hasAnyBase(isVirtual()))));
}

TEST_P(ASTMatchersTest, HasAnyBase_IsVirtual_NoVirtualBase) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Base {};"
                         "class Derived : Base {};",
                         cxxRecordDecl(hasAnyBase(isVirtual()))));
}

TEST_P(ASTMatchersTest, HasDirectBase) {
  if (!GetParam().isCXX()) {
    return;
  }

  DeclarationMatcher ClassHasAnyDirectBase =
      cxxRecordDecl(hasDirectBase(cxxBaseSpecifier()));
  EXPECT_TRUE(notMatches("class X {};", ClassHasAnyDirectBase));
  EXPECT_TRUE(matches("class X {}; class Y : X {};", ClassHasAnyDirectBase));
  EXPECT_TRUE(matches("class X {}; class Y : public virtual X {};",
                      ClassHasAnyDirectBase));

  EXPECT_TRUE(matches(
      R"cc(
    class Base {};
    class Derived : Base{};
    )cc",
      cxxRecordDecl(hasName("Derived"),
                    hasDirectBase(hasType(cxxRecordDecl(hasName("Base")))))));

  StringRef MultiDerived = R"cc(
    class Base {};
    class Base2 {};
    class Derived : Base, Base2{};
    )cc";

  EXPECT_TRUE(matches(
      MultiDerived,
      cxxRecordDecl(hasName("Derived"),
                    hasDirectBase(hasType(cxxRecordDecl(hasName("Base")))))));
  EXPECT_TRUE(matches(
      MultiDerived,
      cxxRecordDecl(hasName("Derived"),
                    hasDirectBase(hasType(cxxRecordDecl(hasName("Base2")))))));

  StringRef Indirect = R"cc(
    class Base {};
    class Intermediate : Base {};
    class Derived : Intermediate{};
    )cc";

  EXPECT_TRUE(
      matches(Indirect, cxxRecordDecl(hasName("Derived"),
                                      hasDirectBase(hasType(cxxRecordDecl(
                                          hasName("Intermediate")))))));
  EXPECT_TRUE(notMatches(
      Indirect,
      cxxRecordDecl(hasName("Derived"),
                    hasDirectBase(hasType(cxxRecordDecl(hasName("Base")))))));
}
} // namespace ast_matchers
} // namespace clang
