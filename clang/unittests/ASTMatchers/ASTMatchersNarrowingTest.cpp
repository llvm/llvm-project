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


TEST(AllOf, AllOverloadsWork) {
  const char Program[] =
      "struct T { };"
      "int f(int, T*, int, int);"
      "void g(int x) { T t; f(x, &t, 3, 4); }";
  EXPECT_TRUE(matches(Program,
      callExpr(allOf(callee(functionDecl(hasName("f"))),
                     hasArgument(0, declRefExpr(to(varDecl())))))));
  EXPECT_TRUE(matches(Program,
      callExpr(allOf(callee(functionDecl(hasName("f"))),
                     hasArgument(0, declRefExpr(to(varDecl()))),
                     hasArgument(1, hasType(pointsTo(
                                        recordDecl(hasName("T")))))))));
  EXPECT_TRUE(matches(Program,
      callExpr(allOf(callee(functionDecl(hasName("f"))),
                     hasArgument(0, declRefExpr(to(varDecl()))),
                     hasArgument(1, hasType(pointsTo(
                                        recordDecl(hasName("T"))))),
                     hasArgument(2, integerLiteral(equals(3)))))));
  EXPECT_TRUE(matches(Program,
      callExpr(allOf(callee(functionDecl(hasName("f"))),
                     hasArgument(0, declRefExpr(to(varDecl()))),
                     hasArgument(1, hasType(pointsTo(
                                        recordDecl(hasName("T"))))),
                     hasArgument(2, integerLiteral(equals(3))),
                     hasArgument(3, integerLiteral(equals(4)))))));
}

TEST(DeclarationMatcher, MatchHas) {
  DeclarationMatcher HasClassX = recordDecl(has(recordDecl(hasName("X"))));
  EXPECT_TRUE(matches("class Y { class X {}; };", HasClassX));
  EXPECT_TRUE(matches("class X {};", HasClassX));

  DeclarationMatcher YHasClassX =
    recordDecl(hasName("Y"), has(recordDecl(hasName("X"))));
  EXPECT_TRUE(matches("class Y { class X {}; };", YHasClassX));
  EXPECT_TRUE(notMatches("class X {};", YHasClassX));
  EXPECT_TRUE(
    notMatches("class Y { class Z { class X {}; }; };", YHasClassX));
}

TEST(DeclarationMatcher, MatchHasRecursiveAllOf) {
  DeclarationMatcher Recursive =
    recordDecl(
      has(recordDecl(
        has(recordDecl(hasName("X"))),
        has(recordDecl(hasName("Y"))),
        hasName("Z"))),
      has(recordDecl(
        has(recordDecl(hasName("A"))),
        has(recordDecl(hasName("B"))),
        hasName("C"))),
      hasName("F"));

  EXPECT_TRUE(matches(
    "class F {"
      "  class Z {"
      "    class X {};"
      "    class Y {};"
      "  };"
      "  class C {"
      "    class A {};"
      "    class B {};"
      "  };"
      "};", Recursive));

  EXPECT_TRUE(matches(
    "class F {"
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
      "};", Recursive));

  EXPECT_TRUE(matches(
    "class O1 {"
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
      "};", Recursive));
}

TEST(DeclarationMatcher, MatchHasRecursiveAnyOf) {
  DeclarationMatcher Recursive =
    recordDecl(
      anyOf(
        has(recordDecl(
          anyOf(
            has(recordDecl(
              hasName("X"))),
            has(recordDecl(
              hasName("Y"))),
            hasName("Z")))),
        has(recordDecl(
          anyOf(
            hasName("C"),
            has(recordDecl(
              hasName("A"))),
            has(recordDecl(
              hasName("B")))))),
        hasName("F")));

  EXPECT_TRUE(matches("class F {};", Recursive));
  EXPECT_TRUE(matches("class Z {};", Recursive));
  EXPECT_TRUE(matches("class C {};", Recursive));
  EXPECT_TRUE(matches("class M { class N { class X {}; }; };", Recursive));
  EXPECT_TRUE(matches("class M { class N { class B {}; }; };", Recursive));
  EXPECT_TRUE(
    matches("class O1 { class O2 {"
              "  class M { class N { class B {}; }; }; "
              "}; };", Recursive));
}

TEST(DeclarationMatcher, MatchNot) {
  DeclarationMatcher NotClassX =
    cxxRecordDecl(
      isDerivedFrom("Y"),
      unless(hasName("X")));
  EXPECT_TRUE(notMatches("", NotClassX));
  EXPECT_TRUE(notMatches("class Y {};", NotClassX));
  EXPECT_TRUE(matches("class Y {}; class Z : public Y {};", NotClassX));
  EXPECT_TRUE(notMatches("class Y {}; class X : public Y {};", NotClassX));
  EXPECT_TRUE(
    notMatches("class Y {}; class Z {}; class X : public Y {};",
               NotClassX));

  DeclarationMatcher ClassXHasNotClassY =
    recordDecl(
      hasName("X"),
      has(recordDecl(hasName("Z"))),
      unless(
        has(recordDecl(hasName("Y")))));
  EXPECT_TRUE(matches("class X { class Z {}; };", ClassXHasNotClassY));
  EXPECT_TRUE(notMatches("class X { class Y {}; class Z {}; };",
                         ClassXHasNotClassY));

  DeclarationMatcher NamedNotRecord =
    namedDecl(hasName("Foo"), unless(recordDecl()));
  EXPECT_TRUE(matches("void Foo(){}", NamedNotRecord));
  EXPECT_TRUE(notMatches("struct Foo {};", NamedNotRecord));
}

TEST(CastExpression, HasCastKind) {
  EXPECT_TRUE(matches("char *p = 0;",
              castExpr(hasCastKind(CK_NullToPointer))));
  EXPECT_TRUE(notMatches("char *p = 0;",
              castExpr(hasCastKind(CK_DerivedToBase))));
  EXPECT_TRUE(matches("char *p = 0;",
              implicitCastExpr(hasCastKind(CK_NullToPointer))));
}

TEST(DeclarationMatcher, HasDescendant) {
  DeclarationMatcher ZDescendantClassX =
    recordDecl(
      hasDescendant(recordDecl(hasName("X"))),
      hasName("Z"));
  EXPECT_TRUE(matches("class Z { class X {}; };", ZDescendantClassX));
  EXPECT_TRUE(
    matches("class Z { class Y { class X {}; }; };", ZDescendantClassX));
  EXPECT_TRUE(
    matches("class Z { class A { class Y { class X {}; }; }; };",
            ZDescendantClassX));
  EXPECT_TRUE(
    matches("class Z { class A { class B { class Y { class X {}; }; }; }; };",
            ZDescendantClassX));
  EXPECT_TRUE(notMatches("class Z {};", ZDescendantClassX));

  DeclarationMatcher ZDescendantClassXHasClassY =
    recordDecl(
      hasDescendant(recordDecl(has(recordDecl(hasName("Y"))),
                               hasName("X"))),
      hasName("Z"));
  EXPECT_TRUE(matches("class Z { class X { class Y {}; }; };",
                      ZDescendantClassXHasClassY));
  EXPECT_TRUE(
    matches("class Z { class A { class B { class X { class Y {}; }; }; }; };",
            ZDescendantClassXHasClassY));
  EXPECT_TRUE(notMatches(
    "class Z {"
      "  class A {"
      "    class B {"
      "      class X {"
      "        class C {"
      "          class Y {};"
      "        };"
      "      };"
      "    }; "
      "  };"
      "};", ZDescendantClassXHasClassY));

  DeclarationMatcher ZDescendantClassXDescendantClassY =
    recordDecl(
      hasDescendant(recordDecl(hasDescendant(recordDecl(hasName("Y"))),
                               hasName("X"))),
      hasName("Z"));
  EXPECT_TRUE(
    matches("class Z { class A { class X { class B { class Y {}; }; }; }; };",
            ZDescendantClassXDescendantClassY));
  EXPECT_TRUE(matches(
    "class Z {"
      "  class A {"
      "    class X {"
      "      class B {"
      "        class Y {};"
      "      };"
      "      class Y {};"
      "    };"
      "  };"
      "};", ZDescendantClassXDescendantClassY));
}

TEST(DeclarationMatcher, HasDescendantMemoization) {
  DeclarationMatcher CannotMemoize =
    decl(hasDescendant(typeLoc().bind("x")), has(decl()));
  EXPECT_TRUE(matches("void f() { int i; }", CannotMemoize));
}

TEST(DeclarationMatcher, HasDescendantMemoizationUsesRestrictKind) {
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

TEST(DeclarationMatcher, HasAncestorMemoization) {
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

TEST(DeclarationMatcher, HasAttr) {
  EXPECT_TRUE(matches("struct __attribute__((warn_unused)) X {};",
                      decl(hasAttr(clang::attr::WarnUnused))));
  EXPECT_FALSE(matches("struct X {};",
                       decl(hasAttr(clang::attr::WarnUnused))));
}


TEST(DeclarationMatcher, MatchAnyOf) {
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

  DeclarationMatcher XOrYOrZOrUOrV =
    recordDecl(anyOf(hasName("X"), hasName("Y"), hasName("Z"), hasName("U"),
                     hasName("V")));
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

TEST(DeclarationMatcher, ClassIsDerived) {
  DeclarationMatcher IsDerivedFromX = cxxRecordDecl(isDerivedFrom("X"));

  EXPECT_TRUE(matches("class X {}; class Y : public X {};", IsDerivedFromX));
  EXPECT_TRUE(notMatches("class X {};", IsDerivedFromX));
  EXPECT_TRUE(notMatches("class X;", IsDerivedFromX));
  EXPECT_TRUE(notMatches("class Y;", IsDerivedFromX));
  EXPECT_TRUE(notMatches("", IsDerivedFromX));

  DeclarationMatcher IsAX = cxxRecordDecl(isSameOrDerivedFrom("X"));

  EXPECT_TRUE(matches("class X {}; class Y : public X {};", IsAX));
  EXPECT_TRUE(matches("class X {};", IsAX));
  EXPECT_TRUE(matches("class X;", IsAX));
  EXPECT_TRUE(notMatches("class Y;", IsAX));
  EXPECT_TRUE(notMatches("", IsAX));

  DeclarationMatcher ZIsDerivedFromX =
    cxxRecordDecl(hasName("Z"), isDerivedFrom("X"));
  EXPECT_TRUE(
    matches("class X {}; class Y : public X {}; class Z : public Y {};",
            ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("class X {};"
              "template<class T> class Y : public X {};"
              "class Z : public Y<int> {};", ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; template<class T> class Z : public X {};",
                      ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("template<class T> class X {}; "
              "template<class T> class Z : public X<T> {};",
            ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("template<class T, class U=T> class X {}; "
              "template<class T> class Z : public X<T> {};",
            ZIsDerivedFromX));
  EXPECT_TRUE(
    notMatches("template<class X> class A { class Z : public X {}; };",
               ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("template<class X> class A { public: class Z : public X {}; }; "
              "class X{}; void y() { A<X>::Z z; }", ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("template <class T> class X {}; "
              "template<class Y> class A { class Z : public X<Y> {}; };",
            ZIsDerivedFromX));
  EXPECT_TRUE(
    notMatches("template<template<class T> class X> class A { "
                 "  class Z : public X<int> {}; };", ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("template<template<class T> class X> class A { "
              "  public: class Z : public X<int> {}; }; "
              "template<class T> class X {}; void y() { A<X>::Z z; }",
            ZIsDerivedFromX));
  EXPECT_TRUE(
    notMatches("template<class X> class A { class Z : public X::D {}; };",
               ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("template<class X> class A { public: "
              "  class Z : public X::D {}; }; "
              "class Y { public: class X {}; typedef X D; }; "
              "void y() { A<Y>::Z z; }", ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("class X {}; typedef X Y; class Z : public Y {};",
            ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("template<class T> class Y { typedef typename T::U X; "
              "  class Z : public X {}; };", ZIsDerivedFromX));
  EXPECT_TRUE(matches("class X {}; class Z : public ::X {};",
                      ZIsDerivedFromX));
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
  EXPECT_TRUE(
    matches("class X {}; class Y : public X {}; "
              "typedef Y V; typedef V W; class Z : public W {};",
            ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("template<class T, class U> class X {}; "
              "template<class T> class A { class Z : public X<T, int> {}; };",
            ZIsDerivedFromX));
  EXPECT_TRUE(
    notMatches("template<class X> class D { typedef X A; typedef A B; "
                 "  typedef B C; class Z : public C {}; };",
               ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("class X {}; typedef X A; typedef A B; "
              "class Z : public B {};", ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("class X {}; typedef X A; typedef A B; typedef B C; "
              "class Z : public C {};", ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("class U {}; typedef U X; typedef X V; "
              "class Z : public V {};", ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("class Base {}; typedef Base X; "
              "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("class Base {}; typedef Base Base2; typedef Base2 X; "
              "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
    notMatches("class Base {}; class Base2 {}; typedef Base2 X; "
                 "class Z : public Base {};", ZIsDerivedFromX));
  EXPECT_TRUE(
    matches("class A {}; typedef A X; typedef A Y; "
              "class Z : public Y {};", ZIsDerivedFromX));
  EXPECT_TRUE(
    notMatches("template <typename T> class Z;"
                 "template <> class Z<void> {};"
                 "template <typename T> class Z : public Z<void> {};",
               IsDerivedFromX));
  EXPECT_TRUE(
    matches("template <typename T> class X;"
              "template <> class X<void> {};"
              "template <typename T> class X : public X<void> {};",
            IsDerivedFromX));
  EXPECT_TRUE(matches(
    "class X {};"
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
  EXPECT_TRUE(matches(
    RecursiveTemplateOneParameter,
    varDecl(hasName("z_char"),
            hasInitializer(hasType(cxxRecordDecl(isDerivedFrom("Base1"),
                                                 isDerivedFrom("Base2")))))));

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
  EXPECT_TRUE(matches(
    RecursiveTemplateTwoParameters,
    varDecl(hasName("z_char"),
            hasInitializer(hasType(cxxRecordDecl(isDerivedFrom("Base1"),
                                                 isDerivedFrom("Base2")))))));
  EXPECT_TRUE(matches(
    "namespace ns { class X {}; class Y : public X {}; }",
    cxxRecordDecl(isDerivedFrom("::ns::X"))));
  EXPECT_TRUE(notMatches(
    "class X {}; class Y : public X {};",
    cxxRecordDecl(isDerivedFrom("::ns::X"))));

  EXPECT_TRUE(matches(
    "class X {}; class Y : public X {};",
    cxxRecordDecl(isDerivedFrom(recordDecl(hasName("X")).bind("test")))));

  EXPECT_TRUE(matches(
    "template<typename T> class X {};"
      "template<typename T> using Z = X<T>;"
      "template <typename T> class Y : Z<T> {};",
    cxxRecordDecl(isDerivedFrom(namedDecl(hasName("X"))))));
}

TEST(DeclarationMatcher, IsLambda) {
  const auto IsLambda = cxxMethodDecl(ofClass(cxxRecordDecl(isLambda())));
  EXPECT_TRUE(matches("auto x = []{};", IsLambda));
  EXPECT_TRUE(notMatches("struct S { void operator()() const; };", IsLambda));
}

TEST(Matcher, BindMatchedNodes) {
  DeclarationMatcher ClassX = has(recordDecl(hasName("::X")).bind("x"));

  EXPECT_TRUE(matchAndVerifyResultTrue("class X {};",
                                       ClassX, llvm::make_unique<VerifyIdIsBoundTo<CXXRecordDecl>>("x")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class X {};",
                                        ClassX, llvm::make_unique<VerifyIdIsBoundTo<CXXRecordDecl>>("other-id")));

  TypeMatcher TypeAHasClassB = hasDeclaration(
    recordDecl(hasName("A"), has(recordDecl(hasName("B")).bind("b"))));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { public: A *a; class B {}; };",
                                       TypeAHasClassB,
                                       llvm::make_unique<VerifyIdIsBoundTo<Decl>>("b")));

  StatementMatcher MethodX =
    callExpr(callee(cxxMethodDecl(hasName("x")))).bind("x");

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { void x() { x(); } };",
                                       MethodX,
                                       llvm::make_unique<VerifyIdIsBoundTo<CXXMemberCallExpr>>("x")));
}

TEST(Matcher, BindTheSameNameInAlternatives) {
  StatementMatcher matcher = anyOf(
    binaryOperator(hasOperatorName("+"),
                   hasLHS(expr().bind("x")),
                   hasRHS(integerLiteral(equals(0)))),
    binaryOperator(hasOperatorName("+"),
                   hasLHS(integerLiteral(equals(0))),
                   hasRHS(expr().bind("x"))));

  EXPECT_TRUE(matchAndVerifyResultTrue(
    // The first branch of the matcher binds x to 0 but then fails.
    // The second branch binds x to f() and succeeds.
    "int f() { return 0 + f(); }",
    matcher,
    llvm::make_unique<VerifyIdIsBoundTo<CallExpr>>("x")));
}

TEST(Matcher, BindsIDForMemoizedResults) {
  // Using the same matcher in two match expressions will make memoization
  // kick in.
  DeclarationMatcher ClassX = recordDecl(hasName("X")).bind("x");
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "class A { class B { class X {}; }; };",
    DeclarationMatcher(anyOf(
      recordDecl(hasName("A"), hasDescendant(ClassX)),
      recordDecl(hasName("B"), hasDescendant(ClassX)))),
    llvm::make_unique<VerifyIdIsBoundTo<Decl>>("x", 2)));
}

TEST(HasType, MatchesAsString) {
  EXPECT_TRUE(
    matches("class Y { public: void x(); }; void z() {Y* y; y->x(); }",
            cxxMemberCallExpr(on(hasType(asString("class Y *"))))));
  EXPECT_TRUE(
    matches("class X { void x(int x) {} };",
            cxxMethodDecl(hasParameter(0, hasType(asString("int"))))));
  EXPECT_TRUE(matches("namespace ns { struct A {}; }  struct B { ns::A a; };",
                      fieldDecl(hasType(asString("ns::A")))));
  EXPECT_TRUE(matches("namespace { struct A {}; }  struct B { A a; };",
                      fieldDecl(hasType(asString("struct (anonymous namespace)::A")))));
}

TEST(Matcher, HasOperatorNameForOverloadedOperatorCall) {
  StatementMatcher OpCallAndAnd =
    cxxOperatorCallExpr(hasOverloadedOperatorName("&&"));
  EXPECT_TRUE(matches("class Y { }; "
                        "bool operator&&(Y x, Y y) { return true; }; "
                        "Y a; Y b; bool c = a && b;", OpCallAndAnd));
  StatementMatcher OpCallLessLess =
    cxxOperatorCallExpr(hasOverloadedOperatorName("<<"));
  EXPECT_TRUE(notMatches("class Y { }; "
                           "bool operator&&(Y x, Y y) { return true; }; "
                           "Y a; Y b; bool c = a && b;",
                         OpCallLessLess));
  StatementMatcher OpStarCall =
    cxxOperatorCallExpr(hasOverloadedOperatorName("*"));
  EXPECT_TRUE(matches("class Y; int operator*(Y &); void f(Y &y) { *y; }",
                      OpStarCall));
  DeclarationMatcher ClassWithOpStar =
    cxxRecordDecl(hasMethod(hasOverloadedOperatorName("*")));
  EXPECT_TRUE(matches("class Y { int operator*(); };",
                      ClassWithOpStar));
  EXPECT_TRUE(notMatches("class Y { void myOperator(); };",
                         ClassWithOpStar)) ;
  DeclarationMatcher AnyOpStar = functionDecl(hasOverloadedOperatorName("*"));
  EXPECT_TRUE(matches("class Y; int operator*(Y &);", AnyOpStar));
  EXPECT_TRUE(matches("class Y { int operator*(); };", AnyOpStar));
}


TEST(Matcher, NestedOverloadedOperatorCalls) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "class Y { }; "
      "Y& operator&&(Y& x, Y& y) { return x; }; "
      "Y a; Y b; Y c; Y d = a && b && c;",
    cxxOperatorCallExpr(hasOverloadedOperatorName("&&")).bind("x"),
    llvm::make_unique<VerifyIdIsBoundTo<CXXOperatorCallExpr>>("x", 2)));
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

TEST(Matcher, VarDecl_Storage) {
  auto M = varDecl(hasName("X"), hasLocalStorage());
  EXPECT_TRUE(matches("void f() { int X; }", M));
  EXPECT_TRUE(notMatches("int X;", M));
  EXPECT_TRUE(notMatches("void f() { static int X; }", M));

  M = varDecl(hasName("X"), hasGlobalStorage());
  EXPECT_TRUE(notMatches("void f() { int X; }", M));
  EXPECT_TRUE(matches("int X;", M));
  EXPECT_TRUE(matches("void f() { static int X; }", M));
}

TEST(Matcher, VarDecl_IsStaticLocal) {
  auto M = varDecl(isStaticLocal());
  EXPECT_TRUE(matches("void f() { static int X; }", M));
  EXPECT_TRUE(notMatches("static int X;", M));
  EXPECT_TRUE(notMatches("void f() { int X; }", M));
  EXPECT_TRUE(notMatches("int X;", M));
}

TEST(Matcher, VarDecl_StorageDuration) {
  std::string T =
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

  // FIXME: It is really hard to test with thread_local itself because not all
  // targets support TLS, which causes this to be an error depending on what
  // platform the test is being run on. We do not have access to the TargetInfo
  // object to be able to test whether the platform supports TLS or not.
  EXPECT_TRUE(notMatches(T, varDecl(hasName("x"), hasThreadStorageDuration())));
  EXPECT_TRUE(notMatches(T, varDecl(hasName("y"), hasThreadStorageDuration())));
  EXPECT_TRUE(notMatches(T, varDecl(hasName("a"), hasThreadStorageDuration())));
}

TEST(Matcher, FindsVarDeclInFunctionParameter) {
  EXPECT_TRUE(matches(
    "void f(int i) {}",
    varDecl(hasName("i"))));
}

TEST(UnaryExpressionOrTypeTraitExpression, MatchesCorrectType) {
  EXPECT_TRUE(matches("void x() { int a = sizeof(a); }", sizeOfExpr(
    hasArgumentOfType(asString("int")))));
  EXPECT_TRUE(notMatches("void x() { int a = sizeof(a); }", sizeOfExpr(
    hasArgumentOfType(asString("float")))));
  EXPECT_TRUE(matches(
    "struct A {}; void x() { A a; int b = sizeof(a); }",
    sizeOfExpr(hasArgumentOfType(hasDeclaration(recordDecl(hasName("A")))))));
  EXPECT_TRUE(notMatches("void x() { int a = sizeof(a); }", sizeOfExpr(
    hasArgumentOfType(hasDeclaration(recordDecl(hasName("string")))))));
}

TEST(IsInteger, MatchesIntegers) {
  EXPECT_TRUE(matches("int i = 0;", varDecl(hasType(isInteger()))));
  EXPECT_TRUE(matches(
    "long long i = 0; void f(long long) { }; void g() {f(i);}",
    callExpr(hasArgument(0, declRefExpr(
      to(varDecl(hasType(isInteger()))))))));
}

TEST(IsInteger, ReportsNoFalsePositives) {
  EXPECT_TRUE(notMatches("int *i;", varDecl(hasType(isInteger()))));
  EXPECT_TRUE(notMatches("struct T {}; T t; void f(T *) { }; void g() {f(&t);}",
                         callExpr(hasArgument(0, declRefExpr(
                           to(varDecl(hasType(isInteger()))))))));
}

TEST(IsSignedInteger, MatchesSignedIntegers) {
  EXPECT_TRUE(matches("int i = 0;", varDecl(hasType(isSignedInteger()))));
  EXPECT_TRUE(notMatches("unsigned i = 0;",
                         varDecl(hasType(isSignedInteger()))));
}

TEST(IsUnsignedInteger, MatchesUnsignedIntegers) {
  EXPECT_TRUE(notMatches("int i = 0;", varDecl(hasType(isUnsignedInteger()))));
  EXPECT_TRUE(matches("unsigned i = 0;",
                      varDecl(hasType(isUnsignedInteger()))));
}

TEST(IsAnyPointer, MatchesPointers) {
  EXPECT_TRUE(matches("int* i = nullptr;", varDecl(hasType(isAnyPointer()))));
}

TEST(IsAnyPointer, MatchesObjcPointer) {
  EXPECT_TRUE(matchesObjC("@interface Foo @end Foo *f;",
                          varDecl(hasType(isAnyPointer()))));
}

TEST(IsAnyPointer, ReportsNoFalsePositives) {
  EXPECT_TRUE(notMatches("int i = 0;", varDecl(hasType(isAnyPointer()))));
}

TEST(IsAnyCharacter, MatchesCharacters) {
  EXPECT_TRUE(matches("char i = 0;", varDecl(hasType(isAnyCharacter()))));
}

TEST(IsAnyCharacter, ReportsNoFalsePositives) {
  EXPECT_TRUE(notMatches("int i;", varDecl(hasType(isAnyCharacter()))));
}

TEST(IsArrow, MatchesMemberVariablesViaArrow) {
  EXPECT_TRUE(matches("class Y { void x() { this->y; } int y; };",
                      memberExpr(isArrow())));
  EXPECT_TRUE(matches("class Y { void x() { y; } int y; };",
                      memberExpr(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { (*this).y; } int y; };",
                         memberExpr(isArrow())));
  EXPECT_TRUE(matches("template <class T> class Y { void x() { this->m; } };",
                      cxxDependentScopeMemberExpr(isArrow())));
  EXPECT_TRUE(
      notMatches("template <class T> class Y { void x() { (*this).m; } };",
                 cxxDependentScopeMemberExpr(isArrow())));
}

TEST(IsArrow, MatchesStaticMemberVariablesViaArrow) {
  EXPECT_TRUE(matches("class Y { void x() { this->y; } static int y; };",
                      memberExpr(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { y; } static int y; };",
                         memberExpr(isArrow())));
  EXPECT_TRUE(notMatches("class Y { void x() { (*this).y; } static int y; };",
                         memberExpr(isArrow())));
}

TEST(IsArrow, MatchesMemberCallsViaArrow) {
  EXPECT_TRUE(matches("class Y { void x() { this->x(); } };",
                      memberExpr(isArrow())));
  EXPECT_TRUE(matches("class Y { void x() { x(); } };",
                      memberExpr(isArrow())));
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

TEST(ConversionDeclaration, IsExplicit) {
  EXPECT_TRUE(matches("struct S { explicit operator int(); };",
                      cxxConversionDecl(isExplicit())));
  EXPECT_TRUE(notMatches("struct S { operator int(); };",
                         cxxConversionDecl(isExplicit())));
}

TEST(Matcher, ArgumentCount) {
  StatementMatcher Call1Arg = callExpr(argumentCountIs(1));

  EXPECT_TRUE(matches("void x(int) { x(0); }", Call1Arg));
  EXPECT_TRUE(matches("class X { void x(int) { x(0); } };", Call1Arg));
  EXPECT_TRUE(notMatches("void x(int, int) { x(0, 0); }", Call1Arg));
}

TEST(Matcher, ParameterCount) {
  DeclarationMatcher Function1Arg = functionDecl(parameterCountIs(1));
  EXPECT_TRUE(matches("void f(int i) {}", Function1Arg));
  EXPECT_TRUE(matches("class X { void f(int i) {} };", Function1Arg));
  EXPECT_TRUE(notMatches("void f() {}", Function1Arg));
  EXPECT_TRUE(notMatches("void f(int i, int j, int k) {}", Function1Arg));
  EXPECT_TRUE(matches("void f(int i, ...) {};", Function1Arg));
}

TEST(Matcher, References) {
  DeclarationMatcher ReferenceClassX = varDecl(
    hasType(references(recordDecl(hasName("X")))));
  EXPECT_TRUE(matches("class X {}; void y(X y) { X &x = y; }",
                      ReferenceClassX));
  EXPECT_TRUE(
    matches("class X {}; void y(X y) { const X &x = y; }", ReferenceClassX));
  // The match here is on the implicit copy constructor code for
  // class X, not on code 'X x = y'.
  EXPECT_TRUE(
    matches("class X {}; void y(X y) { X x = y; }", ReferenceClassX));
  EXPECT_TRUE(
    notMatches("class X {}; extern X x;", ReferenceClassX));
  EXPECT_TRUE(
    notMatches("class X {}; void y(X *y) { X *&x = y; }", ReferenceClassX));
}

TEST(QualType, hasLocalQualifiers) {
  EXPECT_TRUE(notMatches("typedef const int const_int; const_int i = 1;",
                         varDecl(hasType(hasLocalQualifiers()))));
  EXPECT_TRUE(matches("int *const j = nullptr;",
                      varDecl(hasType(hasLocalQualifiers()))));
  EXPECT_TRUE(matches("int *volatile k;",
                      varDecl(hasType(hasLocalQualifiers()))));
  EXPECT_TRUE(notMatches("int m;",
                         varDecl(hasType(hasLocalQualifiers()))));
}

TEST(IsExternC, MatchesExternCFunctionDeclarations) {
  EXPECT_TRUE(matches("extern \"C\" void f() {}", functionDecl(isExternC())));
  EXPECT_TRUE(matches("extern \"C\" { void f() {} }",
                      functionDecl(isExternC())));
  EXPECT_TRUE(notMatches("void f() {}", functionDecl(isExternC())));
}

TEST(IsExternC, MatchesExternCVariableDeclarations) {
  EXPECT_TRUE(matches("extern \"C\" int i;", varDecl(isExternC())));
  EXPECT_TRUE(matches("extern \"C\" { int i; }", varDecl(isExternC())));
  EXPECT_TRUE(notMatches("int i;", varDecl(isExternC())));
}

TEST(IsStaticStorageClass, MatchesStaticDeclarations) {
  EXPECT_TRUE(
      matches("static void f() {}", functionDecl(isStaticStorageClass())));
  EXPECT_TRUE(matches("static int i = 1;", varDecl(isStaticStorageClass())));
  EXPECT_TRUE(notMatches("int i = 1;", varDecl(isStaticStorageClass())));
  EXPECT_TRUE(notMatches("extern int i;", varDecl(isStaticStorageClass())));
  EXPECT_TRUE(notMatches("void f() {}", functionDecl(isStaticStorageClass())));
}

TEST(IsDefaulted, MatchesDefaultedFunctionDeclarations) {
  EXPECT_TRUE(notMatches("class A { ~A(); };",
                         functionDecl(hasName("~A"), isDefaulted())));
  EXPECT_TRUE(matches("class B { ~B() = default; };",
                      functionDecl(hasName("~B"), isDefaulted())));
}

TEST(IsDeleted, MatchesDeletedFunctionDeclarations) {
  EXPECT_TRUE(
    notMatches("void Func();", functionDecl(hasName("Func"), isDeleted())));
  EXPECT_TRUE(matches("void Func() = delete;",
                      functionDecl(hasName("Func"), isDeleted())));
}

TEST(IsNoThrow, MatchesNoThrowFunctionDeclarations) {
  EXPECT_TRUE(notMatches("void f();", functionDecl(isNoThrow())));
  EXPECT_TRUE(notMatches("void f() throw(int);", functionDecl(isNoThrow())));
  EXPECT_TRUE(
    notMatches("void f() noexcept(false);", functionDecl(isNoThrow())));
  EXPECT_TRUE(matches("void f() throw();", functionDecl(isNoThrow())));
  EXPECT_TRUE(matches("void f() noexcept;", functionDecl(isNoThrow())));

  EXPECT_TRUE(notMatches("void f();", functionProtoType(isNoThrow())));
  EXPECT_TRUE(notMatches("void f() throw(int);", functionProtoType(isNoThrow())));
  EXPECT_TRUE(
    notMatches("void f() noexcept(false);", functionProtoType(isNoThrow())));
  EXPECT_TRUE(matches("void f() throw();", functionProtoType(isNoThrow())));
  EXPECT_TRUE(matches("void f() noexcept;", functionProtoType(isNoThrow())));
}

TEST(isConstexpr, MatchesConstexprDeclarations) {
  EXPECT_TRUE(matches("constexpr int foo = 42;",
                      varDecl(hasName("foo"), isConstexpr())));
  EXPECT_TRUE(matches("constexpr int bar();",
                      functionDecl(hasName("bar"), isConstexpr())));
  EXPECT_TRUE(matchesConditionally("void baz() { if constexpr(1 > 0) {} }",
                                   ifStmt(isConstexpr()), true, "-std=c++17"));
  EXPECT_TRUE(matchesConditionally("void baz() { if (1 > 0) {} }",
                                   ifStmt(isConstexpr()), false, "-std=c++17"));
}

TEST(TemplateArgumentCountIs, Matches) {
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

TEST(IsIntegral, Matches) {
  EXPECT_TRUE(matches("template<int T> struct C {}; C<42> c;",
                      classTemplateSpecializationDecl(
                        hasAnyTemplateArgument(isIntegral()))));
  EXPECT_TRUE(notMatches("template<typename T> struct C {}; C<int> c;",
                         classTemplateSpecializationDecl(hasAnyTemplateArgument(
                           templateArgument(isIntegral())))));
}

TEST(EqualsIntegralValue, Matches) {
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

TEST(Matcher, MatchesAccessSpecDecls) {
  EXPECT_TRUE(matches("class C { public: int i; };", accessSpecDecl()));
  EXPECT_TRUE(
      matches("class C { public: int i; };", accessSpecDecl(isPublic())));
  EXPECT_TRUE(
      notMatches("class C { public: int i; };", accessSpecDecl(isProtected())));
  EXPECT_TRUE(
      notMatches("class C { public: int i; };", accessSpecDecl(isPrivate())));

  EXPECT_TRUE(notMatches("class C { int i; };", accessSpecDecl()));
}

TEST(Matcher, MatchesFinal) {
  EXPECT_TRUE(matches("class X final {};", cxxRecordDecl(isFinal())));
  EXPECT_TRUE(matches("class X { virtual void f() final; };",
                      cxxMethodDecl(isFinal())));
  EXPECT_TRUE(notMatches("class X {};", cxxRecordDecl(isFinal())));
  EXPECT_TRUE(
    notMatches("class X { virtual void f(); };", cxxMethodDecl(isFinal())));
}

TEST(Matcher, MatchesVirtualMethod) {
  EXPECT_TRUE(matches("class X { virtual int f(); };",
                      cxxMethodDecl(isVirtual(), hasName("::X::f"))));
  EXPECT_TRUE(notMatches("class X { int f(); };", cxxMethodDecl(isVirtual())));
}

TEST(Matcher, MatchesVirtualAsWrittenMethod) {
  EXPECT_TRUE(matches("class A { virtual int f(); };"
                        "class B : public A { int f(); };",
                      cxxMethodDecl(isVirtualAsWritten(), hasName("::A::f"))));
  EXPECT_TRUE(
    notMatches("class A { virtual int f(); };"
                 "class B : public A { int f(); };",
               cxxMethodDecl(isVirtualAsWritten(), hasName("::B::f"))));
}

TEST(Matcher, MatchesPureMethod) {
  EXPECT_TRUE(matches("class X { virtual int f() = 0; };",
                      cxxMethodDecl(isPure(), hasName("::X::f"))));
  EXPECT_TRUE(notMatches("class X { int f(); };", cxxMethodDecl(isPure())));
}

TEST(Matcher, MatchesCopyAssignmentOperator) {
  EXPECT_TRUE(matches("class X { X &operator=(X); };",
                      cxxMethodDecl(isCopyAssignmentOperator())));
  EXPECT_TRUE(matches("class X { X &operator=(X &); };",
                      cxxMethodDecl(isCopyAssignmentOperator())));
  EXPECT_TRUE(matches("class X { X &operator=(const X &); };",
                      cxxMethodDecl(isCopyAssignmentOperator())));
  EXPECT_TRUE(matches("class X { X &operator=(volatile X &); };",
                      cxxMethodDecl(isCopyAssignmentOperator())));
  EXPECT_TRUE(matches("class X { X &operator=(const volatile X &); };",
                      cxxMethodDecl(isCopyAssignmentOperator())));
  EXPECT_TRUE(notMatches("class X { X &operator=(X &&); };",
                         cxxMethodDecl(isCopyAssignmentOperator())));
}

TEST(Matcher, MatchesMoveAssignmentOperator) {
  EXPECT_TRUE(notMatches("class X { X &operator=(X); };",
                         cxxMethodDecl(isMoveAssignmentOperator())));
  EXPECT_TRUE(matches("class X { X &operator=(X &&); };",
                      cxxMethodDecl(isMoveAssignmentOperator())));
  EXPECT_TRUE(matches("class X { X &operator=(const X &&); };",
                      cxxMethodDecl(isMoveAssignmentOperator())));
  EXPECT_TRUE(matches("class X { X &operator=(volatile X &&); };",
                      cxxMethodDecl(isMoveAssignmentOperator())));
  EXPECT_TRUE(matches("class X { X &operator=(const volatile X &&); };",
                      cxxMethodDecl(isMoveAssignmentOperator())));
  EXPECT_TRUE(notMatches("class X { X &operator=(X &); };",
                         cxxMethodDecl(isMoveAssignmentOperator())));
}

TEST(Matcher, MatchesConstMethod) {
  EXPECT_TRUE(
    matches("struct A { void foo() const; };", cxxMethodDecl(isConst())));
  EXPECT_TRUE(
    notMatches("struct A { void foo(); };", cxxMethodDecl(isConst())));
}

TEST(Matcher, MatchesOverridingMethod) {
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

TEST(Matcher, ConstructorArgument) {
  StatementMatcher Constructor = cxxConstructExpr(
    hasArgument(0, declRefExpr(to(varDecl(hasName("y"))))));

  EXPECT_TRUE(
    matches("class X { public: X(int); }; void x() { int y; X x(y); }",
            Constructor));
  EXPECT_TRUE(
    matches("class X { public: X(int); }; void x() { int y; X x = X(y); }",
            Constructor));
  EXPECT_TRUE(
    matches("class X { public: X(int); }; void x() { int y; X x = y; }",
            Constructor));
  EXPECT_TRUE(
    notMatches("class X { public: X(int); }; void x() { int z; X x(z); }",
               Constructor));

  StatementMatcher WrongIndex = cxxConstructExpr(
    hasArgument(42, declRefExpr(to(varDecl(hasName("y"))))));
  EXPECT_TRUE(
    notMatches("class X { public: X(int); }; void x() { int y; X x(y); }",
               WrongIndex));
}

TEST(Matcher, ConstructorArgumentCount) {
  StatementMatcher Constructor1Arg = cxxConstructExpr(argumentCountIs(1));

  EXPECT_TRUE(
    matches("class X { public: X(int); }; void x() { X x(0); }",
            Constructor1Arg));
  EXPECT_TRUE(
    matches("class X { public: X(int); }; void x() { X x = X(0); }",
            Constructor1Arg));
  EXPECT_TRUE(
    matches("class X { public: X(int); }; void x() { X x = 0; }",
            Constructor1Arg));
  EXPECT_TRUE(
    notMatches("class X { public: X(int, int); }; void x() { X x(0, 0); }",
               Constructor1Arg));
}

TEST(Matcher, ConstructorListInitialization) {
  StatementMatcher ConstructorListInit =
    cxxConstructExpr(isListInitialization());

  EXPECT_TRUE(
    matches("class X { public: X(int); }; void x() { X x{0}; }",
            ConstructorListInit));
  EXPECT_FALSE(
    matches("class X { public: X(int); }; void x() { X x(0); }",
            ConstructorListInit));
}

TEST(ConstructorDeclaration, IsImplicit) {
  // This one doesn't match because the constructor is not added by the
  // compiler (it is not needed).
  EXPECT_TRUE(notMatches("class Foo { };",
                         cxxConstructorDecl(isImplicit())));
  // The compiler added the implicit default constructor.
  EXPECT_TRUE(matches("class Foo { }; Foo* f = new Foo();",
                      cxxConstructorDecl(isImplicit())));
  EXPECT_TRUE(matches("class Foo { Foo(){} };",
                      cxxConstructorDecl(unless(isImplicit()))));
  // The compiler added an implicit assignment operator.
  EXPECT_TRUE(matches("struct A { int x; } a = {0}, b = a; void f() { a = b; }",
                      cxxMethodDecl(isImplicit(), hasName("operator="))));
}

TEST(ConstructorDeclaration, IsExplicit) {
  EXPECT_TRUE(matches("struct S { explicit S(int); };",
                      cxxConstructorDecl(isExplicit())));
  EXPECT_TRUE(notMatches("struct S { S(int); };",
                         cxxConstructorDecl(isExplicit())));
}

TEST(ConstructorDeclaration, Kinds) {
  EXPECT_TRUE(matches(
      "struct S { S(); };",
      cxxConstructorDecl(isDefaultConstructor(), unless(isImplicit()))));
  EXPECT_TRUE(notMatches(
      "struct S { S(); };",
      cxxConstructorDecl(isCopyConstructor(), unless(isImplicit()))));
  EXPECT_TRUE(notMatches(
      "struct S { S(); };",
      cxxConstructorDecl(isMoveConstructor(), unless(isImplicit()))));

  EXPECT_TRUE(notMatches(
      "struct S { S(const S&); };",
      cxxConstructorDecl(isDefaultConstructor(), unless(isImplicit()))));
  EXPECT_TRUE(matches(
      "struct S { S(const S&); };",
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
  EXPECT_TRUE(matches(
      "struct S { S(S&&); };",
      cxxConstructorDecl(isMoveConstructor(), unless(isImplicit()))));
}

TEST(ConstructorDeclaration, IsUserProvided) {
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

TEST(ConstructorDeclaration, IsDelegatingConstructor) {
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

TEST(StringLiteral, HasSize) {
  StatementMatcher Literal = stringLiteral(hasSize(4));
  EXPECT_TRUE(matches("const char *s = \"abcd\";", Literal));
  // wide string
  EXPECT_TRUE(matches("const wchar_t *s = L\"abcd\";", Literal));
  // with escaped characters
  EXPECT_TRUE(matches("const char *s = \"\x05\x06\x07\x08\";", Literal));
  // no matching, too small
  EXPECT_TRUE(notMatches("const char *s = \"ab\";", Literal));
}

TEST(Matcher, HasNameSupportsNamespaces) {
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

TEST(Matcher, HasNameSupportsOuterClasses) {
  EXPECT_TRUE(
    matches("class A { class B { class C; }; };",
            recordDecl(hasName("A::B::C"))));
  EXPECT_TRUE(
    matches("class A { class B { class C; }; };",
            recordDecl(hasName("::A::B::C"))));
  EXPECT_TRUE(
    matches("class A { class B { class C; }; };",
            recordDecl(hasName("B::C"))));
  EXPECT_TRUE(
    matches("class A { class B { class C; }; };",
            recordDecl(hasName("C"))));
  EXPECT_TRUE(
    notMatches("class A { class B { class C; }; };",
               recordDecl(hasName("c::B::C"))));
  EXPECT_TRUE(
    notMatches("class A { class B { class C; }; };",
               recordDecl(hasName("A::c::C"))));
  EXPECT_TRUE(
    notMatches("class A { class B { class C; }; };",
               recordDecl(hasName("A::B::A"))));
  EXPECT_TRUE(
    notMatches("class A { class B { class C; }; };",
               recordDecl(hasName("::C"))));
  EXPECT_TRUE(
    notMatches("class A { class B { class C; }; };",
               recordDecl(hasName("::B::C"))));
  EXPECT_TRUE(notMatches("class A { class B { class C; }; };",
                         recordDecl(hasName("z::A::B::C"))));
  EXPECT_TRUE(
    notMatches("class A { class B { class C; }; };",
               recordDecl(hasName("A+B::C"))));
}

TEST(Matcher, HasNameSupportsInlinedNamespaces) {
  std::string code = "namespace a { inline namespace b { class C; } }";
  EXPECT_TRUE(matches(code, recordDecl(hasName("a::b::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("a::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("::a::b::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("::a::C"))));
}

TEST(Matcher, HasNameSupportsAnonymousNamespaces) {
  std::string code = "namespace a { namespace { class C; } }";
  EXPECT_TRUE(
    matches(code, recordDecl(hasName("a::(anonymous namespace)::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("a::C"))));
  EXPECT_TRUE(
    matches(code, recordDecl(hasName("::a::(anonymous namespace)::C"))));
  EXPECT_TRUE(matches(code, recordDecl(hasName("::a::C"))));
}

TEST(Matcher, HasNameSupportsAnonymousOuterClasses) {
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

TEST(Matcher, HasNameSupportsFunctionScope) {
  std::string code =
    "namespace a { void F(int a) { struct S { int m; }; int i; } }";
  EXPECT_TRUE(matches(code, varDecl(hasName("i"))));
  EXPECT_FALSE(matches(code, varDecl(hasName("F()::i"))));

  EXPECT_TRUE(matches(code, fieldDecl(hasName("m"))));
  EXPECT_TRUE(matches(code, fieldDecl(hasName("S::m"))));
  EXPECT_TRUE(matches(code, fieldDecl(hasName("F(int)::S::m"))));
  EXPECT_TRUE(matches(code, fieldDecl(hasName("a::F(int)::S::m"))));
  EXPECT_TRUE(matches(code, fieldDecl(hasName("::a::F(int)::S::m"))));
}

TEST(Matcher, HasAnyName) {
  const std::string Code = "namespace a { namespace b { class C; } }";

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

TEST(Matcher, IsDefinition) {
  DeclarationMatcher DefinitionOfClassA =
    recordDecl(hasName("A"), isDefinition());
  EXPECT_TRUE(matches("class A {};", DefinitionOfClassA));
  EXPECT_TRUE(notMatches("class A;", DefinitionOfClassA));

  DeclarationMatcher DefinitionOfVariableA =
    varDecl(hasName("a"), isDefinition());
  EXPECT_TRUE(matches("int a;", DefinitionOfVariableA));
  EXPECT_TRUE(notMatches("extern int a;", DefinitionOfVariableA));

  DeclarationMatcher DefinitionOfMethodA =
    cxxMethodDecl(hasName("a"), isDefinition());
  EXPECT_TRUE(matches("class A { void a() {} };", DefinitionOfMethodA));
  EXPECT_TRUE(notMatches("class A { void a(); };", DefinitionOfMethodA));

  DeclarationMatcher DefinitionOfObjCMethodA =
    objcMethodDecl(hasName("a"), isDefinition());
  EXPECT_TRUE(matchesObjC("@interface A @end "
                          "@implementation A; -(void)a {} @end",
                          DefinitionOfObjCMethodA));
  EXPECT_TRUE(notMatchesObjC("@interface A; - (void)a; @end",
                             DefinitionOfObjCMethodA));
}

TEST(Matcher, HandlesNullQualTypes) {
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
    expr(hasType(TypeMatcher(
      anyOf(
        TypeMatcher(hasDeclaration(anything())),
        pointsTo(AnyType),
        references(AnyType)
        // Other QualType matchers should go here.
      ))))));
}

TEST(ObjCIvarRefExprMatcher, IvarExpr) {
  std::string ObjCString =
    "@interface A @end "
    "@implementation A { A *x; } - (void) func { x = 0; } @end";
  EXPECT_TRUE(matchesObjC(ObjCString, objcIvarRefExpr()));
  EXPECT_TRUE(matchesObjC(ObjCString, objcIvarRefExpr(
        hasDeclaration(namedDecl(hasName("x"))))));
  EXPECT_FALSE(matchesObjC(ObjCString, objcIvarRefExpr(
        hasDeclaration(namedDecl(hasName("y"))))));
}

TEST(BlockExprMatcher, BlockExpr) {
  EXPECT_TRUE(matchesObjC("void f() { ^{}(); }", blockExpr()));
}

TEST(StatementCountIs, FindsNoStatementsInAnEmptyCompoundStatement) {
  EXPECT_TRUE(matches("void f() { }",
                      compoundStmt(statementCountIs(0))));
  EXPECT_TRUE(notMatches("void f() {}",
                         compoundStmt(statementCountIs(1))));
}

TEST(StatementCountIs, AppearsToMatchOnlyOneCount) {
  EXPECT_TRUE(matches("void f() { 1; }",
                      compoundStmt(statementCountIs(1))));
  EXPECT_TRUE(notMatches("void f() { 1; }",
                         compoundStmt(statementCountIs(0))));
  EXPECT_TRUE(notMatches("void f() { 1; }",
                         compoundStmt(statementCountIs(2))));
}

TEST(StatementCountIs, WorksWithMultipleStatements) {
  EXPECT_TRUE(matches("void f() { 1; 2; 3; }",
                      compoundStmt(statementCountIs(3))));
}

TEST(StatementCountIs, WorksWithNestedCompoundStatements) {
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
                      compoundStmt(statementCountIs(1))));
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
                      compoundStmt(statementCountIs(2))));
  EXPECT_TRUE(notMatches("void f() { { 1; } { 1; 2; 3; 4; } }",
                         compoundStmt(statementCountIs(3))));
  EXPECT_TRUE(matches("void f() { { 1; } { 1; 2; 3; 4; } }",
                      compoundStmt(statementCountIs(4))));
}

TEST(Member, WorksInSimplestCase) {
  EXPECT_TRUE(matches("struct { int first; } s; int i(s.first);",
                      memberExpr(member(hasName("first")))));
}

TEST(Member, DoesNotMatchTheBaseExpression) {
  // Don't pick out the wrong part of the member expression, this should
  // be checking the member (name) only.
  EXPECT_TRUE(notMatches("struct { int i; } first; int i(first.i);",
                         memberExpr(member(hasName("first")))));
}

TEST(Member, MatchesInMemberFunctionCall) {
  EXPECT_TRUE(matches("void f() {"
                        "  struct { void first() {}; } s;"
                        "  s.first();"
                        "};",
                      memberExpr(member(hasName("first")))));
}

TEST(Member, MatchesMember) {
  EXPECT_TRUE(matches(
    "struct A { int i; }; void f() { A a; a.i = 2; }",
    memberExpr(hasDeclaration(fieldDecl(hasType(isInteger()))))));
  EXPECT_TRUE(notMatches(
    "struct A { float f; }; void f() { A a; a.f = 2.0f; }",
    memberExpr(hasDeclaration(fieldDecl(hasType(isInteger()))))));
}

TEST(Member, BitFields) {
  EXPECT_TRUE(matches("class C { int a : 2; int b; };",
                      fieldDecl(isBitField(), hasName("a"))));
  EXPECT_TRUE(notMatches("class C { int a : 2; int b; };",
                         fieldDecl(isBitField(), hasName("b"))));
  EXPECT_TRUE(matches("class C { int a : 2; int b : 4; };",
                      fieldDecl(isBitField(), hasBitWidth(2), hasName("a"))));
}

TEST(Member, InClassInitializer) {
  EXPECT_TRUE(
      matches("class C { int a = 2; int b; };",
              fieldDecl(hasInClassInitializer(integerLiteral(equals(2))),
                        hasName("a"))));
  EXPECT_TRUE(
      notMatches("class C { int a = 2; int b; };",
                 fieldDecl(hasInClassInitializer(anything()), hasName("b"))));
}

TEST(Member, UnderstandsAccess) {
  EXPECT_TRUE(matches(
    "struct A { int i; };", fieldDecl(isPublic(), hasName("i"))));
  EXPECT_TRUE(notMatches(
    "struct A { int i; };", fieldDecl(isProtected(), hasName("i"))));
  EXPECT_TRUE(notMatches(
    "struct A { int i; };", fieldDecl(isPrivate(), hasName("i"))));

  EXPECT_TRUE(notMatches(
    "class A { int i; };", fieldDecl(isPublic(), hasName("i"))));
  EXPECT_TRUE(notMatches(
    "class A { int i; };", fieldDecl(isProtected(), hasName("i"))));
  EXPECT_TRUE(matches(
    "class A { int i; };", fieldDecl(isPrivate(), hasName("i"))));

  EXPECT_TRUE(notMatches(
    "class A { protected: int i; };", fieldDecl(isPublic(), hasName("i"))));
  EXPECT_TRUE(matches("class A { protected: int i; };",
                      fieldDecl(isProtected(), hasName("i"))));
  EXPECT_TRUE(notMatches(
    "class A { protected: int i; };", fieldDecl(isPrivate(), hasName("i"))));

  // Non-member decls have the AccessSpecifier AS_none and thus aren't matched.
  EXPECT_TRUE(notMatches("int i;", varDecl(isPublic(), hasName("i"))));
  EXPECT_TRUE(notMatches("int i;", varDecl(isProtected(), hasName("i"))));
  EXPECT_TRUE(notMatches("int i;", varDecl(isPrivate(), hasName("i"))));
}

TEST(hasDynamicExceptionSpec, MatchesDynamicExceptionSpecifications) {
  EXPECT_TRUE(notMatches("void f();", functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void g() noexcept;",
                         functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void h() noexcept(true);",
                         functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void i() noexcept(false);",
                         functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void j() throw();", functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void k() throw(int);", functionDecl(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void l() throw(...);", functionDecl(hasDynamicExceptionSpec())));

  EXPECT_TRUE(notMatches("void f();", functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void g() noexcept;",
                         functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void h() noexcept(true);",
                         functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(notMatches("void i() noexcept(false);",
                         functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void j() throw();", functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void k() throw(int);", functionProtoType(hasDynamicExceptionSpec())));
  EXPECT_TRUE(
      matches("void l() throw(...);", functionProtoType(hasDynamicExceptionSpec())));
}

TEST(HasObjectExpression, DoesNotMatchMember) {
  EXPECT_TRUE(notMatches(
    "class X {}; struct Z { X m; }; void f(Z z) { z.m; }",
    memberExpr(hasObjectExpression(hasType(recordDecl(hasName("X")))))));
}

TEST(HasObjectExpression, MatchesBaseOfVariable) {
  EXPECT_TRUE(matches(
    "struct X { int m; }; void f(X x) { x.m; }",
    memberExpr(hasObjectExpression(hasType(recordDecl(hasName("X")))))));
  EXPECT_TRUE(matches(
    "struct X { int m; }; void f(X* x) { x->m; }",
    memberExpr(hasObjectExpression(
      hasType(pointsTo(recordDecl(hasName("X"))))))));
  EXPECT_TRUE(matches("template <class T> struct X { void f() { T t; t.m; } };",
                      cxxDependentScopeMemberExpr(hasObjectExpression(
                          declRefExpr(to(namedDecl(hasName("t"))))))));
  EXPECT_TRUE(
      matches("template <class T> struct X { void f() { T t; t->m; } };",
              cxxDependentScopeMemberExpr(hasObjectExpression(
                  declRefExpr(to(namedDecl(hasName("t"))))))));
}

TEST(HasObjectExpression, MatchesBaseOfMemberFunc) {
  EXPECT_TRUE(matches(
      "struct X { void f(); }; void g(X x) { x.f(); }",
      memberExpr(hasObjectExpression(hasType(recordDecl(hasName("X")))))));
  EXPECT_TRUE(matches("struct X { template <class T> void f(); };"
                      "template <class T> void g(X x) { x.f<T>(); }",
                      unresolvedMemberExpr(hasObjectExpression(
                          hasType(recordDecl(hasName("X")))))));
  EXPECT_TRUE(matches("template <class T> void f(T t) { t.g(); }",
                      cxxDependentScopeMemberExpr(hasObjectExpression(
                          declRefExpr(to(namedDecl(hasName("t"))))))));
}

TEST(HasObjectExpression,
     MatchesObjectExpressionOfImplicitlyFormedMemberExpression) {
  EXPECT_TRUE(matches(
    "class X {}; struct S { X m; void f() { this->m; } };",
    memberExpr(hasObjectExpression(
      hasType(pointsTo(recordDecl(hasName("S"))))))));
  EXPECT_TRUE(matches(
    "class X {}; struct S { X m; void f() { m; } };",
    memberExpr(hasObjectExpression(
      hasType(pointsTo(recordDecl(hasName("S"))))))));
}

TEST(Field, DoesNotMatchNonFieldMembers) {
  EXPECT_TRUE(notMatches("class X { void m(); };", fieldDecl(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { class m {}; };", fieldDecl(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { enum { m }; };", fieldDecl(hasName("m"))));
  EXPECT_TRUE(notMatches("class X { enum m {}; };", fieldDecl(hasName("m"))));
}

TEST(Field, MatchesField) {
  EXPECT_TRUE(matches("class X { int m; };", fieldDecl(hasName("m"))));
}

TEST(IsVolatileQualified, QualifiersMatch) {
  EXPECT_TRUE(matches("volatile int i = 42;",
                      varDecl(hasType(isVolatileQualified()))));
  EXPECT_TRUE(notMatches("volatile int *i;",
                         varDecl(hasType(isVolatileQualified()))));
  EXPECT_TRUE(matches("typedef volatile int v_int; v_int i = 42;",
                      varDecl(hasType(isVolatileQualified()))));
}

TEST(IsConstQualified, MatchesConstInt) {
  EXPECT_TRUE(matches("const int i = 42;",
                      varDecl(hasType(isConstQualified()))));
}

TEST(IsConstQualified, MatchesConstPointer) {
  EXPECT_TRUE(matches("int i = 42; int* const p(&i);",
                      varDecl(hasType(isConstQualified()))));
}

TEST(IsConstQualified, MatchesThroughTypedef) {
  EXPECT_TRUE(matches("typedef const int const_int; const_int i = 42;",
                      varDecl(hasType(isConstQualified()))));
  EXPECT_TRUE(matches("typedef int* int_ptr; const int_ptr p(0);",
                      varDecl(hasType(isConstQualified()))));
}

TEST(IsConstQualified, DoesNotMatchInappropriately) {
  EXPECT_TRUE(notMatches("typedef int nonconst_int; nonconst_int i = 42;",
                         varDecl(hasType(isConstQualified()))));
  EXPECT_TRUE(notMatches("int const* p;",
                         varDecl(hasType(isConstQualified()))));
}

TEST(DeclCount, DeclCountIsCorrect) {
  EXPECT_TRUE(matches("void f() {int i,j;}",
                      declStmt(declCountIs(2))));
  EXPECT_TRUE(notMatches("void f() {int i,j; int k;}",
                         declStmt(declCountIs(3))));
  EXPECT_TRUE(notMatches("void f() {int i,j, k, l;}",
                         declStmt(declCountIs(3))));
}


TEST(EachOf, TriggersForEachMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "class A { int a; int b; };",
    recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                      has(fieldDecl(hasName("b")).bind("v")))),
    llvm::make_unique<VerifyIdIsBoundTo<FieldDecl>>("v", 2)));
}

TEST(EachOf, BehavesLikeAnyOfUnlessBothMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "class A { int a; int c; };",
    recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                      has(fieldDecl(hasName("b")).bind("v")))),
    llvm::make_unique<VerifyIdIsBoundTo<FieldDecl>>("v", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "class A { int c; int b; };",
    recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                      has(fieldDecl(hasName("b")).bind("v")))),
    llvm::make_unique<VerifyIdIsBoundTo<FieldDecl>>("v", 1)));
  EXPECT_TRUE(notMatches(
    "class A { int c; int d; };",
    recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                      has(fieldDecl(hasName("b")).bind("v"))))));
}

TEST(IsTemplateInstantiation, MatchesImplicitClassTemplateInstantiation) {
  // Make sure that we can both match the class by name (::X) and by the type
  // the template was instantiated with (via a field).

  EXPECT_TRUE(matches(
    "template <typename T> class X {}; class A {}; X<A> x;",
    cxxRecordDecl(hasName("::X"), isTemplateInstantiation())));

  EXPECT_TRUE(matches(
    "template <typename T> class X { T t; }; class A {}; X<A> x;",
    cxxRecordDecl(isTemplateInstantiation(), hasDescendant(
      fieldDecl(hasType(recordDecl(hasName("A"))))))));
}

TEST(IsTemplateInstantiation, MatchesImplicitFunctionTemplateInstantiation) {
  EXPECT_TRUE(matches(
    "template <typename T> void f(T t) {} class A {}; void g() { f(A()); }",
    functionDecl(hasParameter(0, hasType(recordDecl(hasName("A")))),
                 isTemplateInstantiation())));
}

TEST(IsTemplateInstantiation, MatchesExplicitClassTemplateInstantiation) {
  EXPECT_TRUE(matches(
    "template <typename T> class X { T t; }; class A {};"
      "template class X<A>;",
    cxxRecordDecl(isTemplateInstantiation(), hasDescendant(
      fieldDecl(hasType(recordDecl(hasName("A"))))))));

  // Make sure that we match the instantiation instead of the template
  // definition by checking whether the member function is present.
  EXPECT_TRUE(
      matches("template <typename T> class X { void f() { T t; } };"
              "extern template class X<int>;",
              cxxRecordDecl(isTemplateInstantiation(),
                            unless(hasDescendant(varDecl(hasName("t")))))));
}

TEST(IsTemplateInstantiation,
     MatchesInstantiationOfPartiallySpecializedClassTemplate) {
  EXPECT_TRUE(matches(
    "template <typename T> class X {};"
      "template <typename T> class X<T*> {}; class A {}; X<A*> x;",
    cxxRecordDecl(hasName("::X"), isTemplateInstantiation())));
}

TEST(IsTemplateInstantiation,
     MatchesInstantiationOfClassTemplateNestedInNonTemplate) {
  EXPECT_TRUE(matches(
    "class A {};"
      "class X {"
      "  template <typename U> class Y { U u; };"
      "  Y<A> y;"
      "};",
    cxxRecordDecl(hasName("::X::Y"), isTemplateInstantiation())));
}

TEST(IsTemplateInstantiation, DoesNotMatchInstantiationsInsideOfInstantiation) {
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

TEST(IsTemplateInstantiation, DoesNotMatchExplicitClassTemplateSpecialization) {
  EXPECT_TRUE(notMatches(
    "template <typename T> class X {}; class A {};"
      "template <> class X<A> {}; X<A> x;",
    cxxRecordDecl(hasName("::X"), isTemplateInstantiation())));
}

TEST(IsTemplateInstantiation, DoesNotMatchNonTemplate) {
  EXPECT_TRUE(notMatches(
    "class A {}; class Y { A a; };",
    cxxRecordDecl(isTemplateInstantiation())));
}

TEST(IsInstantiated, MatchesInstantiation) {
  EXPECT_TRUE(
    matches("template<typename T> class A { T i; }; class Y { A<int> a; };",
            cxxRecordDecl(isInstantiated())));
}

TEST(IsInstantiated, NotMatchesDefinition) {
  EXPECT_TRUE(notMatches("template<typename T> class A { T i; };",
                         cxxRecordDecl(isInstantiated())));
}

TEST(IsInTemplateInstantiation, MatchesInstantiationStmt) {
  EXPECT_TRUE(matches("template<typename T> struct A { A() { T i; } };"
                        "class Y { A<int> a; }; Y y;",
                      declStmt(isInTemplateInstantiation())));
}

TEST(IsInTemplateInstantiation, NotMatchesDefinitionStmt) {
  EXPECT_TRUE(notMatches("template<typename T> struct A { void x() { T i; } };",
                         declStmt(isInTemplateInstantiation())));
}

TEST(IsInstantiated, MatchesFunctionInstantiation) {
  EXPECT_TRUE(
    matches("template<typename T> void A(T t) { T i; } void x() { A(0); }",
            functionDecl(isInstantiated())));
}

TEST(IsInstantiated, NotMatchesFunctionDefinition) {
  EXPECT_TRUE(notMatches("template<typename T> void A(T t) { T i; }",
                         varDecl(isInstantiated())));
}

TEST(IsInTemplateInstantiation, MatchesFunctionInstantiationStmt) {
  EXPECT_TRUE(
    matches("template<typename T> void A(T t) { T i; } void x() { A(0); }",
            declStmt(isInTemplateInstantiation())));
}

TEST(IsInTemplateInstantiation, NotMatchesFunctionDefinitionStmt) {
  EXPECT_TRUE(notMatches("template<typename T> void A(T t) { T i; }",
                         declStmt(isInTemplateInstantiation())));
}

TEST(IsInTemplateInstantiation, Sharing) {
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

TEST(IsInstantiationDependent, MatchesNonValueTypeDependent) {
  EXPECT_TRUE(matches(
      "template<typename T> void f() { (void) sizeof(sizeof(T() + T())); }",
      expr(isInstantiationDependent())));
}

TEST(IsInstantiationDependent, MatchesValueDependent) {
  EXPECT_TRUE(matches("template<int T> int f() { return T; }",
                      expr(isInstantiationDependent())));
}

TEST(IsInstantiationDependent, MatchesTypeDependent) {
  EXPECT_TRUE(matches("template<typename T> T f() { return T(); }",
                      expr(isInstantiationDependent())));
}

TEST(IsTypeDependent, MatchesTypeDependent) {
  EXPECT_TRUE(matches("template<typename T> T f() { return T(); }",
                      expr(isTypeDependent())));
}

TEST(IsTypeDependent, NotMatchesValueDependent) {
  EXPECT_TRUE(notMatches("template<int T> int f() { return T; }",
                         expr(isTypeDependent())));
}

TEST(IsValueDependent, MatchesValueDependent) {
  EXPECT_TRUE(matches("template<int T> int f() { return T; }",
                      expr(isValueDependent())));
}

TEST(IsValueDependent, MatchesTypeDependent) {
  EXPECT_TRUE(matches("template<typename T> T f() { return T(); }",
                      expr(isValueDependent())));
}

TEST(IsValueDependent, MatchesInstantiationDependent) {
  EXPECT_TRUE(matches(
      "template<typename T> void f() { (void) sizeof(sizeof(T() + T())); }",
      expr(isValueDependent())));
}

TEST(IsExplicitTemplateSpecialization,
     DoesNotMatchPrimaryTemplate) {
  EXPECT_TRUE(notMatches(
    "template <typename T> class X {};",
    cxxRecordDecl(isExplicitTemplateSpecialization())));
  EXPECT_TRUE(notMatches(
    "template <typename T> void f(T t);",
    functionDecl(isExplicitTemplateSpecialization())));
}

TEST(IsExplicitTemplateSpecialization,
     DoesNotMatchExplicitTemplateInstantiations) {
  EXPECT_TRUE(notMatches(
    "template <typename T> class X {};"
      "template class X<int>; extern template class X<long>;",
    cxxRecordDecl(isExplicitTemplateSpecialization())));
  EXPECT_TRUE(notMatches(
    "template <typename T> void f(T t) {}"
      "template void f(int t); extern template void f(long t);",
    functionDecl(isExplicitTemplateSpecialization())));
}

TEST(IsExplicitTemplateSpecialization,
     DoesNotMatchImplicitTemplateInstantiations) {
  EXPECT_TRUE(notMatches(
    "template <typename T> class X {}; X<int> x;",
    cxxRecordDecl(isExplicitTemplateSpecialization())));
  EXPECT_TRUE(notMatches(
    "template <typename T> void f(T t); void g() { f(10); }",
    functionDecl(isExplicitTemplateSpecialization())));
}

TEST(IsExplicitTemplateSpecialization,
     MatchesExplicitTemplateSpecializations) {
  EXPECT_TRUE(matches(
    "template <typename T> class X {};"
      "template<> class X<int> {};",
    cxxRecordDecl(isExplicitTemplateSpecialization())));
  EXPECT_TRUE(matches(
    "template <typename T> void f(T t) {}"
      "template<> void f(int t) {}",
    functionDecl(isExplicitTemplateSpecialization())));
}

TEST(TypeMatching, MatchesNoReturn) {
  EXPECT_TRUE(notMatches("void func();", functionDecl(isNoReturn())));
  EXPECT_TRUE(notMatches("void func() {}", functionDecl(isNoReturn())));

  EXPECT_TRUE(notMatchesC("void func();", functionDecl(isNoReturn())));
  EXPECT_TRUE(notMatchesC("void func() {}", functionDecl(isNoReturn())));

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
  EXPECT_TRUE(matches("struct S { [[noreturn]] S() {} };",
                      functionDecl(isNoReturn())));

  // ---

  EXPECT_TRUE(matches("__attribute__((noreturn)) void func();",
                      functionDecl(isNoReturn())));
  EXPECT_TRUE(matches("__attribute__((noreturn)) void func() {}",
                      functionDecl(isNoReturn())));

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

  // ---

  EXPECT_TRUE(matchesC("__attribute__((noreturn)) void func();",
                      functionDecl(isNoReturn())));
  EXPECT_TRUE(matchesC("__attribute__((noreturn)) void func() {}",
                      functionDecl(isNoReturn())));

  EXPECT_TRUE(matchesC("_Noreturn void func();",
                      functionDecl(isNoReturn())));
  EXPECT_TRUE(matchesC("_Noreturn void func() {}",
                      functionDecl(isNoReturn())));
}

TEST(TypeMatching, MatchesBool) {
  EXPECT_TRUE(matches("struct S { bool func(); };",
                      cxxMethodDecl(returns(booleanType()))));
  EXPECT_TRUE(notMatches("struct S { void func(); };",
                         cxxMethodDecl(returns(booleanType()))));
}

TEST(TypeMatching, MatchesVoid) {
  EXPECT_TRUE(matches("struct S { void func(); };",
                      cxxMethodDecl(returns(voidType()))));
}

TEST(TypeMatching, MatchesRealFloats) {
  EXPECT_TRUE(matches("struct S { float func(); };",
                      cxxMethodDecl(returns(realFloatingPointType()))));
  EXPECT_TRUE(notMatches("struct S { int func(); };",
                         cxxMethodDecl(returns(realFloatingPointType()))));
  EXPECT_TRUE(matches("struct S { long double func(); };",
                      cxxMethodDecl(returns(realFloatingPointType()))));
}

TEST(TypeMatching, MatchesArrayTypes) {
  EXPECT_TRUE(matches("int a[] = {2,3};", arrayType()));
  EXPECT_TRUE(matches("int a[42];", arrayType()));
  EXPECT_TRUE(matches("void f(int b) { int a[b]; }", arrayType()));

  EXPECT_TRUE(notMatches("struct A {}; A a[7];",
                         arrayType(hasElementType(builtinType()))));

  EXPECT_TRUE(matches(
    "int const a[] = { 2, 3 };",
    qualType(arrayType(hasElementType(builtinType())))));
  EXPECT_TRUE(matches(
    "int const a[] = { 2, 3 };",
    qualType(isConstQualified(), arrayType(hasElementType(builtinType())))));
  EXPECT_TRUE(matches(
    "typedef const int T; T x[] = { 1, 2 };",
    qualType(isConstQualified(), arrayType())));

  EXPECT_TRUE(notMatches(
    "int a[] = { 2, 3 };",
    qualType(isConstQualified(), arrayType(hasElementType(builtinType())))));
  EXPECT_TRUE(notMatches(
    "int a[] = { 2, 3 };",
    qualType(arrayType(hasElementType(isConstQualified(), builtinType())))));
  EXPECT_TRUE(notMatches(
    "int const a[] = { 2, 3 };",
    qualType(arrayType(hasElementType(builtinType())),
             unless(isConstQualified()))));

  EXPECT_TRUE(matches("int a[2];",
                      constantArrayType(hasElementType(builtinType()))));
  EXPECT_TRUE(matches("const int a = 0;", qualType(isInteger())));
}

TEST(TypeMatching, DecayedType) {
  EXPECT_TRUE(matches("void f(int i[]);", valueDecl(hasType(decayedType(hasDecayedType(pointerType()))))));
  EXPECT_TRUE(notMatches("int i[7];", decayedType()));
}

TEST(TypeMatching, MatchesComplexTypes) {
  EXPECT_TRUE(matches("_Complex float f;", complexType()));
  EXPECT_TRUE(matches(
    "_Complex float f;",
    complexType(hasElementType(builtinType()))));
  EXPECT_TRUE(notMatches(
    "_Complex float f;",
    complexType(hasElementType(isInteger()))));
}

TEST(NS, Anonymous) {
  EXPECT_TRUE(notMatches("namespace N {}", namespaceDecl(isAnonymous())));
  EXPECT_TRUE(matches("namespace {}", namespaceDecl(isAnonymous())));
}

TEST(EqualsBoundNodeMatcher, QualType) {
  EXPECT_TRUE(matches(
    "int i = 1;", varDecl(hasType(qualType().bind("type")),
                          hasInitializer(ignoringParenImpCasts(
                            hasType(qualType(equalsBoundNode("type"))))))));
  EXPECT_TRUE(notMatches("int i = 1.f;",
                         varDecl(hasType(qualType().bind("type")),
                                 hasInitializer(ignoringParenImpCasts(hasType(
                                   qualType(equalsBoundNode("type"))))))));
}

TEST(EqualsBoundNodeMatcher, NonMatchingTypes) {
  EXPECT_TRUE(notMatches(
    "int i = 1;", varDecl(namedDecl(hasName("i")).bind("name"),
                          hasInitializer(ignoringParenImpCasts(
                            hasType(qualType(equalsBoundNode("type"))))))));
}

TEST(EqualsBoundNodeMatcher, Stmt) {
  EXPECT_TRUE(
    matches("void f() { if(true) {} }",
            stmt(allOf(ifStmt().bind("if"),
                       hasParent(stmt(has(stmt(equalsBoundNode("if")))))))));

  EXPECT_TRUE(notMatches(
    "void f() { if(true) { if (true) {} } }",
    stmt(allOf(ifStmt().bind("if"), has(stmt(equalsBoundNode("if")))))));
}

TEST(EqualsBoundNodeMatcher, Decl) {
  EXPECT_TRUE(matches(
    "class X { class Y {}; };",
    decl(allOf(recordDecl(hasName("::X::Y")).bind("record"),
               hasParent(decl(has(decl(equalsBoundNode("record")))))))));

  EXPECT_TRUE(notMatches("class X { class Y {}; };",
                         decl(allOf(recordDecl(hasName("::X")).bind("record"),
                                    has(decl(equalsBoundNode("record")))))));
}

TEST(EqualsBoundNodeMatcher, Type) {
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

TEST(EqualsBoundNodeMatcher, UsingForEachDescendant) {
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
    functionDecl(returns(qualType().bind("type")),
                 forEachDescendant(varDecl(hasType(
                   qualType(equalsBoundNode("type")))).bind("decl"))),
    // Only i and j should match, not k.
    llvm::make_unique<VerifyIdIsBoundTo<VarDecl>>("decl", 2)));
}

TEST(EqualsBoundNodeMatcher, FiltersMatchedCombinations) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "void f() {"
      "  int x;"
      "  double d;"
      "  x = d + x - d + x;"
      "}",
    functionDecl(
      hasName("f"), forEachDescendant(varDecl().bind("d")),
      forEachDescendant(declRefExpr(to(decl(equalsBoundNode("d")))))),
    llvm::make_unique<VerifyIdIsBoundTo<VarDecl>>("d", 5)));
}

TEST(EqualsBoundNodeMatcher, UnlessDescendantsOfAncestorsMatch) {
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
    llvm::make_unique<VerifyIdIsBoundTo<Expr>>("data", 1)));

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

TEST(NullPointerConstants, Basic) {
  EXPECT_TRUE(matches("#define NULL ((void *)0)\n"
                        "void *v1 = NULL;", expr(nullPointerConstant())));
  EXPECT_TRUE(matches("void *v2 = nullptr;", expr(nullPointerConstant())));
  EXPECT_TRUE(matches("void *v3 = __null;", expr(nullPointerConstant())));
  EXPECT_TRUE(matches("char *cp = (char *)0;", expr(nullPointerConstant())));
  EXPECT_TRUE(matches("int *ip = 0;", expr(nullPointerConstant())));
  EXPECT_TRUE(notMatches("int i = 0;", expr(nullPointerConstant())));
}

TEST(HasExternalFormalLinkage, Basic) {
  EXPECT_TRUE(matches("int a = 0;", namedDecl(hasExternalFormalLinkage())));
  EXPECT_TRUE(
      notMatches("static int a = 0;", namedDecl(hasExternalFormalLinkage())));
  EXPECT_TRUE(notMatches("static void f(void) { int a = 0; }",
                         namedDecl(hasExternalFormalLinkage())));
  EXPECT_TRUE(matches("void f(void) { int a = 0; }",
                      namedDecl(hasExternalFormalLinkage())));

  // Despite having internal semantic linkage, the anonymous namespace member
  // has external linkage because the member has a unique name in all
  // translation units.
  EXPECT_TRUE(matches("namespace { int a = 0; }",
                      namedDecl(hasExternalFormalLinkage())));
}

TEST(HasDefaultArgument, Basic) {
  EXPECT_TRUE(matches("void x(int val = 0) {}", 
                      parmVarDecl(hasDefaultArgument())));
  EXPECT_TRUE(notMatches("void x(int val) {}",
                      parmVarDecl(hasDefaultArgument())));
}

TEST(IsArray, Basic) {
  EXPECT_TRUE(matches("struct MyClass {}; MyClass *p1 = new MyClass[10];",
                      cxxNewExpr(isArray())));
}

TEST(HasArraySize, Basic) {
  EXPECT_TRUE(matches("struct MyClass {}; MyClass *p1 = new MyClass[10];",
                      cxxNewExpr(hasArraySize(integerLiteral(equals(10))))));
}

TEST(HasDefinition, MatchesStructDefinition) {
  EXPECT_TRUE(matches("struct x {};",
                      cxxRecordDecl(hasDefinition())));
  EXPECT_TRUE(notMatches("struct x;",
                      cxxRecordDecl(hasDefinition())));
}

TEST(HasDefinition, MatchesClassDefinition) {
  EXPECT_TRUE(matches("class x {};",
                      cxxRecordDecl(hasDefinition())));
  EXPECT_TRUE(notMatches("class x;",
                      cxxRecordDecl(hasDefinition())));
}

TEST(HasDefinition, MatchesUnionDefinition) {
  EXPECT_TRUE(matches("union x {};",
                      cxxRecordDecl(hasDefinition())));
  EXPECT_TRUE(notMatches("union x;",
                      cxxRecordDecl(hasDefinition())));
}

TEST(IsScopedEnum, MatchesScopedEnum) {
  EXPECT_TRUE(matches("enum class X {};", enumDecl(isScoped())));
  EXPECT_TRUE(notMatches("enum X {};", enumDecl(isScoped())));
}

TEST(HasTrailingReturn, MatchesTrailingReturn) {
  EXPECT_TRUE(matches("auto Y() -> int { return 0; }",
                      functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(matches("auto X() -> int;", functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(notMatches("int X() { return 0; }", 
                      functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(notMatches("int X();", functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(notMatchesC("void X();", functionDecl(hasTrailingReturn())));
}

TEST(HasTrailingReturn, MatchesLambdaTrailingReturn) {
  EXPECT_TRUE(matches(
          "auto lambda2 = [](double x, double y) -> double {return x + y;};",
          functionDecl(hasTrailingReturn())));
  EXPECT_TRUE(notMatches(
          "auto lambda2 = [](double x, double y) {return x + y;};",
          functionDecl(hasTrailingReturn())));
}

TEST(IsAssignmentOperator, Basic) {
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

TEST(HasInit, Basic) {
  EXPECT_TRUE(
    matches("int x{0};",
            initListExpr(hasInit(0, expr()))));
  EXPECT_FALSE(
    matches("int x{0};",
            initListExpr(hasInit(1, expr()))));
  EXPECT_FALSE(
    matches("int x;",
            initListExpr(hasInit(0, expr()))));
}

TEST(Matcher, isMain) {
  EXPECT_TRUE(
    matches("int main() {}", functionDecl(isMain())));

  EXPECT_TRUE(
    notMatches("int main2() {}", functionDecl(isMain())));
}

} // namespace ast_matchers
} // namespace clang
