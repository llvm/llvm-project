//===- unittest/Tooling/ASTMatchersTest.cpp - AST matcher unit tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#if GTEST_HAS_DEATH_TEST
TEST(HasNameDeathTest, DiesOnEmptyName) {
  ASSERT_DEBUG_DEATH({
    DeclarationMatcher HasEmptyName = recordDecl(hasName(""));
    EXPECT_TRUE(notMatches("class X {};", HasEmptyName));
  }, "");
}

TEST(HasNameDeathTest, DiesOnEmptyPattern) {
  ASSERT_DEBUG_DEATH({
      DeclarationMatcher HasEmptyName = recordDecl(matchesName(""));
      EXPECT_TRUE(notMatches("class X {};", HasEmptyName));
    }, "");
}

TEST(IsDerivedFromDeathTest, DiesOnEmptyBaseName) {
  ASSERT_DEBUG_DEATH({
    DeclarationMatcher IsDerivedFromEmpty = cxxRecordDecl(isDerivedFrom(""));
    EXPECT_TRUE(notMatches("class X {};", IsDerivedFromEmpty));
  }, "");
}
#endif

TEST(Finder, DynamicOnlyAcceptsSomeMatchers) {
  MatchFinder Finder;
  EXPECT_TRUE(Finder.addDynamicMatcher(decl(), nullptr));
  EXPECT_TRUE(Finder.addDynamicMatcher(callExpr(), nullptr));
  EXPECT_TRUE(Finder.addDynamicMatcher(constantArrayType(hasSize(42)),
                                       nullptr));

  // Do not accept non-toplevel matchers.
  EXPECT_FALSE(Finder.addDynamicMatcher(isArrow(), nullptr));
  EXPECT_FALSE(Finder.addDynamicMatcher(hasSize(2), nullptr));
  EXPECT_FALSE(Finder.addDynamicMatcher(hasName("x"), nullptr));
}

TEST(Decl, MatchesDeclarations) {
  EXPECT_TRUE(notMatches("", decl(usingDecl())));
  EXPECT_TRUE(matches("namespace x { class X {}; } using x::X;",
                      decl(usingDecl())));
}

TEST(NameableDeclaration, MatchesVariousDecls) {
  DeclarationMatcher NamedX = namedDecl(hasName("X"));
  EXPECT_TRUE(matches("typedef int X;", NamedX));
  EXPECT_TRUE(matches("int X;", NamedX));
  EXPECT_TRUE(matches("class foo { virtual void X(); };", NamedX));
  EXPECT_TRUE(matches("void foo() try { } catch(int X) { }", NamedX));
  EXPECT_TRUE(matches("void foo() { int X; }", NamedX));
  EXPECT_TRUE(matches("namespace X { }", NamedX));
  EXPECT_TRUE(matches("enum X { A, B, C };", NamedX));

  EXPECT_TRUE(notMatches("#define X 1", NamedX));
}

TEST(NameableDeclaration, REMatchesVariousDecls) {
  DeclarationMatcher NamedX = namedDecl(matchesName("::X"));
  EXPECT_TRUE(matches("typedef int Xa;", NamedX));
  EXPECT_TRUE(matches("int Xb;", NamedX));
  EXPECT_TRUE(matches("class foo { virtual void Xc(); };", NamedX));
  EXPECT_TRUE(matches("void foo() try { } catch(int Xdef) { }", NamedX));
  EXPECT_TRUE(matches("void foo() { int Xgh; }", NamedX));
  EXPECT_TRUE(matches("namespace Xij { }", NamedX));
  EXPECT_TRUE(matches("enum X { A, B, C };", NamedX));

  EXPECT_TRUE(notMatches("#define Xkl 1", NamedX));

  DeclarationMatcher StartsWithNo = namedDecl(matchesName("::no"));
  EXPECT_TRUE(matches("int no_foo;", StartsWithNo));
  EXPECT_TRUE(matches("class foo { virtual void nobody(); };", StartsWithNo));

  DeclarationMatcher Abc = namedDecl(matchesName("a.*b.*c"));
  EXPECT_TRUE(matches("int abc;", Abc));
  EXPECT_TRUE(matches("int aFOObBARc;", Abc));
  EXPECT_TRUE(notMatches("int cab;", Abc));
  EXPECT_TRUE(matches("int cabc;", Abc));

  DeclarationMatcher StartsWithK = namedDecl(matchesName(":k[^:]*$"));
  EXPECT_TRUE(matches("int k;", StartsWithK));
  EXPECT_TRUE(matches("int kAbc;", StartsWithK));
  EXPECT_TRUE(matches("namespace x { int kTest; }", StartsWithK));
  EXPECT_TRUE(matches("class C { int k; };", StartsWithK));
  EXPECT_TRUE(notMatches("class C { int ckc; };", StartsWithK));
}

TEST(DeclarationMatcher, MatchClass) {
  DeclarationMatcher ClassMatcher(recordDecl());
  llvm::Triple Triple(llvm::sys::getDefaultTargetTriple());
  if (Triple.getOS() != llvm::Triple::Win32 ||
      Triple.getEnvironment() != llvm::Triple::MSVC)
    EXPECT_FALSE(matches("", ClassMatcher));
  else
    // Matches class type_info.
    EXPECT_TRUE(matches("", ClassMatcher));

  DeclarationMatcher ClassX = recordDecl(recordDecl(hasName("X")));
  EXPECT_TRUE(matches("class X;", ClassX));
  EXPECT_TRUE(matches("class X {};", ClassX));
  EXPECT_TRUE(matches("template<class T> class X {};", ClassX));
  EXPECT_TRUE(notMatches("", ClassX));
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

TEST(DeclarationMatcher, hasMethod) {
  EXPECT_TRUE(matches("class A { void func(); };",
                      cxxRecordDecl(hasMethod(hasName("func")))));
  EXPECT_TRUE(notMatches("class A { void func(); };",
                         cxxRecordDecl(hasMethod(isPublic()))));
}

TEST(DeclarationMatcher, ClassDerivedFromDependentTemplateSpecialization) {
  EXPECT_TRUE(matches(
     "template <typename T> struct A {"
     "  template <typename T2> struct F {};"
     "};"
     "template <typename T> struct B : A<T>::template F<T> {};"
     "B<int> b;",
    cxxRecordDecl(hasName("B"), isDerivedFrom(recordDecl()))));
}

TEST(DeclarationMatcher, hasDeclContext) {
  EXPECT_TRUE(matches(
      "namespace N {"
      "  namespace M {"
      "    class D {};"
      "  }"
      "}",
      recordDecl(hasDeclContext(namespaceDecl(hasName("M"))))));
  EXPECT_TRUE(notMatches(
      "namespace N {"
      "  namespace M {"
      "    class D {};"
      "  }"
      "}",
      recordDecl(hasDeclContext(namespaceDecl(hasName("N"))))));

  EXPECT_TRUE(matches("namespace {"
                      "  namespace M {"
                      "    class D {};"
                      "  }"
                      "}",
                      recordDecl(hasDeclContext(namespaceDecl(
                          hasName("M"), hasDeclContext(namespaceDecl()))))));

  EXPECT_TRUE(matches("class D{};", decl(hasDeclContext(decl()))));
}

TEST(DeclarationMatcher, translationUnitDecl) {
  const std::string Code = "int MyVar1;\n"
                           "namespace NameSpace {\n"
                           "int MyVar2;\n"
                           "}  // namespace NameSpace\n";
  EXPECT_TRUE(matches(
      Code, varDecl(hasName("MyVar1"), hasDeclContext(translationUnitDecl()))));
  EXPECT_FALSE(matches(
      Code, varDecl(hasName("MyVar2"), hasDeclContext(translationUnitDecl()))));
  EXPECT_TRUE(matches(
      Code,
      varDecl(hasName("MyVar2"),
              hasDeclContext(decl(hasDeclContext(translationUnitDecl()))))));
}

TEST(DeclarationMatcher, LinkageSpecification) {
  EXPECT_TRUE(matches("extern \"C\" { void foo() {}; }", linkageSpecDecl()));
  EXPECT_TRUE(notMatches("void foo() {};", linkageSpecDecl()));
}

TEST(ClassTemplate, DoesNotMatchClass) {
  DeclarationMatcher ClassX = classTemplateDecl(hasName("X"));
  EXPECT_TRUE(notMatches("class X;", ClassX));
  EXPECT_TRUE(notMatches("class X {};", ClassX));
}

TEST(ClassTemplate, MatchesClassTemplate) {
  DeclarationMatcher ClassX = classTemplateDecl(hasName("X"));
  EXPECT_TRUE(matches("template<typename T> class X {};", ClassX));
  EXPECT_TRUE(matches("class Z { template<class T> class X {}; };", ClassX));
}

TEST(ClassTemplate, DoesNotMatchClassTemplateExplicitSpecialization) {
  EXPECT_TRUE(notMatches("template<typename T> class X { };"
                         "template<> class X<int> { int a; };",
              classTemplateDecl(hasName("X"),
                                hasDescendant(fieldDecl(hasName("a"))))));
}

TEST(ClassTemplate, DoesNotMatchClassTemplatePartialSpecialization) {
  EXPECT_TRUE(notMatches("template<typename T, typename U> class X { };"
                         "template<typename T> class X<T, int> { int a; };",
              classTemplateDecl(hasName("X"),
                                hasDescendant(fieldDecl(hasName("a"))))));
}

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

TEST(ConstructVariadic, MismatchedTypes_Regression) {
  EXPECT_TRUE(
      matches("const int a = 0;",
              internal::DynTypedMatcher::constructVariadic(
                  internal::DynTypedMatcher::VO_AnyOf,
                  ast_type_traits::ASTNodeKind::getFromNodeKind<QualType>(),
                  {isConstQualified(), arrayType()})
                  .convertTo<QualType>()));
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

TEST(DeclarationMatcher, HasAttr) {
  EXPECT_TRUE(matches("struct __attribute__((warn_unused)) X {};",
                      decl(hasAttr(clang::attr::WarnUnused))));
  EXPECT_FALSE(matches("struct X {};",
                       decl(hasAttr(clang::attr::WarnUnused))));
}

TEST(DeclarationMatcher, MatchCudaDecl) {
  EXPECT_TRUE(matchesWithCuda("__global__ void f() { }"
                              "void g() { f<<<1, 2>>>(); }",
                              cudaKernelCallExpr()));
  EXPECT_TRUE(matchesWithCuda("__attribute__((device)) void f() {}",
                              hasAttr(clang::attr::CUDADevice)));
  EXPECT_TRUE(notMatchesWithCuda("void f() {}",
                                 cudaKernelCallExpr()));
  EXPECT_FALSE(notMatchesWithCuda("__attribute__((global)) void f() {}",
                                  hasAttr(clang::attr::CUDAGlobal)));
}

// Implements a run method that returns whether BoundNodes contains a
// Decl bound to Id that can be dynamically cast to T.
// Optionally checks that the check succeeded a specific number of times.
template <typename T>
class VerifyIdIsBoundTo : public BoundNodesCallback {
public:
  // Create an object that checks that a node of type \c T was bound to \c Id.
  // Does not check for a certain number of matches.
  explicit VerifyIdIsBoundTo(llvm::StringRef Id)
    : Id(Id), ExpectedCount(-1), Count(0) {}

  // Create an object that checks that a node of type \c T was bound to \c Id.
  // Checks that there were exactly \c ExpectedCount matches.
  VerifyIdIsBoundTo(llvm::StringRef Id, int ExpectedCount)
    : Id(Id), ExpectedCount(ExpectedCount), Count(0) {}

  // Create an object that checks that a node of type \c T was bound to \c Id.
  // Checks that there was exactly one match with the name \c ExpectedName.
  // Note that \c T must be a NamedDecl for this to work.
  VerifyIdIsBoundTo(llvm::StringRef Id, llvm::StringRef ExpectedName,
                    int ExpectedCount = 1)
      : Id(Id), ExpectedCount(ExpectedCount), Count(0),
        ExpectedName(ExpectedName) {}

  void onEndOfTranslationUnit() override {
    if (ExpectedCount != -1)
      EXPECT_EQ(ExpectedCount, Count);
    if (!ExpectedName.empty())
      EXPECT_EQ(ExpectedName, Name);
    Count = 0;
    Name.clear();
  }

  ~VerifyIdIsBoundTo() override {
    EXPECT_EQ(0, Count);
    EXPECT_EQ("", Name);
  }

  bool run(const BoundNodes *Nodes) override {
    const BoundNodes::IDToNodeMap &M = Nodes->getMap();
    if (Nodes->getNodeAs<T>(Id)) {
      ++Count;
      if (const NamedDecl *Named = Nodes->getNodeAs<NamedDecl>(Id)) {
        Name = Named->getNameAsString();
      } else if (const NestedNameSpecifier *NNS =
                 Nodes->getNodeAs<NestedNameSpecifier>(Id)) {
        llvm::raw_string_ostream OS(Name);
        NNS->print(OS, PrintingPolicy(LangOptions()));
      }
      BoundNodes::IDToNodeMap::const_iterator I = M.find(Id);
      EXPECT_NE(M.end(), I);
      if (I != M.end())
        EXPECT_EQ(Nodes->getNodeAs<T>(Id), I->second.get<T>());
      return true;
    }
    EXPECT_TRUE(M.count(Id) == 0 ||
                M.find(Id)->second.template get<T>() == nullptr);
    return false;
  }

  bool run(const BoundNodes *Nodes, ASTContext *Context) override {
    return run(Nodes);
  }

private:
  const std::string Id;
  const int ExpectedCount;
  int Count;
  const std::string ExpectedName;
  std::string Name;
};

TEST(HasDescendant, MatchesDescendantTypes) {
  EXPECT_TRUE(matches("void f() { int i = 3; }",
                      decl(hasDescendant(loc(builtinType())))));
  EXPECT_TRUE(matches("void f() { int i = 3; }",
                      stmt(hasDescendant(builtinType()))));

  EXPECT_TRUE(matches("void f() { int i = 3; }",
                      stmt(hasDescendant(loc(builtinType())))));
  EXPECT_TRUE(matches("void f() { int i = 3; }",
                      stmt(hasDescendant(qualType(builtinType())))));

  EXPECT_TRUE(notMatches("void f() { float f = 2.0f; }",
                         stmt(hasDescendant(isInteger()))));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f() { int a; float c; int d; int e; }",
      functionDecl(forEachDescendant(
          varDecl(hasDescendant(isInteger())).bind("x"))),
      new VerifyIdIsBoundTo<Decl>("x", 3)));
}

TEST(HasDescendant, MatchesDescendantsOfTypes) {
  EXPECT_TRUE(matches("void f() { int*** i; }",
                      qualType(hasDescendant(builtinType()))));
  EXPECT_TRUE(matches("void f() { int*** i; }",
                      qualType(hasDescendant(
                          pointerType(pointee(builtinType()))))));
  EXPECT_TRUE(matches("void f() { int*** i; }",
                      typeLoc(hasDescendant(loc(builtinType())))));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f() { int*** i; }",
      qualType(asString("int ***"), forEachDescendant(pointerType().bind("x"))),
      new VerifyIdIsBoundTo<Type>("x", 2)));
}

TEST(Has, MatchesChildrenOfTypes) {
  EXPECT_TRUE(matches("int i;",
                      varDecl(hasName("i"), has(isInteger()))));
  EXPECT_TRUE(notMatches("int** i;",
                         varDecl(hasName("i"), has(isInteger()))));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "int (*f)(float, int);",
      qualType(functionType(), forEach(qualType(isInteger()).bind("x"))),
      new VerifyIdIsBoundTo<QualType>("x", 2)));
}

TEST(Has, MatchesChildTypes) {
  EXPECT_TRUE(matches(
      "int* i;",
      varDecl(hasName("i"), hasType(qualType(has(builtinType()))))));
  EXPECT_TRUE(notMatches(
      "int* i;",
      varDecl(hasName("i"), hasType(qualType(has(pointerType()))))));
}

TEST(ValueDecl, Matches) {
  EXPECT_TRUE(matches("enum EnumType { EnumValue };",
                      valueDecl(hasType(asString("enum EnumType")))));
  EXPECT_TRUE(matches("void FunctionDecl();",
                      valueDecl(hasType(asString("void (void)")))));
}

TEST(Enum, DoesNotMatchClasses) {
  EXPECT_TRUE(notMatches("class X {};", enumDecl(hasName("X"))));
}

TEST(Enum, MatchesEnums) {
  EXPECT_TRUE(matches("enum X {};", enumDecl(hasName("X"))));
}

TEST(EnumConstant, Matches) {
  DeclarationMatcher Matcher = enumConstantDecl(hasName("A"));
  EXPECT_TRUE(matches("enum X{ A };", Matcher));
  EXPECT_TRUE(notMatches("enum X{ B };", Matcher));
  EXPECT_TRUE(notMatches("enum X {};", Matcher));
}

TEST(StatementMatcher, Has) {
  StatementMatcher HasVariableI =
      expr(hasType(pointsTo(recordDecl(hasName("X")))),
           has(declRefExpr(to(varDecl(hasName("i"))))));

  EXPECT_TRUE(matches(
      "class X; X *x(int); void c() { int i; x(i); }", HasVariableI));
  EXPECT_TRUE(notMatches(
      "class X; X *x(int); void c() { int i; x(42); }", HasVariableI));
}

TEST(StatementMatcher, HasDescendant) {
  StatementMatcher HasDescendantVariableI =
      expr(hasType(pointsTo(recordDecl(hasName("X")))),
           hasDescendant(declRefExpr(to(varDecl(hasName("i"))))));

  EXPECT_TRUE(matches(
      "class X; X *x(bool); bool b(int); void c() { int i; x(b(i)); }",
      HasDescendantVariableI));
  EXPECT_TRUE(notMatches(
      "class X; X *x(bool); bool b(int); void c() { int i; x(b(42)); }",
      HasDescendantVariableI));
}

TEST(TypeMatcher, MatchesClassType) {
  TypeMatcher TypeA = hasDeclaration(recordDecl(hasName("A")));

  EXPECT_TRUE(matches("class A { public: A *a; };", TypeA));
  EXPECT_TRUE(notMatches("class A {};", TypeA));

  TypeMatcher TypeDerivedFromA =
      hasDeclaration(cxxRecordDecl(isDerivedFrom("A")));

  EXPECT_TRUE(matches("class A {}; class B : public A { public: B *b; };",
              TypeDerivedFromA));
  EXPECT_TRUE(notMatches("class A {};", TypeA));

  TypeMatcher TypeAHasClassB = hasDeclaration(
      recordDecl(hasName("A"), has(recordDecl(hasName("B")))));

  EXPECT_TRUE(
      matches("class A { public: A *a; class B {}; };", TypeAHasClassB));

  EXPECT_TRUE(matchesC("struct S {}; void f(void) { struct S s; }",
                       varDecl(hasType(namedDecl(hasName("S"))))));
}

TEST(TypeMatcher, MatchesDeclTypes) {
  // TypedefType -> TypedefNameDecl
  EXPECT_TRUE(matches("typedef int I; void f(I i);",
                      parmVarDecl(hasType(namedDecl(hasName("I"))))));
  // ObjCObjectPointerType
  EXPECT_TRUE(matchesObjC("@interface Foo @end void f(Foo *f);",
                          parmVarDecl(hasType(objcObjectPointerType()))));
  // ObjCObjectPointerType -> ObjCInterfaceType -> ObjCInterfaceDecl
  EXPECT_TRUE(matchesObjC(
      "@interface Foo @end void f(Foo *f);",
      parmVarDecl(hasType(pointsTo(objcInterfaceDecl(hasName("Foo")))))));
  // TemplateTypeParmType
  EXPECT_TRUE(matches("template <typename T> void f(T t);",
                      parmVarDecl(hasType(templateTypeParmType()))));
  // TemplateTypeParmType -> TemplateTypeParmDecl
  EXPECT_TRUE(matches("template <typename T> void f(T t);",
                      parmVarDecl(hasType(namedDecl(hasName("T"))))));
  // InjectedClassNameType
  EXPECT_TRUE(matches("template <typename T> struct S {"
                      "  void f(S s);"
                      "};",
                      parmVarDecl(hasType(injectedClassNameType()))));
  EXPECT_TRUE(notMatches("template <typename T> struct S {"
                         "  void g(S<T> s);"
                         "};",
                         parmVarDecl(hasType(injectedClassNameType()))));
  // InjectedClassNameType -> CXXRecordDecl
  EXPECT_TRUE(matches("template <typename T> struct S {"
                      "  void f(S s);"
                      "};",
                      parmVarDecl(hasType(namedDecl(hasName("S"))))));

  static const char Using[] = "template <typename T>"
                              "struct Base {"
                              "  typedef T Foo;"
                              "};"
                              ""
                              "template <typename T>"
                              "struct S : private Base<T> {"
                              "  using typename Base<T>::Foo;"
                              "  void f(Foo);"
                              "};";
  // UnresolvedUsingTypenameDecl
  EXPECT_TRUE(matches(Using, unresolvedUsingTypenameDecl(hasName("Foo"))));
  // UnresolvedUsingTypenameType -> UnresolvedUsingTypenameDecl
  EXPECT_TRUE(matches(Using, parmVarDecl(hasType(namedDecl(hasName("Foo"))))));
}

TEST(Matcher, BindMatchedNodes) {
  DeclarationMatcher ClassX = has(recordDecl(hasName("::X")).bind("x"));

  EXPECT_TRUE(matchAndVerifyResultTrue("class X {};",
      ClassX, new VerifyIdIsBoundTo<CXXRecordDecl>("x")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class X {};",
      ClassX, new VerifyIdIsBoundTo<CXXRecordDecl>("other-id")));

  TypeMatcher TypeAHasClassB = hasDeclaration(
      recordDecl(hasName("A"), has(recordDecl(hasName("B")).bind("b"))));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { public: A *a; class B {}; };",
      TypeAHasClassB,
      new VerifyIdIsBoundTo<Decl>("b")));

  StatementMatcher MethodX =
      callExpr(callee(cxxMethodDecl(hasName("x")))).bind("x");

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { void x() { x(); } };",
      MethodX,
      new VerifyIdIsBoundTo<CXXMemberCallExpr>("x")));
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
      new VerifyIdIsBoundTo<CallExpr>("x")));
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
      new VerifyIdIsBoundTo<Decl>("x", 2)));
}

TEST(HasDeclaration, HasDeclarationOfEnumType) {
  EXPECT_TRUE(matches("enum X {}; void y(X *x) { x; }",
                      expr(hasType(pointsTo(
                          qualType(hasDeclaration(enumDecl(hasName("X")))))))));
}

TEST(HasDeclaration, HasGetDeclTraitTest) {
  EXPECT_TRUE(internal::has_getDecl<TypedefType>::value);
  EXPECT_TRUE(internal::has_getDecl<RecordType>::value);
  EXPECT_FALSE(internal::has_getDecl<TemplateSpecializationType>::value);
}

TEST(HasDeclaration, HasDeclarationOfTypeWithDecl) {
  EXPECT_TRUE(matches("typedef int X; X a;",
                      varDecl(hasName("a"),
                              hasType(typedefType(hasDeclaration(decl()))))));

  // FIXME: Add tests for other types with getDecl() (e.g. RecordType)
}

TEST(HasDeclaration, HasDeclarationOfTemplateSpecializationType) {
  EXPECT_TRUE(matches("template <typename T> class A {}; A<int> a;",
                      varDecl(hasType(templateSpecializationType(
                          hasDeclaration(namedDecl(hasName("A"))))))));
}

TEST(HasType, TakesQualTypeMatcherAndMatchesExpr) {
  TypeMatcher ClassX = hasDeclaration(recordDecl(hasName("X")));
  EXPECT_TRUE(
      matches("class X {}; void y(X &x) { x; }", expr(hasType(ClassX))));
  EXPECT_TRUE(
      notMatches("class X {}; void y(X *x) { x; }",
                 expr(hasType(ClassX))));
  EXPECT_TRUE(
      matches("class X {}; void y(X *x) { x; }",
              expr(hasType(pointsTo(ClassX)))));
}

TEST(HasType, TakesQualTypeMatcherAndMatchesValueDecl) {
  TypeMatcher ClassX = hasDeclaration(recordDecl(hasName("X")));
  EXPECT_TRUE(
      matches("class X {}; void y() { X x; }", varDecl(hasType(ClassX))));
  EXPECT_TRUE(
      notMatches("class X {}; void y() { X *x; }", varDecl(hasType(ClassX))));
  EXPECT_TRUE(
      matches("class X {}; void y() { X *x; }",
              varDecl(hasType(pointsTo(ClassX)))));
}

TEST(HasType, TakesDeclMatcherAndMatchesExpr) {
  DeclarationMatcher ClassX = recordDecl(hasName("X"));
  EXPECT_TRUE(
      matches("class X {}; void y(X &x) { x; }", expr(hasType(ClassX))));
  EXPECT_TRUE(
      notMatches("class X {}; void y(X *x) { x; }",
                 expr(hasType(ClassX))));
}

TEST(HasType, TakesDeclMatcherAndMatchesValueDecl) {
  DeclarationMatcher ClassX = recordDecl(hasName("X"));
  EXPECT_TRUE(
      matches("class X {}; void y() { X x; }", varDecl(hasType(ClassX))));
  EXPECT_TRUE(
      notMatches("class X {}; void y() { X *x; }", varDecl(hasType(ClassX))));
}

TEST(HasType, MatchesTypedefDecl) {
  EXPECT_TRUE(matches("typedef int X;", typedefDecl(hasType(asString("int")))));
  EXPECT_TRUE(matches("typedef const int T;",
                      typedefDecl(hasType(asString("const int")))));
  EXPECT_TRUE(notMatches("typedef const int T;",
                         typedefDecl(hasType(asString("int")))));
  EXPECT_TRUE(matches("typedef int foo; typedef foo bar;",
                      typedefDecl(hasType(asString("foo")), hasName("bar"))));
}

TEST(HasTypeLoc, MatchesDeclaratorDecls) {
  EXPECT_TRUE(matches("int x;",
                      varDecl(hasName("x"), hasTypeLoc(loc(asString("int"))))));

  // Make sure we don't crash on implicit constructors.
  EXPECT_TRUE(notMatches("class X {}; X x;",
                         declaratorDecl(hasTypeLoc(loc(asString("int"))))));
}

TEST(Matcher, Call) {
  // FIXME: Do we want to overload Call() to directly take
  // Matcher<Decl>, too?
  StatementMatcher MethodX =
      callExpr(hasDeclaration(cxxMethodDecl(hasName("x"))));

  EXPECT_TRUE(matches("class Y { void x() { x(); } };", MethodX));
  EXPECT_TRUE(notMatches("class Y { void x() {} };", MethodX));

  StatementMatcher MethodOnY =
      cxxMemberCallExpr(on(hasType(recordDecl(hasName("Y")))));

  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
                 MethodOnY));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
                 MethodOnY));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
                 MethodOnY));

  StatementMatcher MethodOnYPointer =
      cxxMemberCallExpr(on(hasType(pointsTo(recordDecl(hasName("Y"))))));

  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
              MethodOnYPointer));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
                 MethodOnYPointer));
  EXPECT_TRUE(
      notMatches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
                 MethodOnYPointer));
}

TEST(Matcher, Lambda) {
  EXPECT_TRUE(matches("auto f = [] (int i) { return i; };",
                      lambdaExpr()));
}

TEST(Matcher, ForRange) {
  EXPECT_TRUE(matches("int as[] = { 1, 2, 3 };"
                      "void f() { for (auto &a : as); }",
                      cxxForRangeStmt()));
  EXPECT_TRUE(notMatches("void f() { for (int i; i<5; ++i); }",
                         cxxForRangeStmt()));
}

TEST(Matcher, SubstNonTypeTemplateParm) {
  EXPECT_FALSE(matches("template<int N>\n"
                       "struct A {  static const int n = 0; };\n"
                       "struct B : public A<42> {};",
                       substNonTypeTemplateParmExpr()));
  EXPECT_TRUE(matches("template<int N>\n"
                      "struct A {  static const int n = N; };\n"
                      "struct B : public A<42> {};",
                      substNonTypeTemplateParmExpr()));
}

TEST(Matcher, NonTypeTemplateParmDecl) {
  EXPECT_TRUE(matches("template <int N> void f();",
                      nonTypeTemplateParmDecl(hasName("N"))));
  EXPECT_TRUE(
      notMatches("template <typename T> void f();", nonTypeTemplateParmDecl()));
}

TEST(Matcher, templateTypeParmDecl) {
  EXPECT_TRUE(matches("template <typename T> void f();",
                      templateTypeParmDecl(hasName("T"))));
  EXPECT_TRUE(
      notMatches("template <int N> void f();", templateTypeParmDecl()));
}

TEST(Matcher, UserDefinedLiteral) {
  EXPECT_TRUE(matches("constexpr char operator \"\" _inc (const char i) {"
                      "  return i + 1;"
                      "}"
                      "char c = 'a'_inc;",
                      userDefinedLiteral()));
}

TEST(Matcher, FlowControl) {
  EXPECT_TRUE(matches("void f() { while(true) { break; } }", breakStmt()));
  EXPECT_TRUE(matches("void f() { while(true) { continue; } }",
                      continueStmt()));
  EXPECT_TRUE(matches("void f() { goto FOO; FOO: ;}", gotoStmt()));
  EXPECT_TRUE(matches("void f() { goto FOO; FOO: ;}", labelStmt()));
  EXPECT_TRUE(matches("void f() { return; }", returnStmt()));
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

TEST(Matcher, OverloadedOperatorCall) {
  StatementMatcher OpCall = cxxOperatorCallExpr();
  // Unary operator
  EXPECT_TRUE(matches("class Y { }; "
              "bool operator!(Y x) { return false; }; "
              "Y y; bool c = !y;", OpCall));
  // No match -- special operators like "new", "delete"
  // FIXME: operator new takes size_t, for which we need stddef.h, for which
  // we need to figure out include paths in the test.
  // EXPECT_TRUE(NotMatches("#include <stddef.h>\n"
  //             "class Y { }; "
  //             "void *operator new(size_t size) { return 0; } "
  //             "Y *y = new Y;", OpCall));
  EXPECT_TRUE(notMatches("class Y { }; "
              "void operator delete(void *p) { } "
              "void a() {Y *y = new Y; delete y;}", OpCall));
  // Binary operator
  EXPECT_TRUE(matches("class Y { }; "
              "bool operator&&(Y x, Y y) { return true; }; "
              "Y a; Y b; bool c = a && b;",
              OpCall));
  // No match -- normal operator, not an overloaded one.
  EXPECT_TRUE(notMatches("bool x = true, y = true; bool t = x && y;", OpCall));
  EXPECT_TRUE(notMatches("int t = 5 << 2;", OpCall));
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
      new VerifyIdIsBoundTo<CXXOperatorCallExpr>("x", 2)));
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

TEST(Matcher, ThisPointerType) {
  StatementMatcher MethodOnY =
    cxxMemberCallExpr(thisPointerType(recordDecl(hasName("Y"))));

  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y *&y) { y->x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z(Y y[]) { y->x(); }",
              MethodOnY));
  EXPECT_TRUE(
      matches("class Y { public: void x(); }; void z() { Y *y; y->x(); }",
              MethodOnY));

  EXPECT_TRUE(matches(
      "class Y {"
      "  public: virtual void x();"
      "};"
      "class X : public Y {"
      "  public: virtual void x();"
      "};"
      "void z() { X *x; x->Y::x(); }", MethodOnY));
}

TEST(Matcher, VariableUsage) {
  StatementMatcher Reference =
      declRefExpr(to(
          varDecl(hasInitializer(
              cxxMemberCallExpr(thisPointerType(recordDecl(hasName("Y"))))))));

  EXPECT_TRUE(matches(
      "class Y {"
      " public:"
      "  bool x() const;"
      "};"
      "void z(const Y &y) {"
      "  bool b = y.x();"
      "  if (b) {}"
      "}", Reference));

  EXPECT_TRUE(notMatches(
      "class Y {"
      " public:"
      "  bool x() const;"
      "};"
      "void z(const Y &y) {"
      "  bool b = y.x();"
      "}", Reference));
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

TEST(Matcher, VarDecl_StorageDuration) {
  std::string T =
      "void f() { int x; static int y; } int a;";

  EXPECT_TRUE(matches(T, varDecl(hasName("x"), hasAutomaticStorageDuration())));
  EXPECT_TRUE(
      notMatches(T, varDecl(hasName("y"), hasAutomaticStorageDuration())));
  EXPECT_TRUE(
      notMatches(T, varDecl(hasName("a"), hasAutomaticStorageDuration())));

  EXPECT_TRUE(matches(T, varDecl(hasName("y"), hasStaticStorageDuration())));
  EXPECT_TRUE(matches(T, varDecl(hasName("a"), hasStaticStorageDuration())));
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

TEST(Matcher, CalledVariable) {
  StatementMatcher CallOnVariableY =
      cxxMemberCallExpr(on(declRefExpr(to(varDecl(hasName("y"))))));

  EXPECT_TRUE(matches(
      "class Y { public: void x() { Y y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(matches(
      "class Y { public: void x() const { Y y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(matches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { X y; y.x(); } };", CallOnVariableY));
  EXPECT_TRUE(matches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { X *y; y->x(); } };", CallOnVariableY));
  EXPECT_TRUE(notMatches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { unsigned long y; ((X*)y)->x(); } };",
      CallOnVariableY));
}

TEST(UnaryExprOrTypeTraitExpr, MatchesSizeOfAndAlignOf) {
  EXPECT_TRUE(matches("void x() { int a = sizeof(a); }",
                      unaryExprOrTypeTraitExpr()));
  EXPECT_TRUE(notMatches("void x() { int a = sizeof(a); }",
                         alignOfExpr(anything())));
  // FIXME: Uncomment once alignof is enabled.
  // EXPECT_TRUE(matches("void x() { int a = alignof(a); }",
  //                     unaryExprOrTypeTraitExpr()));
  // EXPECT_TRUE(notMatches("void x() { int a = alignof(a); }",
  //                        sizeOfExpr()));
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

TEST(MemberExpression, DoesNotMatchClasses) {
  EXPECT_TRUE(notMatches("class Y { void x() {} };", memberExpr()));
}

TEST(MemberExpression, MatchesMemberFunctionCall) {
  EXPECT_TRUE(matches("class Y { void x() { x(); } };", memberExpr()));
}

TEST(MemberExpression, MatchesVariable) {
  EXPECT_TRUE(
      matches("class Y { void x() { this->y; } int y; };", memberExpr()));
  EXPECT_TRUE(
      matches("class Y { void x() { y; } int y; };", memberExpr()));
  EXPECT_TRUE(
      matches("class Y { void x() { Y y; y.y; } int y; };", memberExpr()));
}

TEST(MemberExpression, MatchesStaticVariable) {
  EXPECT_TRUE(matches("class Y { void x() { this->y; } static int y; };",
              memberExpr()));
  EXPECT_TRUE(notMatches("class Y { void x() { y; } static int y; };",
              memberExpr()));
  EXPECT_TRUE(notMatches("class Y { void x() { Y::y; } static int y; };",
              memberExpr()));
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
}

TEST(Callee, MatchesDeclarations) {
  StatementMatcher CallMethodX = callExpr(callee(cxxMethodDecl(hasName("x"))));

  EXPECT_TRUE(matches("class Y { void x() { x(); } };", CallMethodX));
  EXPECT_TRUE(notMatches("class Y { void x() {} };", CallMethodX));

  CallMethodX = callExpr(callee(cxxConversionDecl()));
  EXPECT_TRUE(
      matches("struct Y { operator int() const; }; int i = Y();", CallMethodX));
  EXPECT_TRUE(notMatches("struct Y { operator int() const; }; Y y = Y();",
                         CallMethodX));
}

TEST(ConversionDeclaration, IsExplicit) {
  EXPECT_TRUE(matches("struct S { explicit operator int(); };",
                      cxxConversionDecl(isExplicit())));
  EXPECT_TRUE(notMatches("struct S { operator int(); };",
                         cxxConversionDecl(isExplicit())));
}

TEST(Callee, MatchesMemberExpressions) {
  EXPECT_TRUE(matches("class Y { void x() { this->x(); } };",
              callExpr(callee(memberExpr()))));
  EXPECT_TRUE(
      notMatches("class Y { void x() { this->x(); } };", callExpr(callee(callExpr()))));
}

TEST(Function, MatchesFunctionDeclarations) {
  StatementMatcher CallFunctionF = callExpr(callee(functionDecl(hasName("f"))));

  EXPECT_TRUE(matches("void f() { f(); }", CallFunctionF));
  EXPECT_TRUE(notMatches("void f() { }", CallFunctionF));

  if (llvm::Triple(llvm::sys::getDefaultTargetTriple()).getOS() !=
      llvm::Triple::Win32) {
    // FIXME: Make this work for MSVC.
    // Dependent contexts, but a non-dependent call.
    EXPECT_TRUE(matches("void f(); template <int N> void g() { f(); }",
                        CallFunctionF));
    EXPECT_TRUE(
        matches("void f(); template <int N> struct S { void g() { f(); } };",
                CallFunctionF));
  }

  // Depedent calls don't match.
  EXPECT_TRUE(
      notMatches("void f(int); template <typename T> void g(T t) { f(t); }",
                 CallFunctionF));
  EXPECT_TRUE(
      notMatches("void f(int);"
                 "template <typename T> struct S { void g(T t) { f(t); } };",
                 CallFunctionF));

  EXPECT_TRUE(matches("void f(...);", functionDecl(isVariadic())));
  EXPECT_TRUE(notMatches("void f(int);", functionDecl(isVariadic())));
  EXPECT_TRUE(notMatches("template <typename... Ts> void f(Ts...);",
                         functionDecl(isVariadic())));
  EXPECT_TRUE(notMatches("void f();", functionDecl(isVariadic())));
  EXPECT_TRUE(notMatchesC("void f();", functionDecl(isVariadic())));
  EXPECT_TRUE(matches("void f(...);", functionDecl(parameterCountIs(0))));
  EXPECT_TRUE(matchesC("void f();", functionDecl(parameterCountIs(0))));
  EXPECT_TRUE(matches("void f(int, ...);", functionDecl(parameterCountIs(1))));
}

TEST(FunctionTemplate, MatchesFunctionTemplateDeclarations) {
  EXPECT_TRUE(
      matches("template <typename T> void f(T t) {}",
      functionTemplateDecl(hasName("f"))));
}

TEST(FunctionTemplate, DoesNotMatchFunctionDeclarations) {
  EXPECT_TRUE(
      notMatches("void f(double d); void f(int t) {}",
      functionTemplateDecl(hasName("f"))));
}

TEST(FunctionTemplate, DoesNotMatchFunctionTemplateSpecializations) {
  EXPECT_TRUE(
      notMatches("void g(); template <typename T> void f(T t) {}"
                 "template <> void f(int t) { g(); }",
      functionTemplateDecl(hasName("f"),
                           hasDescendant(declRefExpr(to(
                               functionDecl(hasName("g"))))))));
}

TEST(Matcher, Argument) {
  StatementMatcher CallArgumentY = callExpr(
      hasArgument(0, declRefExpr(to(varDecl(hasName("y"))))));

  EXPECT_TRUE(matches("void x(int) { int y; x(y); }", CallArgumentY));
  EXPECT_TRUE(
      matches("class X { void x(int) { int y; x(y); } };", CallArgumentY));
  EXPECT_TRUE(notMatches("void x(int) { int z; x(z); }", CallArgumentY));

  StatementMatcher WrongIndex = callExpr(
      hasArgument(42, declRefExpr(to(varDecl(hasName("y"))))));
  EXPECT_TRUE(notMatches("void x(int) { int y; x(y); }", WrongIndex));
}

TEST(Matcher, AnyArgument) {
  StatementMatcher CallArgumentY = callExpr(
      hasAnyArgument(declRefExpr(to(varDecl(hasName("y"))))));
  EXPECT_TRUE(matches("void x(int, int) { int y; x(1, y); }", CallArgumentY));
  EXPECT_TRUE(matches("void x(int, int) { int y; x(y, 42); }", CallArgumentY));
  EXPECT_TRUE(notMatches("void x(int, int) { x(1, 2); }", CallArgumentY));
}

TEST(ForEachArgumentWithParam, ReportsNoFalsePositives) {
  StatementMatcher ArgumentY =
      declRefExpr(to(varDecl(hasName("y")))).bind("arg");
  DeclarationMatcher IntParam = parmVarDecl(hasType(isInteger())).bind("param");
  StatementMatcher CallExpr =
      callExpr(forEachArgumentWithParam(ArgumentY, IntParam));

  // IntParam does not match.
  EXPECT_TRUE(notMatches("void f(int* i) { int* y; f(y); }", CallExpr));
  // ArgumentY does not match.
  EXPECT_TRUE(notMatches("void f(int i) { int x; f(x); }", CallExpr));
}

TEST(ForEachArgumentWithParam, MatchesCXXMemberCallExpr) {
  StatementMatcher ArgumentY =
      declRefExpr(to(varDecl(hasName("y")))).bind("arg");
  DeclarationMatcher IntParam = parmVarDecl(hasType(isInteger())).bind("param");
  StatementMatcher CallExpr =
      callExpr(forEachArgumentWithParam(ArgumentY, IntParam));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "struct S {"
      "  const S& operator[](int i) { return *this; }"
      "};"
      "void f(S S1) {"
      "  int y = 1;"
      "  S1[y];"
      "}",
      CallExpr, new VerifyIdIsBoundTo<ParmVarDecl>("param", 1)));

  StatementMatcher CallExpr2 =
      callExpr(forEachArgumentWithParam(ArgumentY, IntParam));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "struct S {"
      "  static void g(int i);"
      "};"
      "void f() {"
      "  int y = 1;"
      "  S::g(y);"
      "}",
      CallExpr2, new VerifyIdIsBoundTo<ParmVarDecl>("param", 1)));
}

TEST(ForEachArgumentWithParam, MatchesCallExpr) {
  StatementMatcher ArgumentY =
      declRefExpr(to(varDecl(hasName("y")))).bind("arg");
  DeclarationMatcher IntParam = parmVarDecl(hasType(isInteger())).bind("param");
  StatementMatcher CallExpr =
      callExpr(forEachArgumentWithParam(ArgumentY, IntParam));

  EXPECT_TRUE(
      matchAndVerifyResultTrue("void f(int i) { int y; f(y); }", CallExpr,
                               new VerifyIdIsBoundTo<ParmVarDecl>("param")));
  EXPECT_TRUE(
      matchAndVerifyResultTrue("void f(int i) { int y; f(y); }", CallExpr,
                               new VerifyIdIsBoundTo<DeclRefExpr>("arg")));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f(int i, int j) { int y; f(y, y); }", CallExpr,
      new VerifyIdIsBoundTo<ParmVarDecl>("param", 2)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f(int i, int j) { int y; f(y, y); }", CallExpr,
      new VerifyIdIsBoundTo<DeclRefExpr>("arg", 2)));
}

TEST(ForEachArgumentWithParam, MatchesConstructExpr) {
  StatementMatcher ArgumentY =
      declRefExpr(to(varDecl(hasName("y")))).bind("arg");
  DeclarationMatcher IntParam = parmVarDecl(hasType(isInteger())).bind("param");
  StatementMatcher ConstructExpr =
      cxxConstructExpr(forEachArgumentWithParam(ArgumentY, IntParam));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "struct C {"
      "  C(int i) {}"
      "};"
      "int y = 0;"
      "C Obj(y);",
      ConstructExpr, new VerifyIdIsBoundTo<ParmVarDecl>("param")));
}

TEST(ForEachArgumentWithParam, HandlesBoundNodesForNonMatches) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void g(int i, int j) {"
      "  int a;"
      "  int b;"
      "  int c;"
      "  g(a, 0);"
      "  g(a, b);"
      "  g(0, b);"
      "}",
      functionDecl(
          forEachDescendant(varDecl().bind("v")),
          forEachDescendant(callExpr(forEachArgumentWithParam(
              declRefExpr(to(decl(equalsBoundNode("v")))), parmVarDecl())))),
      new VerifyIdIsBoundTo<VarDecl>("v", 4)));
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

TEST(QualType, hasCanonicalType) {
  EXPECT_TRUE(notMatches("typedef int &int_ref;"
                         "int a;"
                         "int_ref b = a;",
                         varDecl(hasType(qualType(referenceType())))));
  EXPECT_TRUE(
      matches("typedef int &int_ref;"
              "int a;"
              "int_ref b = a;",
              varDecl(hasType(qualType(hasCanonicalType(referenceType()))))));
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

TEST(HasParameter, CallsInnerMatcher) {
  EXPECT_TRUE(matches("class X { void x(int) {} };",
                      cxxMethodDecl(hasParameter(0, varDecl()))));
  EXPECT_TRUE(notMatches("class X { void x(int) {} };",
                         cxxMethodDecl(hasParameter(0, hasName("x")))));
}

TEST(HasParameter, DoesNotMatchIfIndexOutOfBounds) {
  EXPECT_TRUE(notMatches("class X { void x(int) {} };",
                         cxxMethodDecl(hasParameter(42, varDecl()))));
}

TEST(HasType, MatchesParameterVariableTypesStrictly) {
  EXPECT_TRUE(matches(
      "class X { void x(X x) {} };",
      cxxMethodDecl(hasParameter(0, hasType(recordDecl(hasName("X")))))));
  EXPECT_TRUE(notMatches(
      "class X { void x(const X &x) {} };",
      cxxMethodDecl(hasParameter(0, hasType(recordDecl(hasName("X")))))));
  EXPECT_TRUE(matches("class X { void x(const X *x) {} };",
                      cxxMethodDecl(hasParameter(
                          0, hasType(pointsTo(recordDecl(hasName("X"))))))));
  EXPECT_TRUE(matches("class X { void x(const X &x) {} };",
                      cxxMethodDecl(hasParameter(
                          0, hasType(references(recordDecl(hasName("X"))))))));
}

TEST(HasAnyParameter, MatchesIndependentlyOfPosition) {
  EXPECT_TRUE(matches(
      "class Y {}; class X { void x(X x, Y y) {} };",
      cxxMethodDecl(hasAnyParameter(hasType(recordDecl(hasName("X")))))));
  EXPECT_TRUE(matches(
      "class Y {}; class X { void x(Y y, X x) {} };",
      cxxMethodDecl(hasAnyParameter(hasType(recordDecl(hasName("X")))))));
}

TEST(Returns, MatchesReturnTypes) {
  EXPECT_TRUE(matches("class Y { int f() { return 1; } };",
                      functionDecl(returns(asString("int")))));
  EXPECT_TRUE(notMatches("class Y { int f() { return 1; } };",
                         functionDecl(returns(asString("float")))));
  EXPECT_TRUE(matches("class Y { Y getMe() { return *this; } };",
                      functionDecl(returns(hasDeclaration(
                          recordDecl(hasName("Y")))))));
}

TEST(IsExternC, MatchesExternCFunctionDeclarations) {
  EXPECT_TRUE(matches("extern \"C\" void f() {}", functionDecl(isExternC())));
  EXPECT_TRUE(matches("extern \"C\" { void f() {} }",
              functionDecl(isExternC())));
  EXPECT_TRUE(notMatches("void f() {}", functionDecl(isExternC())));
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
}

TEST(isConstexpr, MatchesConstexprDeclarations) {
  EXPECT_TRUE(matches("constexpr int foo = 42;",
                      varDecl(hasName("foo"), isConstexpr())));
  EXPECT_TRUE(matches("constexpr int bar();",
                      functionDecl(hasName("bar"), isConstexpr())));
}

TEST(HasAnyParameter, DoesntMatchIfInnerMatcherDoesntMatch) {
  EXPECT_TRUE(notMatches(
      "class Y {}; class X { void x(int) {} };",
      cxxMethodDecl(hasAnyParameter(hasType(recordDecl(hasName("X")))))));
}

TEST(HasAnyParameter, DoesNotMatchThisPointer) {
  EXPECT_TRUE(notMatches("class Y {}; class X { void x() {} };",
                         cxxMethodDecl(hasAnyParameter(
                             hasType(pointsTo(recordDecl(hasName("X"))))))));
}

TEST(HasName, MatchesParameterVariableDeclarations) {
  EXPECT_TRUE(matches("class Y {}; class X { void x(int x) {} };",
                      cxxMethodDecl(hasAnyParameter(hasName("x")))));
  EXPECT_TRUE(notMatches("class Y {}; class X { void x(int) {} };",
                         cxxMethodDecl(hasAnyParameter(hasName("x")))));
}

TEST(Matcher, MatchesClassTemplateSpecialization) {
  EXPECT_TRUE(matches("template<typename T> struct A {};"
                      "template<> struct A<int> {};",
                      classTemplateSpecializationDecl()));
  EXPECT_TRUE(matches("template<typename T> struct A {}; A<int> a;",
                      classTemplateSpecializationDecl()));
  EXPECT_TRUE(notMatches("template<typename T> struct A {};",
                         classTemplateSpecializationDecl()));
}

TEST(DeclaratorDecl, MatchesDeclaratorDecls) {
  EXPECT_TRUE(matches("int x;", declaratorDecl()));
  EXPECT_TRUE(notMatches("class A {};", declaratorDecl()));
}

TEST(ParmVarDecl, MatchesParmVars) {
  EXPECT_TRUE(matches("void f(int x);", parmVarDecl()));
  EXPECT_TRUE(notMatches("void f();", parmVarDecl()));
}

TEST(Matcher, MatchesTypeTemplateArgument) {
  EXPECT_TRUE(matches(
      "template<typename T> struct B {};"
      "B<int> b;",
      classTemplateSpecializationDecl(hasAnyTemplateArgument(refersToType(
          asString("int"))))));
}

TEST(Matcher, MatchesDeclarationReferenceTemplateArgument) {
  EXPECT_TRUE(matches(
      "struct B { int next; };"
      "template<int(B::*next_ptr)> struct A {};"
      "A<&B::next> a;",
      classTemplateSpecializationDecl(hasAnyTemplateArgument(
          refersToDeclaration(fieldDecl(hasName("next")))))));

  EXPECT_TRUE(notMatches(
      "template <typename T> struct A {};"
      "A<int> a;",
      classTemplateSpecializationDecl(hasAnyTemplateArgument(
          refersToDeclaration(decl())))));

  EXPECT_TRUE(matches(
      "struct B { int next; };"
      "template<int(B::*next_ptr)> struct A {};"
      "A<&B::next> a;",
      templateSpecializationType(hasAnyTemplateArgument(isExpr(
          hasDescendant(declRefExpr(to(fieldDecl(hasName("next"))))))))));

  EXPECT_TRUE(notMatches(
      "template <typename T> struct A {};"
      "A<int> a;",
      templateSpecializationType(hasAnyTemplateArgument(
          refersToDeclaration(decl())))));
}

TEST(Matcher, MatchesSpecificArgument) {
  EXPECT_TRUE(matches(
      "template<typename T, typename U> class A {};"
      "A<bool, int> a;",
      classTemplateSpecializationDecl(hasTemplateArgument(
          1, refersToType(asString("int"))))));
  EXPECT_TRUE(notMatches(
      "template<typename T, typename U> class A {};"
      "A<int, bool> a;",
      classTemplateSpecializationDecl(hasTemplateArgument(
          1, refersToType(asString("int"))))));

  EXPECT_TRUE(matches(
      "template<typename T, typename U> class A {};"
      "A<bool, int> a;",
      templateSpecializationType(hasTemplateArgument(
          1, refersToType(asString("int"))))));
  EXPECT_TRUE(notMatches(
      "template<typename T, typename U> class A {};"
      "A<int, bool> a;",
      templateSpecializationType(hasTemplateArgument(
          1, refersToType(asString("int"))))));
}

TEST(TemplateArgument, Matches) {
  EXPECT_TRUE(matches("template<typename T> struct C {}; C<int> c;",
                      classTemplateSpecializationDecl(
                          hasAnyTemplateArgument(templateArgument()))));
  EXPECT_TRUE(matches(
      "template<typename T> struct C {}; C<int> c;",
      templateSpecializationType(hasAnyTemplateArgument(templateArgument()))));
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

TEST(RefersToIntegralType, Matches) {
  EXPECT_TRUE(matches("template<int T> struct C {}; C<42> c;",
                      classTemplateSpecializationDecl(
                          hasAnyTemplateArgument(refersToIntegralType(
                              asString("int"))))));
  EXPECT_TRUE(notMatches("template<unsigned T> struct C {}; C<42> c;",
                         classTemplateSpecializationDecl(hasAnyTemplateArgument(
                             refersToIntegralType(asString("int"))))));
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

TEST(Matcher, ConstructorCall) {
  StatementMatcher Constructor = cxxConstructExpr();

  EXPECT_TRUE(
      matches("class X { public: X(); }; void x() { X x; }", Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(); }; void x() { X x = X(); }",
              Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { X x = 0; }",
              Constructor));
  EXPECT_TRUE(matches("class X {}; void x(int) { X x; }", Constructor));
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

TEST(Matcher,ThisExpr) {
  EXPECT_TRUE(
      matches("struct X { int a; int f () { return a; } };", cxxThisExpr()));
  EXPECT_TRUE(
      notMatches("struct X { int f () { int a; return a; } };", cxxThisExpr()));
}

TEST(Matcher, BindTemporaryExpression) {
  StatementMatcher TempExpression = cxxBindTemporaryExpr();

  std::string ClassString = "class string { public: string(); ~string(); }; ";

  EXPECT_TRUE(
      matches(ClassString +
              "string GetStringByValue();"
              "void FunctionTakesString(string s);"
              "void run() { FunctionTakesString(GetStringByValue()); }",
              TempExpression));

  EXPECT_TRUE(
      notMatches(ClassString +
                 "string* GetStringPointer(); "
                 "void FunctionTakesStringPtr(string* s);"
                 "void run() {"
                 "  string* s = GetStringPointer();"
                 "  FunctionTakesStringPtr(GetStringPointer());"
                 "  FunctionTakesStringPtr(s);"
                 "}",
                 TempExpression));

  EXPECT_TRUE(
      notMatches("class no_dtor {};"
                 "no_dtor GetObjByValue();"
                 "void ConsumeObj(no_dtor param);"
                 "void run() { ConsumeObj(GetObjByValue()); }",
                 TempExpression));
}

TEST(MaterializeTemporaryExpr, MatchesTemporary) {
  std::string ClassString =
      "class string { public: string(); int length(); }; ";

  EXPECT_TRUE(
      matches(ClassString +
              "string GetStringByValue();"
              "void FunctionTakesString(string s);"
              "void run() { FunctionTakesString(GetStringByValue()); }",
              materializeTemporaryExpr()));

  EXPECT_TRUE(
      notMatches(ClassString +
                 "string* GetStringPointer(); "
                 "void FunctionTakesStringPtr(string* s);"
                 "void run() {"
                 "  string* s = GetStringPointer();"
                 "  FunctionTakesStringPtr(GetStringPointer());"
                 "  FunctionTakesStringPtr(s);"
                 "}",
                 materializeTemporaryExpr()));

  EXPECT_TRUE(
      notMatches(ClassString +
                 "string GetStringByValue();"
                 "void run() { int k = GetStringByValue().length(); }",
                 materializeTemporaryExpr()));

  EXPECT_TRUE(
      notMatches(ClassString +
                 "string GetStringByValue();"
                 "void run() { GetStringByValue(); }",
                 materializeTemporaryExpr()));
}

TEST(ConstructorDeclaration, SimpleCase) {
  EXPECT_TRUE(matches("class Foo { Foo(int i); };",
                      cxxConstructorDecl(ofClass(hasName("Foo")))));
  EXPECT_TRUE(notMatches("class Foo { Foo(int i); };",
                         cxxConstructorDecl(ofClass(hasName("Bar")))));
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
  EXPECT_TRUE(matches("struct S { S(); };",
                      cxxConstructorDecl(isDefaultConstructor())));
  EXPECT_TRUE(notMatches("struct S { S(); };",
                         cxxConstructorDecl(isCopyConstructor())));
  EXPECT_TRUE(notMatches("struct S { S(); };",
                         cxxConstructorDecl(isMoveConstructor())));

  EXPECT_TRUE(notMatches("struct S { S(const S&); };",
                         cxxConstructorDecl(isDefaultConstructor())));
  EXPECT_TRUE(matches("struct S { S(const S&); };",
                      cxxConstructorDecl(isCopyConstructor())));
  EXPECT_TRUE(notMatches("struct S { S(const S&); };",
                         cxxConstructorDecl(isMoveConstructor())));

  EXPECT_TRUE(notMatches("struct S { S(S&&); };",
                         cxxConstructorDecl(isDefaultConstructor())));
  EXPECT_TRUE(notMatches("struct S { S(S&&); };",
                         cxxConstructorDecl(isCopyConstructor())));
  EXPECT_TRUE(matches("struct S { S(S&&); };",
                      cxxConstructorDecl(isMoveConstructor())));
}

TEST(DestructorDeclaration, MatchesVirtualDestructor) {
  EXPECT_TRUE(matches("class Foo { virtual ~Foo(); };",
                      cxxDestructorDecl(ofClass(hasName("Foo")))));
}

TEST(DestructorDeclaration, DoesNotMatchImplicitDestructor) {
  EXPECT_TRUE(notMatches("class Foo {};",
                         cxxDestructorDecl(ofClass(hasName("Foo")))));
}

TEST(HasAnyConstructorInitializer, SimpleCase) {
  EXPECT_TRUE(
      notMatches("class Foo { Foo() { } };",
                 cxxConstructorDecl(hasAnyConstructorInitializer(anything()))));
  EXPECT_TRUE(
      matches("class Foo {"
              "  Foo() : foo_() { }"
              "  int foo_;"
              "};",
              cxxConstructorDecl(hasAnyConstructorInitializer(anything()))));
}

TEST(HasAnyConstructorInitializer, ForField) {
  static const char Code[] =
      "class Baz { };"
      "class Foo {"
      "  Foo() : foo_() { }"
      "  Baz foo_;"
      "  Baz bar_;"
      "};";
  EXPECT_TRUE(matches(Code, cxxConstructorDecl(hasAnyConstructorInitializer(
      forField(hasType(recordDecl(hasName("Baz"))))))));
  EXPECT_TRUE(matches(Code, cxxConstructorDecl(hasAnyConstructorInitializer(
      forField(hasName("foo_"))))));
  EXPECT_TRUE(notMatches(Code, cxxConstructorDecl(hasAnyConstructorInitializer(
      forField(hasType(recordDecl(hasName("Bar"))))))));
}

TEST(HasAnyConstructorInitializer, WithInitializer) {
  static const char Code[] =
      "class Foo {"
      "  Foo() : foo_(0) { }"
      "  int foo_;"
      "};";
  EXPECT_TRUE(matches(Code, cxxConstructorDecl(hasAnyConstructorInitializer(
      withInitializer(integerLiteral(equals(0)))))));
  EXPECT_TRUE(notMatches(Code, cxxConstructorDecl(hasAnyConstructorInitializer(
      withInitializer(integerLiteral(equals(1)))))));
}

TEST(HasAnyConstructorInitializer, IsWritten) {
  static const char Code[] =
      "struct Bar { Bar(){} };"
      "class Foo {"
      "  Foo() : foo_() { }"
      "  Bar foo_;"
      "  Bar bar_;"
      "};";
  EXPECT_TRUE(matches(Code, cxxConstructorDecl(hasAnyConstructorInitializer(
      allOf(forField(hasName("foo_")), isWritten())))));
  EXPECT_TRUE(notMatches(Code, cxxConstructorDecl(hasAnyConstructorInitializer(
      allOf(forField(hasName("bar_")), isWritten())))));
  EXPECT_TRUE(matches(Code, cxxConstructorDecl(hasAnyConstructorInitializer(
      allOf(forField(hasName("bar_")), unless(isWritten()))))));
}

TEST(HasAnyConstructorInitializer, IsBaseInitializer) {
  static const char Code[] =
      "struct B {};"
      "struct D : B {"
      "  int I;"
      "  D(int i) : I(i) {}"
      "};"
      "struct E : B {"
      "  E() : B() {}"
      "};";
  EXPECT_TRUE(matches(Code, cxxConstructorDecl(allOf(
    hasAnyConstructorInitializer(allOf(isBaseInitializer(), isWritten())),
    hasName("E")))));
  EXPECT_TRUE(notMatches(Code, cxxConstructorDecl(allOf(
    hasAnyConstructorInitializer(allOf(isBaseInitializer(), isWritten())),
    hasName("D")))));
  EXPECT_TRUE(matches(Code, cxxConstructorDecl(allOf(
    hasAnyConstructorInitializer(allOf(isMemberInitializer(), isWritten())),
    hasName("D")))));
  EXPECT_TRUE(notMatches(Code, cxxConstructorDecl(allOf(
    hasAnyConstructorInitializer(allOf(isMemberInitializer(), isWritten())),
    hasName("E")))));
}

TEST(Matcher, NewExpression) {
  StatementMatcher New = cxxNewExpr();

  EXPECT_TRUE(matches("class X { public: X(); }; void x() { new X; }", New));
  EXPECT_TRUE(
      matches("class X { public: X(); }; void x() { new X(); }", New));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { new X(0); }", New));
  EXPECT_TRUE(matches("class X {}; void x(int) { new X; }", New));
}

TEST(Matcher, NewExpressionArgument) {
  StatementMatcher New = cxxConstructExpr(
      hasArgument(0, declRefExpr(to(varDecl(hasName("y"))))));

  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; new X(y); }",
              New));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { int y; new X(y); }",
              New));
  EXPECT_TRUE(
      notMatches("class X { public: X(int); }; void x() { int z; new X(z); }",
                 New));

  StatementMatcher WrongIndex = cxxConstructExpr(
      hasArgument(42, declRefExpr(to(varDecl(hasName("y"))))));
  EXPECT_TRUE(
      notMatches("class X { public: X(int); }; void x() { int y; new X(y); }",
                 WrongIndex));
}

TEST(Matcher, NewExpressionArgumentCount) {
  StatementMatcher New = cxxConstructExpr(argumentCountIs(1));

  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { new X(0); }", New));
  EXPECT_TRUE(
      notMatches("class X { public: X(int, int); }; void x() { new X(0, 0); }",
                 New));
}

TEST(Matcher, DeleteExpression) {
  EXPECT_TRUE(matches("struct A {}; void f(A* a) { delete a; }",
                      cxxDeleteExpr()));
}

TEST(Matcher, DefaultArgument) {
  StatementMatcher Arg = cxxDefaultArgExpr();

  EXPECT_TRUE(matches("void x(int, int = 0) { int y; x(y); }", Arg));
  EXPECT_TRUE(
      matches("class X { void x(int, int = 0) { int y; x(y); } };", Arg));
  EXPECT_TRUE(notMatches("void x(int, int = 0) { int y; x(y, 0); }", Arg));
}

TEST(Matcher, StringLiterals) {
  StatementMatcher Literal = stringLiteral();
  EXPECT_TRUE(matches("const char *s = \"string\";", Literal));
  // wide string
  EXPECT_TRUE(matches("const wchar_t *s = L\"string\";", Literal));
  // with escaped characters
  EXPECT_TRUE(matches("const char *s = \"\x05five\";", Literal));
  // no matching -- though the data type is the same, there is no string literal
  EXPECT_TRUE(notMatches("const char s[1] = {'a'};", Literal));
}

TEST(Matcher, CharacterLiterals) {
  StatementMatcher CharLiteral = characterLiteral();
  EXPECT_TRUE(matches("const char c = 'c';", CharLiteral));
  // wide character
  EXPECT_TRUE(matches("const char c = L'c';", CharLiteral));
  // wide character, Hex encoded, NOT MATCHED!
  EXPECT_TRUE(notMatches("const wchar_t c = 0x2126;", CharLiteral));
  EXPECT_TRUE(notMatches("const char c = 0x1;", CharLiteral));
}

TEST(Matcher, IntegerLiterals) {
  StatementMatcher HasIntLiteral = integerLiteral();
  EXPECT_TRUE(matches("int i = 10;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 0x1AB;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 10L;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 10U;", HasIntLiteral));

  // Non-matching cases (character literals, float and double)
  EXPECT_TRUE(notMatches("int i = L'a';",
                HasIntLiteral));  // this is actually a character
                                  // literal cast to int
  EXPECT_TRUE(notMatches("int i = 'a';", HasIntLiteral));
  EXPECT_TRUE(notMatches("int i = 1e10;", HasIntLiteral));
  EXPECT_TRUE(notMatches("int i = 10.0;", HasIntLiteral));
}

TEST(Matcher, FloatLiterals) {
  StatementMatcher HasFloatLiteral = floatLiteral();
  EXPECT_TRUE(matches("float i = 10.0;", HasFloatLiteral));
  EXPECT_TRUE(matches("float i = 10.0f;", HasFloatLiteral));
  EXPECT_TRUE(matches("double i = 10.0;", HasFloatLiteral));
  EXPECT_TRUE(matches("double i = 10.0L;", HasFloatLiteral));
  EXPECT_TRUE(matches("double i = 1e10;", HasFloatLiteral));
  EXPECT_TRUE(matches("double i = 5.0;", floatLiteral(equals(5.0))));
  EXPECT_TRUE(matches("double i = 5.0;", floatLiteral(equals(5.0f))));
  EXPECT_TRUE(
      matches("double i = 5.0;", floatLiteral(equals(llvm::APFloat(5.0)))));

  EXPECT_TRUE(notMatches("float i = 10;", HasFloatLiteral));
  EXPECT_TRUE(notMatches("double i = 5.0;", floatLiteral(equals(6.0))));
  EXPECT_TRUE(notMatches("double i = 5.0;", floatLiteral(equals(6.0f))));
  EXPECT_TRUE(
      notMatches("double i = 5.0;", floatLiteral(equals(llvm::APFloat(6.0)))));
}

TEST(Matcher, NullPtrLiteral) {
  EXPECT_TRUE(matches("int* i = nullptr;", cxxNullPtrLiteralExpr()));
}

TEST(Matcher, GNUNullExpr) {
  EXPECT_TRUE(matches("int* i = __null;", gnuNullExpr()));
}

TEST(Matcher, AsmStatement) {
  EXPECT_TRUE(matches("void foo() { __asm(\"mov al, 2\"); }", asmStmt()));
}

TEST(Matcher, Conditions) {
  StatementMatcher Condition =
      ifStmt(hasCondition(cxxBoolLiteral(equals(true))));

  EXPECT_TRUE(matches("void x() { if (true) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (false) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { bool a = true; if (a) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (true || false) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (1) {} }", Condition));
}

TEST(IfStmt, ChildTraversalMatchers) {
  EXPECT_TRUE(matches("void f() { if (false) true; else false; }",
                      ifStmt(hasThen(cxxBoolLiteral(equals(true))))));
  EXPECT_TRUE(notMatches("void f() { if (false) false; else true; }",
                         ifStmt(hasThen(cxxBoolLiteral(equals(true))))));
  EXPECT_TRUE(matches("void f() { if (false) false; else true; }",
                      ifStmt(hasElse(cxxBoolLiteral(equals(true))))));
  EXPECT_TRUE(notMatches("void f() { if (false) true; else false; }",
                         ifStmt(hasElse(cxxBoolLiteral(equals(true))))));
}

TEST(MatchBinaryOperator, HasOperatorName) {
  StatementMatcher OperatorOr = binaryOperator(hasOperatorName("||"));

  EXPECT_TRUE(matches("void x() { true || false; }", OperatorOr));
  EXPECT_TRUE(notMatches("void x() { true && false; }", OperatorOr));
}

TEST(MatchBinaryOperator, HasLHSAndHasRHS) {
  StatementMatcher OperatorTrueFalse =
      binaryOperator(hasLHS(cxxBoolLiteral(equals(true))),
                     hasRHS(cxxBoolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true || false; }", OperatorTrueFalse));
  EXPECT_TRUE(matches("void x() { true && false; }", OperatorTrueFalse));
  EXPECT_TRUE(notMatches("void x() { false || true; }", OperatorTrueFalse));

  StatementMatcher OperatorIntPointer = arraySubscriptExpr(
      hasLHS(hasType(isInteger())), hasRHS(hasType(pointsTo(qualType()))));
  EXPECT_TRUE(matches("void x() { 1[\"abc\"]; }", OperatorIntPointer));
  EXPECT_TRUE(notMatches("void x() { \"abc\"[1]; }", OperatorIntPointer));
}

TEST(MatchBinaryOperator, HasEitherOperand) {
  StatementMatcher HasOperand =
      binaryOperator(hasEitherOperand(cxxBoolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true || false; }", HasOperand));
  EXPECT_TRUE(matches("void x() { false && true; }", HasOperand));
  EXPECT_TRUE(notMatches("void x() { true || true; }", HasOperand));
}

TEST(Matcher, BinaryOperatorTypes) {
  // Integration test that verifies the AST provides all binary operators in
  // a way we expect.
  // FIXME: Operator ','
  EXPECT_TRUE(
      matches("void x() { 3, 4; }", binaryOperator(hasOperatorName(","))));
  EXPECT_TRUE(
      matches("bool b; bool c = (b = true);",
              binaryOperator(hasOperatorName("="))));
  EXPECT_TRUE(
      matches("bool b = 1 != 2;", binaryOperator(hasOperatorName("!="))));
  EXPECT_TRUE(
      matches("bool b = 1 == 2;", binaryOperator(hasOperatorName("=="))));
  EXPECT_TRUE(matches("bool b = 1 < 2;", binaryOperator(hasOperatorName("<"))));
  EXPECT_TRUE(
      matches("bool b = 1 <= 2;", binaryOperator(hasOperatorName("<="))));
  EXPECT_TRUE(
      matches("int i = 1 << 2;", binaryOperator(hasOperatorName("<<"))));
  EXPECT_TRUE(
      matches("int i = 1; int j = (i <<= 2);",
              binaryOperator(hasOperatorName("<<="))));
  EXPECT_TRUE(matches("bool b = 1 > 2;", binaryOperator(hasOperatorName(">"))));
  EXPECT_TRUE(
      matches("bool b = 1 >= 2;", binaryOperator(hasOperatorName(">="))));
  EXPECT_TRUE(
      matches("int i = 1 >> 2;", binaryOperator(hasOperatorName(">>"))));
  EXPECT_TRUE(
      matches("int i = 1; int j = (i >>= 2);",
              binaryOperator(hasOperatorName(">>="))));
  EXPECT_TRUE(
      matches("int i = 42 ^ 23;", binaryOperator(hasOperatorName("^"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i ^= 42);",
              binaryOperator(hasOperatorName("^="))));
  EXPECT_TRUE(
      matches("int i = 42 % 23;", binaryOperator(hasOperatorName("%"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i %= 42);",
              binaryOperator(hasOperatorName("%="))));
  EXPECT_TRUE(
      matches("bool b = 42  &23;", binaryOperator(hasOperatorName("&"))));
  EXPECT_TRUE(
      matches("bool b = true && false;",
              binaryOperator(hasOperatorName("&&"))));
  EXPECT_TRUE(
      matches("bool b = true; bool c = (b &= false);",
              binaryOperator(hasOperatorName("&="))));
  EXPECT_TRUE(
      matches("bool b = 42 | 23;", binaryOperator(hasOperatorName("|"))));
  EXPECT_TRUE(
      matches("bool b = true || false;",
              binaryOperator(hasOperatorName("||"))));
  EXPECT_TRUE(
      matches("bool b = true; bool c = (b |= false);",
              binaryOperator(hasOperatorName("|="))));
  EXPECT_TRUE(
      matches("int i = 42  *23;", binaryOperator(hasOperatorName("*"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i *= 23);",
              binaryOperator(hasOperatorName("*="))));
  EXPECT_TRUE(
      matches("int i = 42 / 23;", binaryOperator(hasOperatorName("/"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i /= 23);",
              binaryOperator(hasOperatorName("/="))));
  EXPECT_TRUE(
      matches("int i = 42 + 23;", binaryOperator(hasOperatorName("+"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i += 23);",
              binaryOperator(hasOperatorName("+="))));
  EXPECT_TRUE(
      matches("int i = 42 - 23;", binaryOperator(hasOperatorName("-"))));
  EXPECT_TRUE(
      matches("int i = 42; int j = (i -= 23);",
              binaryOperator(hasOperatorName("-="))));
  EXPECT_TRUE(
      matches("struct A { void x() { void (A::*a)(); (this->*a)(); } };",
              binaryOperator(hasOperatorName("->*"))));
  EXPECT_TRUE(
      matches("struct A { void x() { void (A::*a)(); ((*this).*a)(); } };",
              binaryOperator(hasOperatorName(".*"))));

  // Member expressions as operators are not supported in matches.
  EXPECT_TRUE(
      notMatches("struct A { void x(A *a) { a->x(this); } };",
                 binaryOperator(hasOperatorName("->"))));

  // Initializer assignments are not represented as operator equals.
  EXPECT_TRUE(
      notMatches("bool b = true;", binaryOperator(hasOperatorName("="))));

  // Array indexing is not represented as operator.
  EXPECT_TRUE(notMatches("int a[42]; void x() { a[23]; }", unaryOperator()));

  // Overloaded operators do not match at all.
  EXPECT_TRUE(notMatches(
      "struct A { bool operator&&(const A &a) const { return false; } };"
      "void x() { A a, b; a && b; }",
      binaryOperator()));
}

TEST(MatchUnaryOperator, HasOperatorName) {
  StatementMatcher OperatorNot = unaryOperator(hasOperatorName("!"));

  EXPECT_TRUE(matches("void x() { !true; } ", OperatorNot));
  EXPECT_TRUE(notMatches("void x() { true; } ", OperatorNot));
}

TEST(MatchUnaryOperator, HasUnaryOperand) {
  StatementMatcher OperatorOnFalse =
      unaryOperator(hasUnaryOperand(cxxBoolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { !false; }", OperatorOnFalse));
  EXPECT_TRUE(notMatches("void x() { !true; }", OperatorOnFalse));
}

TEST(Matcher, UnaryOperatorTypes) {
  // Integration test that verifies the AST provides all unary operators in
  // a way we expect.
  EXPECT_TRUE(matches("bool b = !true;", unaryOperator(hasOperatorName("!"))));
  EXPECT_TRUE(
      matches("bool b; bool *p = &b;", unaryOperator(hasOperatorName("&"))));
  EXPECT_TRUE(matches("int i = ~ 1;", unaryOperator(hasOperatorName("~"))));
  EXPECT_TRUE(
      matches("bool *p; bool b = *p;", unaryOperator(hasOperatorName("*"))));
  EXPECT_TRUE(
      matches("int i; int j = +i;", unaryOperator(hasOperatorName("+"))));
  EXPECT_TRUE(
      matches("int i; int j = -i;", unaryOperator(hasOperatorName("-"))));
  EXPECT_TRUE(
      matches("int i; int j = ++i;", unaryOperator(hasOperatorName("++"))));
  EXPECT_TRUE(
      matches("int i; int j = i++;", unaryOperator(hasOperatorName("++"))));
  EXPECT_TRUE(
      matches("int i; int j = --i;", unaryOperator(hasOperatorName("--"))));
  EXPECT_TRUE(
      matches("int i; int j = i--;", unaryOperator(hasOperatorName("--"))));

  // We don't match conversion operators.
  EXPECT_TRUE(notMatches("int i; double d = (double)i;", unaryOperator()));

  // Function calls are not represented as operator.
  EXPECT_TRUE(notMatches("void f(); void x() { f(); }", unaryOperator()));

  // Overloaded operators do not match at all.
  // FIXME: We probably want to add that.
  EXPECT_TRUE(notMatches(
      "struct A { bool operator!() const { return false; } };"
      "void x() { A a; !a; }", unaryOperator(hasOperatorName("!"))));
}

TEST(Matcher, ConditionalOperator) {
  StatementMatcher Conditional = conditionalOperator(
      hasCondition(cxxBoolLiteral(equals(true))),
      hasTrueExpression(cxxBoolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true ? false : true; }", Conditional));
  EXPECT_TRUE(notMatches("void x() { false ? false : true; }", Conditional));
  EXPECT_TRUE(notMatches("void x() { true ? true : false; }", Conditional));

  StatementMatcher ConditionalFalse = conditionalOperator(
      hasFalseExpression(cxxBoolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true ? true : false; }", ConditionalFalse));
  EXPECT_TRUE(
      notMatches("void x() { true ? false : true; }", ConditionalFalse));
}

TEST(ArraySubscriptMatchers, ArraySubscripts) {
  EXPECT_TRUE(matches("int i[2]; void f() { i[1] = 1; }",
                      arraySubscriptExpr()));
  EXPECT_TRUE(notMatches("int i; void f() { i = 1; }",
                         arraySubscriptExpr()));
}

TEST(ArraySubscriptMatchers, ArrayIndex) {
  EXPECT_TRUE(matches(
      "int i[2]; void f() { i[1] = 1; }",
      arraySubscriptExpr(hasIndex(integerLiteral(equals(1))))));
  EXPECT_TRUE(matches(
      "int i[2]; void f() { 1[i] = 1; }",
      arraySubscriptExpr(hasIndex(integerLiteral(equals(1))))));
  EXPECT_TRUE(notMatches(
      "int i[2]; void f() { i[1] = 1; }",
      arraySubscriptExpr(hasIndex(integerLiteral(equals(0))))));
}

TEST(ArraySubscriptMatchers, MatchesArrayBase) {
  EXPECT_TRUE(matches(
      "int i[2]; void f() { i[1] = 2; }",
      arraySubscriptExpr(hasBase(implicitCastExpr(
          hasSourceExpression(declRefExpr()))))));
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
}

TEST(Matcher, OfClass) {
  StatementMatcher Constructor = cxxConstructExpr(hasDeclaration(cxxMethodDecl(
      ofClass(hasName("X")))));

  EXPECT_TRUE(
      matches("class X { public: X(); }; void x(int) { X x; }", Constructor));
  EXPECT_TRUE(
      matches("class X { public: X(); }; void x(int) { X x = X(); }",
              Constructor));
  EXPECT_TRUE(
      notMatches("class Y { public: Y(); }; void x(int) { Y y; }",
                 Constructor));
}

TEST(Matcher, VisitsTemplateInstantiations) {
  EXPECT_TRUE(matches(
      "class A { public: void x(); };"
      "template <typename T> class B { public: void y() { T t; t.x(); } };"
      "void f() { B<A> b; b.y(); }",
      callExpr(callee(cxxMethodDecl(hasName("x"))))));

  EXPECT_TRUE(matches(
      "class A { public: void x(); };"
      "class C {"
      " public:"
      "  template <typename T> class B { public: void y() { T t; t.x(); } };"
      "};"
      "void f() {"
      "  C::B<A> b; b.y();"
      "}",
      recordDecl(hasName("C"), hasDescendant(callExpr(
                                   callee(cxxMethodDecl(hasName("x"))))))));
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

// For testing AST_MATCHER_P().
AST_MATCHER_P(Decl, just, internal::Matcher<Decl>, AMatcher) {
  // Make sure all special variables are used: node, match_finder,
  // bound_nodes_builder, and the parameter named 'AMatcher'.
  return AMatcher.matches(Node, Finder, Builder);
}

TEST(AstMatcherPMacro, Works) {
  DeclarationMatcher HasClassB = just(has(recordDecl(hasName("B")).bind("b")));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundTo<Decl>("b")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundTo<Decl>("a")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, new VerifyIdIsBoundTo<Decl>("b")));
}

AST_POLYMORPHIC_MATCHER_P(polymorphicHas,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt),
                          internal::Matcher<Decl>, AMatcher) {
  return Finder->matchesChildOf(
      Node, AMatcher, Builder,
      ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses,
      ASTMatchFinder::BK_First);
}

TEST(AstPolymorphicMatcherPMacro, Works) {
  DeclarationMatcher HasClassB =
      polymorphicHas(recordDecl(hasName("B")).bind("b"));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundTo<Decl>("b")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, new VerifyIdIsBoundTo<Decl>("a")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, new VerifyIdIsBoundTo<Decl>("b")));

  StatementMatcher StatementHasClassB =
      polymorphicHas(recordDecl(hasName("B")));

  EXPECT_TRUE(matches("void x() { class B {}; }", StatementHasClassB));
}

TEST(For, FindsForLoops) {
  EXPECT_TRUE(matches("void f() { for(;;); }", forStmt()));
  EXPECT_TRUE(matches("void f() { if(true) for(;;); }", forStmt()));
  EXPECT_TRUE(notMatches("int as[] = { 1, 2, 3 };"
                         "void f() { for (auto &a : as); }",
                         forStmt()));
}

TEST(For, ForLoopInternals) {
  EXPECT_TRUE(matches("void f(){ int i; for (; i < 3 ; ); }",
                      forStmt(hasCondition(anything()))));
  EXPECT_TRUE(matches("void f() { for (int i = 0; ;); }",
                      forStmt(hasLoopInit(anything()))));
}

TEST(For, ForRangeLoopInternals) {
  EXPECT_TRUE(matches("void f(){ int a[] {1, 2}; for (int i : a); }",
                      cxxForRangeStmt(hasLoopVariable(anything()))));
  EXPECT_TRUE(matches(
      "void f(){ int a[] {1, 2}; for (int i : a); }",
      cxxForRangeStmt(hasRangeInit(declRefExpr(to(varDecl(hasName("a"))))))));
}

TEST(For, NegativeForLoopInternals) {
  EXPECT_TRUE(notMatches("void f(){ for (int i = 0; ; ++i); }",
                         forStmt(hasCondition(expr()))));
  EXPECT_TRUE(notMatches("void f() {int i; for (; i < 4; ++i) {} }",
                         forStmt(hasLoopInit(anything()))));
}

TEST(For, ReportsNoFalsePositives) {
  EXPECT_TRUE(notMatches("void f() { ; }", forStmt()));
  EXPECT_TRUE(notMatches("void f() { if(true); }", forStmt()));
}

TEST(CompoundStatement, HandlesSimpleCases) {
  EXPECT_TRUE(notMatches("void f();", compoundStmt()));
  EXPECT_TRUE(matches("void f() {}", compoundStmt()));
  EXPECT_TRUE(matches("void f() {{}}", compoundStmt()));
}

TEST(CompoundStatement, DoesNotMatchEmptyStruct) {
  // It's not a compound statement just because there's "{}" in the source
  // text. This is an AST search, not grep.
  EXPECT_TRUE(notMatches("namespace n { struct S {}; }",
              compoundStmt()));
  EXPECT_TRUE(matches("namespace n { struct S { void f() {{}} }; }",
              compoundStmt()));
}

TEST(HasBody, FindsBodyOfForWhileDoLoops) {
  EXPECT_TRUE(matches("void f() { for(;;) {} }",
              forStmt(hasBody(compoundStmt()))));
  EXPECT_TRUE(notMatches("void f() { for(;;); }",
              forStmt(hasBody(compoundStmt()))));
  EXPECT_TRUE(matches("void f() { while(true) {} }",
              whileStmt(hasBody(compoundStmt()))));
  EXPECT_TRUE(matches("void f() { do {} while(true); }",
              doStmt(hasBody(compoundStmt()))));
  EXPECT_TRUE(matches("void f() { int p[2]; for (auto x : p) {} }",
              cxxForRangeStmt(hasBody(compoundStmt()))));
  EXPECT_TRUE(matches("void f() {}", functionDecl(hasBody(compoundStmt()))));
  EXPECT_TRUE(notMatches("void f();", functionDecl(hasBody(compoundStmt()))));
  EXPECT_TRUE(matches("void f(); void f() {}",
                         functionDecl(hasBody(compoundStmt()))));
}

TEST(HasAnySubstatement, MatchesForTopLevelCompoundStatement) {
  // The simplest case: every compound statement is in a function
  // definition, and the function body itself must be a compound
  // statement.
  EXPECT_TRUE(matches("void f() { for (;;); }",
              compoundStmt(hasAnySubstatement(forStmt()))));
}

TEST(HasAnySubstatement, IsNotRecursive) {
  // It's really "has any immediate substatement".
  EXPECT_TRUE(notMatches("void f() { if (true) for (;;); }",
              compoundStmt(hasAnySubstatement(forStmt()))));
}

TEST(HasAnySubstatement, MatchesInNestedCompoundStatements) {
  EXPECT_TRUE(matches("void f() { if (true) { for (;;); } }",
              compoundStmt(hasAnySubstatement(forStmt()))));
}

TEST(HasAnySubstatement, FindsSubstatementBetweenOthers) {
  EXPECT_TRUE(matches("void f() { 1; 2; 3; for (;;); 4; 5; 6; }",
              compoundStmt(hasAnySubstatement(forStmt()))));
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

TEST(Member, MatchesMemberAllocationFunction) {
  // Fails in C++11 mode
  EXPECT_TRUE(matchesConditionally(
      "namespace std { typedef typeof(sizeof(int)) size_t; }"
      "class X { void *operator new(std::size_t); };",
      cxxMethodDecl(ofClass(hasName("X"))), true, "-std=gnu++98"));

  EXPECT_TRUE(matches("class X { void operator delete(void*); };",
                      cxxMethodDecl(ofClass(hasName("X")))));

  // Fails in C++11 mode
  EXPECT_TRUE(matchesConditionally(
      "namespace std { typedef typeof(sizeof(int)) size_t; }"
      "class X { void operator delete[](void*, std::size_t); };",
      cxxMethodDecl(ofClass(hasName("X"))), true, "-std=gnu++98"));
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

TEST(CastExpression, MatchesExplicitCasts) {
  EXPECT_TRUE(matches("char *p = reinterpret_cast<char *>(&p);",castExpr()));
  EXPECT_TRUE(matches("void *p = (void *)(&p);", castExpr()));
  EXPECT_TRUE(matches("char q, *p = const_cast<char *>(&q);", castExpr()));
  EXPECT_TRUE(matches("char c = char(0);", castExpr()));
}
TEST(CastExpression, MatchesImplicitCasts) {
  // This test creates an implicit cast from int to char.
  EXPECT_TRUE(matches("char c = 0;", castExpr()));
  // This test creates an implicit cast from lvalue to rvalue.
  EXPECT_TRUE(matches("char c = 0, d = c;", castExpr()));
}

TEST(CastExpression, DoesNotMatchNonCasts) {
  EXPECT_TRUE(notMatches("char c = '0';", castExpr()));
  EXPECT_TRUE(notMatches("char c, &q = c;", castExpr()));
  EXPECT_TRUE(notMatches("int i = (0);", castExpr()));
  EXPECT_TRUE(notMatches("int i = 0;", castExpr()));
}

TEST(ReinterpretCast, MatchesSimpleCase) {
  EXPECT_TRUE(matches("char* p = reinterpret_cast<char*>(&p);",
                      cxxReinterpretCastExpr()));
}

TEST(ReinterpretCast, DoesNotMatchOtherCasts) {
  EXPECT_TRUE(notMatches("char* p = (char*)(&p);", cxxReinterpretCastExpr()));
  EXPECT_TRUE(notMatches("char q, *p = const_cast<char*>(&q);",
                         cxxReinterpretCastExpr()));
  EXPECT_TRUE(notMatches("void* p = static_cast<void*>(&p);",
                         cxxReinterpretCastExpr()));
  EXPECT_TRUE(notMatches("struct B { virtual ~B() {} }; struct D : B {};"
                         "B b;"
                         "D* p = dynamic_cast<D*>(&b);",
                         cxxReinterpretCastExpr()));
}

TEST(FunctionalCast, MatchesSimpleCase) {
  std::string foo_class = "class Foo { public: Foo(const char*); };";
  EXPECT_TRUE(matches(foo_class + "void r() { Foo f = Foo(\"hello world\"); }",
                      cxxFunctionalCastExpr()));
}

TEST(FunctionalCast, DoesNotMatchOtherCasts) {
  std::string FooClass = "class Foo { public: Foo(const char*); };";
  EXPECT_TRUE(
      notMatches(FooClass + "void r() { Foo f = (Foo) \"hello world\"; }",
                 cxxFunctionalCastExpr()));
  EXPECT_TRUE(
      notMatches(FooClass + "void r() { Foo f = \"hello world\"; }",
                 cxxFunctionalCastExpr()));
}

TEST(DynamicCast, MatchesSimpleCase) {
  EXPECT_TRUE(matches("struct B { virtual ~B() {} }; struct D : B {};"
                      "B b;"
                      "D* p = dynamic_cast<D*>(&b);",
                      cxxDynamicCastExpr()));
}

TEST(StaticCast, MatchesSimpleCase) {
  EXPECT_TRUE(matches("void* p(static_cast<void*>(&p));",
                      cxxStaticCastExpr()));
}

TEST(StaticCast, DoesNotMatchOtherCasts) {
  EXPECT_TRUE(notMatches("char* p = (char*)(&p);", cxxStaticCastExpr()));
  EXPECT_TRUE(notMatches("char q, *p = const_cast<char*>(&q);",
                         cxxStaticCastExpr()));
  EXPECT_TRUE(notMatches("void* p = reinterpret_cast<char*>(&p);",
                         cxxStaticCastExpr()));
  EXPECT_TRUE(notMatches("struct B { virtual ~B() {} }; struct D : B {};"
                         "B b;"
                         "D* p = dynamic_cast<D*>(&b);",
                         cxxStaticCastExpr()));
}

TEST(CStyleCast, MatchesSimpleCase) {
  EXPECT_TRUE(matches("int i = (int) 2.2f;", cStyleCastExpr()));
}

TEST(CStyleCast, DoesNotMatchOtherCasts) {
  EXPECT_TRUE(notMatches("char* p = static_cast<char*>(0);"
                         "char q, *r = const_cast<char*>(&q);"
                         "void* s = reinterpret_cast<char*>(&s);"
                         "struct B { virtual ~B() {} }; struct D : B {};"
                         "B b;"
                         "D* t = dynamic_cast<D*>(&b);",
                         cStyleCastExpr()));
}

TEST(HasDestinationType, MatchesSimpleCase) {
  EXPECT_TRUE(matches("char* p = static_cast<char*>(0);",
                      cxxStaticCastExpr(hasDestinationType(
                          pointsTo(TypeMatcher(anything()))))));
}

TEST(HasImplicitDestinationType, MatchesSimpleCase) {
  // This test creates an implicit const cast.
  EXPECT_TRUE(matches("int x; const int i = x;",
                      implicitCastExpr(
                          hasImplicitDestinationType(isInteger()))));
  // This test creates an implicit array-to-pointer cast.
  EXPECT_TRUE(matches("int arr[3]; int *p = arr;",
                      implicitCastExpr(hasImplicitDestinationType(
                          pointsTo(TypeMatcher(anything()))))));
}

TEST(HasImplicitDestinationType, DoesNotMatchIncorrectly) {
  // This test creates an implicit cast from int to char.
  EXPECT_TRUE(notMatches("char c = 0;",
                      implicitCastExpr(hasImplicitDestinationType(
                          unless(anything())))));
  // This test creates an implicit array-to-pointer cast.
  EXPECT_TRUE(notMatches("int arr[3]; int *p = arr;",
                      implicitCastExpr(hasImplicitDestinationType(
                          unless(anything())))));
}

TEST(ImplicitCast, MatchesSimpleCase) {
  // This test creates an implicit const cast.
  EXPECT_TRUE(matches("int x = 0; const int y = x;",
                      varDecl(hasInitializer(implicitCastExpr()))));
  // This test creates an implicit cast from int to char.
  EXPECT_TRUE(matches("char c = 0;",
                      varDecl(hasInitializer(implicitCastExpr()))));
  // This test creates an implicit array-to-pointer cast.
  EXPECT_TRUE(matches("int arr[6]; int *p = arr;",
                      varDecl(hasInitializer(implicitCastExpr()))));
}

TEST(ImplicitCast, DoesNotMatchIncorrectly) {
  // This test verifies that implicitCastExpr() matches exactly when implicit casts
  // are present, and that it ignores explicit and paren casts.

  // These two test cases have no casts.
  EXPECT_TRUE(notMatches("int x = 0;",
                         varDecl(hasInitializer(implicitCastExpr()))));
  EXPECT_TRUE(notMatches("int x = 0, &y = x;",
                         varDecl(hasInitializer(implicitCastExpr()))));

  EXPECT_TRUE(notMatches("int x = 0; double d = (double) x;",
                         varDecl(hasInitializer(implicitCastExpr()))));
  EXPECT_TRUE(notMatches("const int *p; int *q = const_cast<int *>(p);",
                         varDecl(hasInitializer(implicitCastExpr()))));

  EXPECT_TRUE(notMatches("int x = (0);",
                         varDecl(hasInitializer(implicitCastExpr()))));
}

TEST(IgnoringImpCasts, MatchesImpCasts) {
  // This test checks that ignoringImpCasts matches when implicit casts are
  // present and its inner matcher alone does not match.
  // Note that this test creates an implicit const cast.
  EXPECT_TRUE(matches("int x = 0; const int y = x;",
                      varDecl(hasInitializer(ignoringImpCasts(
                          declRefExpr(to(varDecl(hasName("x")))))))));
  // This test creates an implict cast from int to char.
  EXPECT_TRUE(matches("char x = 0;",
                      varDecl(hasInitializer(ignoringImpCasts(
                          integerLiteral(equals(0)))))));
}

TEST(IgnoringImpCasts, DoesNotMatchIncorrectly) {
  // These tests verify that ignoringImpCasts does not match if the inner
  // matcher does not match.
  // Note that the first test creates an implicit const cast.
  EXPECT_TRUE(notMatches("int x; const int y = x;",
                         varDecl(hasInitializer(ignoringImpCasts(
                             unless(anything()))))));
  EXPECT_TRUE(notMatches("int x; int y = x;",
                         varDecl(hasInitializer(ignoringImpCasts(
                             unless(anything()))))));

  // These tests verify that ignoringImplictCasts does not look through explicit
  // casts or parentheses.
  EXPECT_TRUE(notMatches("char* p = static_cast<char*>(0);",
                         varDecl(hasInitializer(ignoringImpCasts(
                             integerLiteral())))));
  EXPECT_TRUE(notMatches("int i = (0);",
                         varDecl(hasInitializer(ignoringImpCasts(
                             integerLiteral())))));
  EXPECT_TRUE(notMatches("float i = (float)0;",
                         varDecl(hasInitializer(ignoringImpCasts(
                             integerLiteral())))));
  EXPECT_TRUE(notMatches("float i = float(0);",
                         varDecl(hasInitializer(ignoringImpCasts(
                             integerLiteral())))));
}

TEST(IgnoringImpCasts, MatchesWithoutImpCasts) {
  // This test verifies that expressions that do not have implicit casts
  // still match the inner matcher.
  EXPECT_TRUE(matches("int x = 0; int &y = x;",
                      varDecl(hasInitializer(ignoringImpCasts(
                          declRefExpr(to(varDecl(hasName("x")))))))));
}

TEST(IgnoringParenCasts, MatchesParenCasts) {
  // This test checks that ignoringParenCasts matches when parentheses and/or
  // casts are present and its inner matcher alone does not match.
  EXPECT_TRUE(matches("int x = (0);",
                      varDecl(hasInitializer(ignoringParenCasts(
                          integerLiteral(equals(0)))))));
  EXPECT_TRUE(matches("int x = (((((0)))));",
                      varDecl(hasInitializer(ignoringParenCasts(
                          integerLiteral(equals(0)))))));

  // This test creates an implict cast from int to char in addition to the
  // parentheses.
  EXPECT_TRUE(matches("char x = (0);",
                      varDecl(hasInitializer(ignoringParenCasts(
                          integerLiteral(equals(0)))))));

  EXPECT_TRUE(matches("char x = (char)0;",
                      varDecl(hasInitializer(ignoringParenCasts(
                          integerLiteral(equals(0)))))));
  EXPECT_TRUE(matches("char* p = static_cast<char*>(0);",
                      varDecl(hasInitializer(ignoringParenCasts(
                          integerLiteral(equals(0)))))));
}

TEST(IgnoringParenCasts, MatchesWithoutParenCasts) {
  // This test verifies that expressions that do not have any casts still match.
  EXPECT_TRUE(matches("int x = 0;",
                      varDecl(hasInitializer(ignoringParenCasts(
                          integerLiteral(equals(0)))))));
}

TEST(IgnoringParenCasts, DoesNotMatchIncorrectly) {
  // These tests verify that ignoringImpCasts does not match if the inner
  // matcher does not match.
  EXPECT_TRUE(notMatches("int x = ((0));",
                         varDecl(hasInitializer(ignoringParenCasts(
                             unless(anything()))))));

  // This test creates an implicit cast from int to char in addition to the
  // parentheses.
  EXPECT_TRUE(notMatches("char x = ((0));",
                         varDecl(hasInitializer(ignoringParenCasts(
                             unless(anything()))))));

  EXPECT_TRUE(notMatches("char *x = static_cast<char *>((0));",
                         varDecl(hasInitializer(ignoringParenCasts(
                             unless(anything()))))));
}

TEST(IgnoringParenAndImpCasts, MatchesParenImpCasts) {
  // This test checks that ignoringParenAndImpCasts matches when
  // parentheses and/or implicit casts are present and its inner matcher alone
  // does not match.
  // Note that this test creates an implicit const cast.
  EXPECT_TRUE(matches("int x = 0; const int y = x;",
                      varDecl(hasInitializer(ignoringParenImpCasts(
                          declRefExpr(to(varDecl(hasName("x")))))))));
  // This test creates an implicit cast from int to char.
  EXPECT_TRUE(matches("const char x = (0);",
                      varDecl(hasInitializer(ignoringParenImpCasts(
                          integerLiteral(equals(0)))))));
}

TEST(IgnoringParenAndImpCasts, MatchesWithoutParenImpCasts) {
  // This test verifies that expressions that do not have parentheses or
  // implicit casts still match.
  EXPECT_TRUE(matches("int x = 0; int &y = x;",
                      varDecl(hasInitializer(ignoringParenImpCasts(
                          declRefExpr(to(varDecl(hasName("x")))))))));
  EXPECT_TRUE(matches("int x = 0;",
                      varDecl(hasInitializer(ignoringParenImpCasts(
                          integerLiteral(equals(0)))))));
}

TEST(IgnoringParenAndImpCasts, DoesNotMatchIncorrectly) {
  // These tests verify that ignoringParenImpCasts does not match if
  // the inner matcher does not match.
  // This test creates an implicit cast.
  EXPECT_TRUE(notMatches("char c = ((3));",
                         varDecl(hasInitializer(ignoringParenImpCasts(
                             unless(anything()))))));
  // These tests verify that ignoringParenAndImplictCasts does not look
  // through explicit casts.
  EXPECT_TRUE(notMatches("float y = (float(0));",
                         varDecl(hasInitializer(ignoringParenImpCasts(
                             integerLiteral())))));
  EXPECT_TRUE(notMatches("float y = (float)0;",
                         varDecl(hasInitializer(ignoringParenImpCasts(
                             integerLiteral())))));
  EXPECT_TRUE(notMatches("char* p = static_cast<char*>(0);",
                         varDecl(hasInitializer(ignoringParenImpCasts(
                             integerLiteral())))));
}

TEST(HasSourceExpression, MatchesImplicitCasts) {
  EXPECT_TRUE(matches("class string {}; class URL { public: URL(string s); };"
                      "void r() {string a_string; URL url = a_string; }",
                      implicitCastExpr(
                          hasSourceExpression(cxxConstructExpr()))));
}

TEST(HasSourceExpression, MatchesExplicitCasts) {
  EXPECT_TRUE(matches("float x = static_cast<float>(42);",
                      explicitCastExpr(
                          hasSourceExpression(hasDescendant(
                              expr(integerLiteral()))))));
}

TEST(Statement, DoesNotMatchDeclarations) {
  EXPECT_TRUE(notMatches("class X {};", stmt()));
}

TEST(Statement, MatchesCompoundStatments) {
  EXPECT_TRUE(matches("void x() {}", stmt()));
}

TEST(DeclarationStatement, DoesNotMatchCompoundStatements) {
  EXPECT_TRUE(notMatches("void x() {}", declStmt()));
}

TEST(DeclarationStatement, MatchesVariableDeclarationStatements) {
  EXPECT_TRUE(matches("void x() { int a; }", declStmt()));
}

TEST(ExprWithCleanups, MatchesExprWithCleanups) {
  EXPECT_TRUE(matches("struct Foo { ~Foo(); };"
                      "const Foo f = Foo();",
                      varDecl(hasInitializer(exprWithCleanups()))));
  EXPECT_FALSE(matches("struct Foo { };"
                      "const Foo f = Foo();",
                      varDecl(hasInitializer(exprWithCleanups()))));
}

TEST(InitListExpression, MatchesInitListExpression) {
  EXPECT_TRUE(matches("int a[] = { 1, 2 };",
                      initListExpr(hasType(asString("int [2]")))));
  EXPECT_TRUE(matches("struct B { int x, y; }; B b = { 5, 6 };",
                      initListExpr(hasType(recordDecl(hasName("B"))))));
  EXPECT_TRUE(matches("struct S { S(void (*a)()); };"
                      "void f();"
                      "S s[1] = { &f };",
                      declRefExpr(to(functionDecl(hasName("f"))))));
  EXPECT_TRUE(
      matches("int i[1] = {42, [0] = 43};", integerLiteral(equals(42))));
}

TEST(UsingDeclaration, MatchesUsingDeclarations) {
  EXPECT_TRUE(matches("namespace X { int x; } using X::x;",
                      usingDecl()));
}

TEST(UsingDeclaration, MatchesShadowUsingDelcarations) {
  EXPECT_TRUE(matches("namespace f { int a; } using f::a;",
                      usingDecl(hasAnyUsingShadowDecl(hasName("a")))));
}

TEST(UsingDeclaration, MatchesSpecificTarget) {
  EXPECT_TRUE(matches("namespace f { int a; void b(); } using f::b;",
                      usingDecl(hasAnyUsingShadowDecl(
                          hasTargetDecl(functionDecl())))));
  EXPECT_TRUE(notMatches("namespace f { int a; void b(); } using f::a;",
                         usingDecl(hasAnyUsingShadowDecl(
                             hasTargetDecl(functionDecl())))));
}

TEST(UsingDeclaration, ThroughUsingDeclaration) {
  EXPECT_TRUE(matches(
      "namespace a { void f(); } using a::f; void g() { f(); }",
      declRefExpr(throughUsingDecl(anything()))));
  EXPECT_TRUE(notMatches(
      "namespace a { void f(); } using a::f; void g() { a::f(); }",
      declRefExpr(throughUsingDecl(anything()))));
}

TEST(UsingDirectiveDeclaration, MatchesUsingNamespace) {
  EXPECT_TRUE(matches("namespace X { int x; } using namespace X;",
                      usingDirectiveDecl()));
  EXPECT_FALSE(
      matches("namespace X { int x; } using X::x;", usingDirectiveDecl()));
}

TEST(SingleDecl, IsSingleDecl) {
  StatementMatcher SingleDeclStmt =
      declStmt(hasSingleDecl(varDecl(hasInitializer(anything()))));
  EXPECT_TRUE(matches("void f() {int a = 4;}", SingleDeclStmt));
  EXPECT_TRUE(notMatches("void f() {int a;}", SingleDeclStmt));
  EXPECT_TRUE(notMatches("void f() {int a = 4, b = 3;}",
                          SingleDeclStmt));
}

TEST(DeclStmt, ContainsDeclaration) {
  DeclarationMatcher MatchesInit = varDecl(hasInitializer(anything()));

  EXPECT_TRUE(matches("void f() {int a = 4;}",
                      declStmt(containsDeclaration(0, MatchesInit))));
  EXPECT_TRUE(matches("void f() {int a = 4, b = 3;}",
                      declStmt(containsDeclaration(0, MatchesInit),
                               containsDeclaration(1, MatchesInit))));
  unsigned WrongIndex = 42;
  EXPECT_TRUE(notMatches("void f() {int a = 4, b = 3;}",
                         declStmt(containsDeclaration(WrongIndex,
                                                      MatchesInit))));
}

TEST(DeclCount, DeclCountIsCorrect) {
  EXPECT_TRUE(matches("void f() {int i,j;}",
                      declStmt(declCountIs(2))));
  EXPECT_TRUE(notMatches("void f() {int i,j; int k;}",
                         declStmt(declCountIs(3))));
  EXPECT_TRUE(notMatches("void f() {int i,j, k, l;}",
                         declStmt(declCountIs(3))));
}

TEST(While, MatchesWhileLoops) {
  EXPECT_TRUE(notMatches("void x() {}", whileStmt()));
  EXPECT_TRUE(matches("void x() { while(true); }", whileStmt()));
  EXPECT_TRUE(notMatches("void x() { do {} while(true); }", whileStmt()));
}

TEST(Do, MatchesDoLoops) {
  EXPECT_TRUE(matches("void x() { do {} while(true); }", doStmt()));
  EXPECT_TRUE(matches("void x() { do ; while(false); }", doStmt()));
}

TEST(Do, DoesNotMatchWhileLoops) {
  EXPECT_TRUE(notMatches("void x() { while(true) {} }", doStmt()));
}

TEST(SwitchCase, MatchesCase) {
  EXPECT_TRUE(matches("void x() { switch(42) { case 42:; } }", switchCase()));
  EXPECT_TRUE(matches("void x() { switch(42) { default:; } }", switchCase()));
  EXPECT_TRUE(matches("void x() { switch(42) default:; }", switchCase()));
  EXPECT_TRUE(notMatches("void x() { switch(42) {} }", switchCase()));
}

TEST(SwitchCase, MatchesSwitch) {
  EXPECT_TRUE(matches("void x() { switch(42) { case 42:; } }", switchStmt()));
  EXPECT_TRUE(matches("void x() { switch(42) { default:; } }", switchStmt()));
  EXPECT_TRUE(matches("void x() { switch(42) default:; }", switchStmt()));
  EXPECT_TRUE(notMatches("void x() {}", switchStmt()));
}

TEST(SwitchCase, MatchesEachCase) {
  EXPECT_TRUE(notMatches("void x() { switch(42); }",
                         switchStmt(forEachSwitchCase(caseStmt()))));
  EXPECT_TRUE(matches("void x() { switch(42) case 42:; }",
                      switchStmt(forEachSwitchCase(caseStmt()))));
  EXPECT_TRUE(matches("void x() { switch(42) { case 42:; } }",
                      switchStmt(forEachSwitchCase(caseStmt()))));
  EXPECT_TRUE(notMatches(
      "void x() { if (1) switch(42) { case 42: switch (42) { default:; } } }",
      ifStmt(has(switchStmt(forEachSwitchCase(defaultStmt()))))));
  EXPECT_TRUE(matches("void x() { switch(42) { case 1+1: case 4:; } }",
                      switchStmt(forEachSwitchCase(
                          caseStmt(hasCaseConstant(integerLiteral()))))));
  EXPECT_TRUE(notMatches("void x() { switch(42) { case 1+1: case 2+2:; } }",
                         switchStmt(forEachSwitchCase(
                             caseStmt(hasCaseConstant(integerLiteral()))))));
  EXPECT_TRUE(notMatches("void x() { switch(42) { case 1 ... 2:; } }",
                         switchStmt(forEachSwitchCase(
                             caseStmt(hasCaseConstant(integerLiteral()))))));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void x() { switch (42) { case 1: case 2: case 3: default:; } }",
      switchStmt(forEachSwitchCase(caseStmt().bind("x"))),
      new VerifyIdIsBoundTo<CaseStmt>("x", 3)));
}

TEST(ForEachConstructorInitializer, MatchesInitializers) {
  EXPECT_TRUE(matches(
      "struct X { X() : i(42), j(42) {} int i, j; };",
      cxxConstructorDecl(forEachConstructorInitializer(cxxCtorInitializer()))));
}

TEST(ExceptionHandling, SimpleCases) {
  EXPECT_TRUE(matches("void foo() try { } catch(int X) { }", cxxCatchStmt()));
  EXPECT_TRUE(matches("void foo() try { } catch(int X) { }", cxxTryStmt()));
  EXPECT_TRUE(
      notMatches("void foo() try { } catch(int X) { }", cxxThrowExpr()));
  EXPECT_TRUE(matches("void foo() try { throw; } catch(int X) { }",
                      cxxThrowExpr()));
  EXPECT_TRUE(matches("void foo() try { throw 5;} catch(int X) { }",
                      cxxThrowExpr()));
  EXPECT_TRUE(matches("void foo() try { throw; } catch(...) { }",
                      cxxCatchStmt(isCatchAll())));
  EXPECT_TRUE(notMatches("void foo() try { throw; } catch(int) { }",
                         cxxCatchStmt(isCatchAll())));
  EXPECT_TRUE(matches("void foo() try {} catch(int X) { }",
                      varDecl(isExceptionVariable())));
  EXPECT_TRUE(notMatches("void foo() try { int X; } catch (...) { }",
                         varDecl(isExceptionVariable())));
}

TEST(ParenExpression, SimpleCases) {
  EXPECT_TRUE(matches("int i = (3);", parenExpr()));
  EXPECT_TRUE(matches("int i = (3 + 7);", parenExpr()));
  EXPECT_TRUE(notMatches("int i = 3;", parenExpr()));
  EXPECT_TRUE(notMatches("int foo() { return 1; }; int a = foo();",
                         parenExpr()));
}

TEST(HasConditionVariableStatement, DoesNotMatchCondition) {
  EXPECT_TRUE(notMatches(
      "void x() { if(true) {} }",
      ifStmt(hasConditionVariableStatement(declStmt()))));
  EXPECT_TRUE(notMatches(
      "void x() { int x; if((x = 42)) {} }",
      ifStmt(hasConditionVariableStatement(declStmt()))));
}

TEST(HasConditionVariableStatement, MatchesConditionVariables) {
  EXPECT_TRUE(matches(
      "void x() { if(int* a = 0) {} }",
      ifStmt(hasConditionVariableStatement(declStmt()))));
}

TEST(ForEach, BindsOneNode) {
  EXPECT_TRUE(matchAndVerifyResultTrue("class C { int x; };",
      recordDecl(hasName("C"), forEach(fieldDecl(hasName("x")).bind("x"))),
      new VerifyIdIsBoundTo<FieldDecl>("x", 1)));
}

TEST(ForEach, BindsMultipleNodes) {
  EXPECT_TRUE(matchAndVerifyResultTrue("class C { int x; int y; int z; };",
      recordDecl(hasName("C"), forEach(fieldDecl().bind("f"))),
      new VerifyIdIsBoundTo<FieldDecl>("f", 3)));
}

TEST(ForEach, BindsRecursiveCombinations) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { class D { int x; int y; }; class E { int y; int z; }; };",
      recordDecl(hasName("C"),
                 forEach(recordDecl(forEach(fieldDecl().bind("f"))))),
      new VerifyIdIsBoundTo<FieldDecl>("f", 4)));
}

TEST(ForEachDescendant, BindsOneNode) {
  EXPECT_TRUE(matchAndVerifyResultTrue("class C { class D { int x; }; };",
      recordDecl(hasName("C"),
                 forEachDescendant(fieldDecl(hasName("x")).bind("x"))),
      new VerifyIdIsBoundTo<FieldDecl>("x", 1)));
}

TEST(ForEachDescendant, NestedForEachDescendant) {
  DeclarationMatcher m = recordDecl(
      isDefinition(), decl().bind("x"), hasName("C"));
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "class A { class B { class C {}; }; };",
    recordDecl(hasName("A"), anyOf(m, forEachDescendant(m))),
    new VerifyIdIsBoundTo<Decl>("x", "C")));

  // Check that a partial match of 'm' that binds 'x' in the
  // first part of anyOf(m, anything()) will not overwrite the
  // binding created by the earlier binding in the hasDescendant.
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { class B { class C {}; }; };",
      recordDecl(hasName("A"), allOf(hasDescendant(m), anyOf(m, anything()))),
      new VerifyIdIsBoundTo<Decl>("x", "C")));
}

TEST(ForEachDescendant, BindsMultipleNodes) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { class D { int x; int y; }; "
      "          class E { class F { int y; int z; }; }; };",
      recordDecl(hasName("C"), forEachDescendant(fieldDecl().bind("f"))),
      new VerifyIdIsBoundTo<FieldDecl>("f", 4)));
}

TEST(ForEachDescendant, BindsRecursiveCombinations) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { class D { "
      "          class E { class F { class G { int y; int z; }; }; }; }; };",
      recordDecl(hasName("C"), forEachDescendant(recordDecl(
          forEachDescendant(fieldDecl().bind("f"))))),
      new VerifyIdIsBoundTo<FieldDecl>("f", 8)));
}

TEST(ForEachDescendant, BindsCombinations) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f() { if(true) {} if (true) {} while (true) {} if (true) {} while "
      "(true) {} }",
      compoundStmt(forEachDescendant(ifStmt().bind("if")),
                   forEachDescendant(whileStmt().bind("while"))),
      new VerifyIdIsBoundTo<IfStmt>("if", 6)));
}

TEST(Has, DoesNotDeleteBindings) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { int a; };", recordDecl(decl().bind("x"), has(fieldDecl())),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
}

TEST(LoopingMatchers, DoNotOverwritePreviousMatchResultOnFailure) {
  // Those matchers cover all the cases where an inner matcher is called
  // and there is not a 1:1 relationship between the match of the outer
  // matcher and the match of the inner matcher.
  // The pattern to look for is:
  //   ... return InnerMatcher.matches(...); ...
  // In which case no special handling is needed.
  //
  // On the other hand, if there are multiple alternative matches
  // (for example forEach*) or matches might be discarded (for example has*)
  // the implementation must make sure that the discarded matches do not
  // affect the bindings.
  // When new such matchers are added, add a test here that:
  // - matches a simple node, and binds it as the first thing in the matcher:
  //     recordDecl(decl().bind("x"), hasName("X")))
  // - uses the matcher under test afterwards in a way that not the first
  //   alternative is matched; for anyOf, that means the first branch
  //   would need to return false; for hasAncestor, it means that not
  //   the direct parent matches the inner matcher.

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { int y; };",
      recordDecl(
          recordDecl().bind("x"), hasName("::X"),
          anyOf(forEachDescendant(recordDecl(hasName("Y"))), anything())),
      new VerifyIdIsBoundTo<CXXRecordDecl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X {};", recordDecl(recordDecl().bind("x"), hasName("::X"),
                                anyOf(unless(anything()), anything())),
      new VerifyIdIsBoundTo<CXXRecordDecl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "template<typename T1, typename T2> class X {}; X<float, int> x;",
      classTemplateSpecializationDecl(
          decl().bind("x"),
          hasAnyTemplateArgument(refersToType(asString("int")))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { void f(); void g(); };",
      cxxRecordDecl(decl().bind("x"), hasMethod(hasName("g"))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { X() : a(1), b(2) {} double a; int b; };",
      recordDecl(decl().bind("x"),
                 has(cxxConstructorDecl(
                     hasAnyConstructorInitializer(forField(hasName("b")))))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void x(int, int) { x(0, 42); }",
      callExpr(expr().bind("x"), hasAnyArgument(integerLiteral(equals(42)))),
      new VerifyIdIsBoundTo<Expr>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void x(int, int y) {}",
      functionDecl(decl().bind("x"), hasAnyParameter(hasName("y"))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void x() { return; if (true) {} }",
      functionDecl(decl().bind("x"),
                   has(compoundStmt(hasAnySubstatement(ifStmt())))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "namespace X { void b(int); void b(); }"
      "using X::b;",
      usingDecl(decl().bind("x"), hasAnyUsingShadowDecl(hasTargetDecl(
                                      functionDecl(parameterCountIs(1))))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A{}; class B{}; class C : B, A {};",
      cxxRecordDecl(decl().bind("x"), isDerivedFrom("::A")),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A{}; typedef A B; typedef A C; typedef A D;"
      "class E : A {};",
      cxxRecordDecl(decl().bind("x"), isDerivedFrom("C")),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { class B { void f() {} }; };",
      functionDecl(decl().bind("x"), hasAncestor(recordDecl(hasName("::A")))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "template <typename T> struct A { struct B {"
      "  void f() { if(true) {} }"
      "}; };"
      "void t() { A<int>::B b; b.f(); }",
      ifStmt(stmt().bind("x"), hasAncestor(recordDecl(hasName("::A")))),
      new VerifyIdIsBoundTo<Stmt>("x", 2)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A {};",
      recordDecl(hasName("::A"), decl().bind("x"), unless(hasName("fooble"))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { A() : s(), i(42) {} const char *s; int i; };",
      cxxConstructorDecl(hasName("::A::A"), decl().bind("x"),
                         forEachConstructorInitializer(forField(hasName("i")))),
      new VerifyIdIsBoundTo<Decl>("x", 1)));
}

TEST(ForEachDescendant, BindsCorrectNodes) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { void f(); int i; };",
      recordDecl(hasName("C"), forEachDescendant(decl().bind("decl"))),
      new VerifyIdIsBoundTo<FieldDecl>("decl", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { void f() {} int i; };",
      recordDecl(hasName("C"), forEachDescendant(decl().bind("decl"))),
      new VerifyIdIsBoundTo<FunctionDecl>("decl", 1)));
}

TEST(FindAll, BindsNodeOnMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A {};",
      recordDecl(hasName("::A"), findAll(recordDecl(hasName("::A")).bind("v"))),
      new VerifyIdIsBoundTo<CXXRecordDecl>("v", 1)));
}

TEST(FindAll, BindsDescendantNodeOnMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { int a; int b; };",
      recordDecl(hasName("::A"), findAll(fieldDecl().bind("v"))),
      new VerifyIdIsBoundTo<FieldDecl>("v", 2)));
}

TEST(FindAll, BindsNodeAndDescendantNodesOnOneMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { int a; int b; };",
      recordDecl(hasName("::A"),
                 findAll(decl(anyOf(recordDecl(hasName("::A")).bind("v"),
                                    fieldDecl().bind("v"))))),
      new VerifyIdIsBoundTo<Decl>("v", 3)));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { class B {}; class C {}; };",
      recordDecl(hasName("::A"), findAll(recordDecl(isDefinition()).bind("v"))),
      new VerifyIdIsBoundTo<CXXRecordDecl>("v", 3)));
}

TEST(EachOf, TriggersForEachMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { int a; int b; };",
      recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                        has(fieldDecl(hasName("b")).bind("v")))),
      new VerifyIdIsBoundTo<FieldDecl>("v", 2)));
}

TEST(EachOf, BehavesLikeAnyOfUnlessBothMatch) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { int a; int c; };",
      recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                        has(fieldDecl(hasName("b")).bind("v")))),
      new VerifyIdIsBoundTo<FieldDecl>("v", 1)));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class A { int c; int b; };",
      recordDecl(eachOf(has(fieldDecl(hasName("a")).bind("v")),
                        has(fieldDecl(hasName("b")).bind("v")))),
      new VerifyIdIsBoundTo<FieldDecl>("v", 1)));
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

TEST(HasAncenstor, MatchesDeclarationAncestors) {
  EXPECT_TRUE(matches(
      "class A { class B { class C {}; }; };",
      recordDecl(hasName("C"), hasAncestor(recordDecl(hasName("A"))))));
}

TEST(HasAncenstor, FailsIfNoAncestorMatches) {
  EXPECT_TRUE(notMatches(
      "class A { class B { class C {}; }; };",
      recordDecl(hasName("C"), hasAncestor(recordDecl(hasName("X"))))));
}

TEST(HasAncestor, MatchesDeclarationsThatGetVisitedLater) {
  EXPECT_TRUE(matches(
      "class A { class B { void f() { C c; } class C {}; }; };",
      varDecl(hasName("c"), hasType(recordDecl(hasName("C"), 
          hasAncestor(recordDecl(hasName("A"))))))));
}

TEST(HasAncenstor, MatchesStatementAncestors) {
  EXPECT_TRUE(matches(
      "void f() { if (true) { while (false) { 42; } } }",
      integerLiteral(equals(42), hasAncestor(ifStmt()))));
}

TEST(HasAncestor, DrillsThroughDifferentHierarchies) {
  EXPECT_TRUE(matches(
      "void f() { if (true) { int x = 42; } }",
      integerLiteral(equals(42), hasAncestor(functionDecl(hasName("f"))))));
}

TEST(HasAncestor, BindsRecursiveCombinations) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { class D { class E { class F { int y; }; }; }; };",
      fieldDecl(hasAncestor(recordDecl(hasAncestor(recordDecl().bind("r"))))),
      new VerifyIdIsBoundTo<CXXRecordDecl>("r", 1)));
}

TEST(HasAncestor, BindsCombinationsWithHasDescendant) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class C { class D { class E { class F { int y; }; }; }; };",
      fieldDecl(hasAncestor(
          decl(
            hasDescendant(recordDecl(isDefinition(),
                                     hasAncestor(recordDecl())))
          ).bind("d")
      )),
      new VerifyIdIsBoundTo<CXXRecordDecl>("d", "E")));
}

TEST(HasAncestor, MatchesClosestAncestor) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "template <typename T> struct C {"
      "  void f(int) {"
      "    struct I { void g(T) { int x; } } i; i.g(42);"
      "  }"
      "};"
      "template struct C<int>;",
      varDecl(hasName("x"),
              hasAncestor(functionDecl(hasParameter(
                  0, varDecl(hasType(asString("int"))))).bind("f"))).bind("v"),
      new VerifyIdIsBoundTo<FunctionDecl>("f", "g", 2)));
}

TEST(HasAncestor, MatchesInTemplateInstantiations) {
  EXPECT_TRUE(matches(
      "template <typename T> struct A { struct B { struct C { T t; }; }; }; "
      "A<int>::B::C a;",
      fieldDecl(hasType(asString("int")),
                hasAncestor(recordDecl(hasName("A"))))));
}

TEST(HasAncestor, MatchesInImplicitCode) {
  EXPECT_TRUE(matches(
      "struct X {}; struct A { A() {} X x; };",
      cxxConstructorDecl(
          hasAnyConstructorInitializer(withInitializer(expr(
              hasAncestor(recordDecl(hasName("A")))))))));
}

TEST(HasParent, MatchesOnlyParent) {
  EXPECT_TRUE(matches(
      "void f() { if (true) { int x = 42; } }",
      compoundStmt(hasParent(ifStmt()))));
  EXPECT_TRUE(notMatches(
      "void f() { for (;;) { int x = 42; } }",
      compoundStmt(hasParent(ifStmt()))));
  EXPECT_TRUE(notMatches(
      "void f() { if (true) for (;;) { int x = 42; } }",
      compoundStmt(hasParent(ifStmt()))));
}

TEST(HasAncestor, MatchesAllAncestors) {
  EXPECT_TRUE(matches(
      "template <typename T> struct C { static void f() { 42; } };"
      "void t() { C<int>::f(); }",
      integerLiteral(
          equals(42),
          allOf(
              hasAncestor(cxxRecordDecl(isTemplateInstantiation())),
              hasAncestor(cxxRecordDecl(unless(isTemplateInstantiation())))))));
}

TEST(HasAncestor, ImplicitArrayCopyCtorDeclRefExpr) {
  EXPECT_TRUE(matches("struct MyClass {\n"
                      "  int c[1];\n"
                      "  static MyClass Create() { return MyClass(); }\n"
                      "};",
                      declRefExpr(to(decl(hasAncestor(decl()))))));
}

TEST(HasAncestor, AnonymousUnionMemberExpr) {
  EXPECT_TRUE(matches("int F() {\n"
                      "  union { int i; };\n"
                      "  return i;\n"
                      "}\n",
                      memberExpr(member(hasAncestor(decl())))));
  EXPECT_TRUE(matches("void f() {\n"
                      "  struct {\n"
                      "    struct { int a; int b; };\n"
                      "  } s;\n"
                      "  s.a = 4;\n"
                      "}\n",
                      memberExpr(member(hasAncestor(decl())))));
  EXPECT_TRUE(matches("void f() {\n"
                      "  struct {\n"
                      "    struct { int a; int b; };\n"
                      "  } s;\n"
                      "  s.a = 4;\n"
                      "}\n",
                      declRefExpr(to(decl(hasAncestor(decl()))))));
}

TEST(HasParent, MatchesAllParents) {
  EXPECT_TRUE(matches(
      "template <typename T> struct C { static void f() { 42; } };"
      "void t() { C<int>::f(); }",
      integerLiteral(
          equals(42),
          hasParent(compoundStmt(hasParent(functionDecl(
              hasParent(cxxRecordDecl(isTemplateInstantiation())))))))));
  EXPECT_TRUE(
      matches("template <typename T> struct C { static void f() { 42; } };"
              "void t() { C<int>::f(); }",
              integerLiteral(
                  equals(42),
                  hasParent(compoundStmt(hasParent(functionDecl(hasParent(
                      cxxRecordDecl(unless(isTemplateInstantiation()))))))))));
  EXPECT_TRUE(matches(
      "template <typename T> struct C { static void f() { 42; } };"
      "void t() { C<int>::f(); }",
      integerLiteral(equals(42),
                     hasParent(compoundStmt(
                         allOf(hasParent(functionDecl(hasParent(
                                   cxxRecordDecl(isTemplateInstantiation())))),
                               hasParent(functionDecl(hasParent(cxxRecordDecl(
                                   unless(isTemplateInstantiation())))))))))));
  EXPECT_TRUE(
      notMatches("template <typename T> struct C { static void f() {} };"
                 "void t() { C<int>::f(); }",
                 compoundStmt(hasParent(recordDecl()))));
}

TEST(HasParent, NoDuplicateParents) {
  class HasDuplicateParents : public BoundNodesCallback {
  public:
    bool run(const BoundNodes *Nodes) override { return false; }
    bool run(const BoundNodes *Nodes, ASTContext *Context) override {
      const Stmt *Node = Nodes->getNodeAs<Stmt>("node");
      std::set<const void *> Parents;
      for (const auto &Parent : Context->getParents(*Node)) {
        if (!Parents.insert(Parent.getMemoizationData()).second) {
          return true;
        }
      }
      return false;
    }
  };
  EXPECT_FALSE(matchAndVerifyResultTrue(
      "template <typename T> int Foo() { return 1 + 2; }\n"
      "int x = Foo<int>() + Foo<unsigned>();",
      stmt().bind("node"), new HasDuplicateParents()));
}

TEST(TypeMatching, MatchesTypes) {
  EXPECT_TRUE(matches("struct S {};", qualType().bind("loc")));
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

TEST(TypeMatching, MatchesConstantArrayTypes) {
  EXPECT_TRUE(matches("int a[2];", constantArrayType()));
  EXPECT_TRUE(notMatches(
    "void f() { int a[] = { 2, 3 }; int b[a[0]]; }",
    constantArrayType(hasElementType(builtinType()))));

  EXPECT_TRUE(matches("int a[42];", constantArrayType(hasSize(42))));
  EXPECT_TRUE(matches("int b[2*21];", constantArrayType(hasSize(42))));
  EXPECT_TRUE(notMatches("int c[41], d[43];", constantArrayType(hasSize(42))));
}

TEST(TypeMatching, MatchesDependentSizedArrayTypes) {
  EXPECT_TRUE(matches(
    "template <typename T, int Size> class array { T data[Size]; };",
    dependentSizedArrayType()));
  EXPECT_TRUE(notMatches(
    "int a[42]; int b[] = { 2, 3 }; void f() { int c[b[0]]; }",
    dependentSizedArrayType()));
}

TEST(TypeMatching, MatchesIncompleteArrayType) {
  EXPECT_TRUE(matches("int a[] = { 2, 3 };", incompleteArrayType()));
  EXPECT_TRUE(matches("void f(int a[]) {}", incompleteArrayType()));

  EXPECT_TRUE(notMatches("int a[42]; void f() { int b[a[0]]; }",
                         incompleteArrayType()));
}

TEST(TypeMatching, MatchesVariableArrayType) {
  EXPECT_TRUE(matches("void f(int b) { int a[b]; }", variableArrayType()));
  EXPECT_TRUE(notMatches("int a[] = {2, 3}; int b[42];", variableArrayType()));
  
  EXPECT_TRUE(matches(
    "void f(int b) { int a[b]; }",
    variableArrayType(hasSizeExpr(ignoringImpCasts(declRefExpr(to(
      varDecl(hasName("b")))))))));
}

TEST(TypeMatching, MatchesAtomicTypes) {
  if (llvm::Triple(llvm::sys::getDefaultTargetTriple()).getOS() !=
      llvm::Triple::Win32) {
    // FIXME: Make this work for MSVC.
    EXPECT_TRUE(matches("_Atomic(int) i;", atomicType()));

    EXPECT_TRUE(matches("_Atomic(int) i;",
                        atomicType(hasValueType(isInteger()))));
    EXPECT_TRUE(notMatches("_Atomic(float) f;",
                           atomicType(hasValueType(isInteger()))));
  }
}

TEST(TypeMatching, MatchesAutoTypes) {
  EXPECT_TRUE(matches("auto i = 2;", autoType()));
  EXPECT_TRUE(matches("int v[] = { 2, 3 }; void f() { for (int i : v) {} }",
                      autoType()));

  // FIXME: Matching against the type-as-written can't work here, because the
  //        type as written was not deduced.
  //EXPECT_TRUE(matches("auto a = 1;",
  //                    autoType(hasDeducedType(isInteger()))));
  //EXPECT_TRUE(notMatches("auto b = 2.0;",
  //                       autoType(hasDeducedType(isInteger()))));
}

TEST(TypeMatching, MatchesFunctionTypes) {
  EXPECT_TRUE(matches("int (*f)(int);", functionType()));
  EXPECT_TRUE(matches("void f(int i) {}", functionType()));
}

TEST(TypeMatching, MatchesFunctionProtoTypes) {
  EXPECT_TRUE(matches("int (*f)(int);", functionProtoType()));
  EXPECT_TRUE(matches("void f(int i);", functionProtoType()));
  EXPECT_TRUE(matches("void f();", functionProtoType(parameterCountIs(0))));
  EXPECT_TRUE(notMatchesC("void f();", functionProtoType()));
  EXPECT_TRUE(
      matchesC("void f(void);", functionProtoType(parameterCountIs(0))));
}

TEST(TypeMatching, MatchesParenType) {
  EXPECT_TRUE(
      matches("int (*array)[4];", varDecl(hasType(pointsTo(parenType())))));
  EXPECT_TRUE(notMatches("int *array[4];", varDecl(hasType(parenType()))));

  EXPECT_TRUE(matches(
      "int (*ptr_to_func)(int);",
      varDecl(hasType(pointsTo(parenType(innerType(functionType())))))));
  EXPECT_TRUE(notMatches(
      "int (*ptr_to_array)[4];",
      varDecl(hasType(pointsTo(parenType(innerType(functionType())))))));
}

TEST(TypeMatching, PointerTypes) {
  // FIXME: Reactive when these tests can be more specific (not matching
  // implicit code on certain platforms), likely when we have hasDescendant for
  // Types/TypeLocs.
  //EXPECT_TRUE(matchAndVerifyResultTrue(
  //    "int* a;",
  //    pointerTypeLoc(pointeeLoc(typeLoc().bind("loc"))),
  //    new VerifyIdIsBoundTo<TypeLoc>("loc", 1)));
  //EXPECT_TRUE(matchAndVerifyResultTrue(
  //    "int* a;",
  //    pointerTypeLoc().bind("loc"),
  //    new VerifyIdIsBoundTo<TypeLoc>("loc", 1)));
  EXPECT_TRUE(matches(
      "int** a;",
      loc(pointerType(pointee(qualType())))));
  EXPECT_TRUE(matches(
      "int** a;",
      loc(pointerType(pointee(pointerType())))));
  EXPECT_TRUE(matches(
      "int* b; int* * const a = &b;",
      loc(qualType(isConstQualified(), pointerType()))));

  std::string Fragment = "struct A { int i; }; int A::* ptr = &A::i;";
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ptr"),
                                           hasType(blockPointerType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("ptr"),
                                        hasType(memberPointerType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ptr"),
                                           hasType(pointerType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ptr"),
                                           hasType(referenceType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ptr"),
                                           hasType(lValueReferenceType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ptr"),
                                           hasType(rValueReferenceType()))));

  Fragment = "int *ptr;";
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ptr"),
                                           hasType(blockPointerType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ptr"),
                                           hasType(memberPointerType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("ptr"),
                                        hasType(pointerType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ptr"),
                                           hasType(referenceType()))));

  Fragment = "int a; int &ref = a;";
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ref"),
                                           hasType(blockPointerType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ref"),
                                           hasType(memberPointerType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ref"),
                                           hasType(pointerType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("ref"),
                                        hasType(referenceType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("ref"),
                                        hasType(lValueReferenceType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ref"),
                                           hasType(rValueReferenceType()))));

  Fragment = "int &&ref = 2;";
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ref"),
                                           hasType(blockPointerType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ref"),
                                           hasType(memberPointerType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ref"),
                                           hasType(pointerType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("ref"),
                                        hasType(referenceType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("ref"),
                                           hasType(lValueReferenceType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("ref"),
                                        hasType(rValueReferenceType()))));
}

TEST(TypeMatching, AutoRefTypes) {
  std::string Fragment = "auto a = 1;"
                         "auto b = a;"
                         "auto &c = a;"
                         "auto &&d = c;"
                         "auto &&e = 2;";
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("a"),
                                           hasType(referenceType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("b"),
                                           hasType(referenceType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("c"),
                                        hasType(referenceType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("c"),
                                        hasType(lValueReferenceType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("c"),
                                           hasType(rValueReferenceType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("d"),
                                        hasType(referenceType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("d"),
                                        hasType(lValueReferenceType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("d"),
                                           hasType(rValueReferenceType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("e"),
                                        hasType(referenceType()))));
  EXPECT_TRUE(notMatches(Fragment, varDecl(hasName("e"),
                                           hasType(lValueReferenceType()))));
  EXPECT_TRUE(matches(Fragment, varDecl(hasName("e"),
                                        hasType(rValueReferenceType()))));
}

TEST(TypeMatching, PointeeTypes) {
  EXPECT_TRUE(matches("int b; int &a = b;",
                      referenceType(pointee(builtinType()))));
  EXPECT_TRUE(matches("int *a;", pointerType(pointee(builtinType()))));

  EXPECT_TRUE(matches("int *a;",
                      loc(pointerType(pointee(builtinType())))));

  EXPECT_TRUE(matches(
      "int const *A;",
      pointerType(pointee(isConstQualified(), builtinType()))));
  EXPECT_TRUE(notMatches(
      "int *A;",
      pointerType(pointee(isConstQualified(), builtinType()))));
}

TEST(TypeMatching, MatchesPointersToConstTypes) {
  EXPECT_TRUE(matches("int b; int * const a = &b;",
                      loc(pointerType())));
  EXPECT_TRUE(matches("int b; int * const a = &b;",
                      loc(pointerType())));
  EXPECT_TRUE(matches(
      "int b; const int * a = &b;",
      loc(pointerType(pointee(builtinType())))));
  EXPECT_TRUE(matches(
      "int b; const int * a = &b;",
      pointerType(pointee(builtinType()))));
}

TEST(TypeMatching, MatchesTypedefTypes) {
  EXPECT_TRUE(matches("typedef int X; X a;", varDecl(hasName("a"),
                                                     hasType(typedefType()))));
}

TEST(TypeMatching, MatchesTemplateSpecializationType) {
  EXPECT_TRUE(matches("template <typename T> class A{}; A<int> a;",
                      templateSpecializationType()));
}

TEST(TypeMatching, MatchesRecordType) {
  EXPECT_TRUE(matches("class C{}; C c;", recordType()));
  EXPECT_TRUE(matches("struct S{}; S s;",
                      recordType(hasDeclaration(recordDecl(hasName("S"))))));
  EXPECT_TRUE(notMatches("int i;",
                         recordType(hasDeclaration(recordDecl(hasName("S"))))));
}

TEST(TypeMatching, MatchesElaboratedType) {
  EXPECT_TRUE(matches(
    "namespace N {"
    "  namespace M {"
    "    class D {};"
    "  }"
    "}"
    "N::M::D d;", elaboratedType()));
  EXPECT_TRUE(matches("class C {} c;", elaboratedType()));
  EXPECT_TRUE(notMatches("class C {}; C c;", elaboratedType()));
}

TEST(ElaboratedTypeNarrowing, hasQualifier) {
  EXPECT_TRUE(matches(
    "namespace N {"
    "  namespace M {"
    "    class D {};"
    "  }"
    "}"
    "N::M::D d;",
    elaboratedType(hasQualifier(hasPrefix(specifiesNamespace(hasName("N")))))));
  EXPECT_TRUE(notMatches(
    "namespace M {"
    "  class D {};"
    "}"
    "M::D d;",
    elaboratedType(hasQualifier(hasPrefix(specifiesNamespace(hasName("N")))))));
  EXPECT_TRUE(notMatches(
    "struct D {"
    "} d;",
    elaboratedType(hasQualifier(nestedNameSpecifier()))));
}

TEST(ElaboratedTypeNarrowing, namesType) {
  EXPECT_TRUE(matches(
    "namespace N {"
    "  namespace M {"
    "    class D {};"
    "  }"
    "}"
    "N::M::D d;",
    elaboratedType(elaboratedType(namesType(recordType(
        hasDeclaration(namedDecl(hasName("D")))))))));
  EXPECT_TRUE(notMatches(
    "namespace M {"
    "  class D {};"
    "}"
    "M::D d;",
    elaboratedType(elaboratedType(namesType(typedefType())))));
}

TEST(TypeMatching, MatchesSubstTemplateTypeParmType) {
  const std::string code = "template <typename T>"
                           "int F() {"
                           "  return 1 + T();"
                           "}"
                           "int i = F<int>();";
  EXPECT_FALSE(matches(code, binaryOperator(hasLHS(
                                 expr(hasType(substTemplateTypeParmType()))))));
  EXPECT_TRUE(matches(code, binaryOperator(hasRHS(
                                expr(hasType(substTemplateTypeParmType()))))));
}

TEST(NNS, MatchesNestedNameSpecifiers) {
  EXPECT_TRUE(matches("namespace ns { struct A {}; } ns::A a;",
                      nestedNameSpecifier()));
  EXPECT_TRUE(matches("template <typename T> class A { typename T::B b; };",
                      nestedNameSpecifier()));
  EXPECT_TRUE(matches("struct A { void f(); }; void A::f() {}",
                      nestedNameSpecifier()));
  EXPECT_TRUE(matches("namespace a { namespace b {} } namespace ab = a::b;",
                      nestedNameSpecifier()));

  EXPECT_TRUE(matches(
    "struct A { static void f() {} }; void g() { A::f(); }",
    nestedNameSpecifier()));
  EXPECT_TRUE(notMatches(
    "struct A { static void f() {} }; void g(A* a) { a->f(); }",
    nestedNameSpecifier()));
}

TEST(NullStatement, SimpleCases) {
  EXPECT_TRUE(matches("void f() {int i;;}", nullStmt()));
  EXPECT_TRUE(notMatches("void f() {int i;}", nullStmt()));
}

TEST(NS, Anonymous) {
  EXPECT_TRUE(notMatches("namespace N {}", namespaceDecl(isAnonymous())));
  EXPECT_TRUE(matches("namespace {}", namespaceDecl(isAnonymous())));
}

TEST(NS, Alias) {
  EXPECT_TRUE(matches("namespace test {} namespace alias = ::test;",
                      namespaceAliasDecl(hasName("alias"))));
}

TEST(NNS, MatchesTypes) {
  NestedNameSpecifierMatcher Matcher = nestedNameSpecifier(
    specifiesType(hasDeclaration(recordDecl(hasName("A")))));
  EXPECT_TRUE(matches("struct A { struct B {}; }; A::B b;", Matcher));
  EXPECT_TRUE(matches("struct A { struct B { struct C {}; }; }; A::B::C c;",
                      Matcher));
  EXPECT_TRUE(notMatches("namespace A { struct B {}; } A::B b;", Matcher));
}

TEST(NNS, MatchesNamespaceDecls) {
  NestedNameSpecifierMatcher Matcher = nestedNameSpecifier(
    specifiesNamespace(hasName("ns")));
  EXPECT_TRUE(matches("namespace ns { struct A {}; } ns::A a;", Matcher));
  EXPECT_TRUE(notMatches("namespace xx { struct A {}; } xx::A a;", Matcher));
  EXPECT_TRUE(notMatches("struct ns { struct A {}; }; ns::A a;", Matcher));
}

TEST(NNS, BindsNestedNameSpecifiers) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "namespace ns { struct E { struct B {}; }; } ns::E::B b;",
      nestedNameSpecifier(specifiesType(asString("struct ns::E"))).bind("nns"),
      new VerifyIdIsBoundTo<NestedNameSpecifier>("nns", "ns::struct E::")));
}

TEST(NNS, BindsNestedNameSpecifierLocs) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "namespace ns { struct B {}; } ns::B b;",
      loc(nestedNameSpecifier()).bind("loc"),
      new VerifyIdIsBoundTo<NestedNameSpecifierLoc>("loc", 1)));
}

TEST(NNS, MatchesNestedNameSpecifierPrefixes) {
  EXPECT_TRUE(matches(
      "struct A { struct B { struct C {}; }; }; A::B::C c;",
      nestedNameSpecifier(hasPrefix(specifiesType(asString("struct A"))))));
  EXPECT_TRUE(matches(
      "struct A { struct B { struct C {}; }; }; A::B::C c;",
      nestedNameSpecifierLoc(hasPrefix(
          specifiesTypeLoc(loc(qualType(asString("struct A"))))))));
}

TEST(NNS, DescendantsOfNestedNameSpecifiers) {
  std::string Fragment =
      "namespace a { struct A { struct B { struct C {}; }; }; };"
      "void f() { a::A::B::C c; }";
  EXPECT_TRUE(matches(
      Fragment,
      nestedNameSpecifier(specifiesType(asString("struct a::A::B")),
                          hasDescendant(nestedNameSpecifier(
                              specifiesNamespace(hasName("a")))))));
  EXPECT_TRUE(notMatches(
      Fragment,
      nestedNameSpecifier(specifiesType(asString("struct a::A::B")),
                          has(nestedNameSpecifier(
                              specifiesNamespace(hasName("a")))))));
  EXPECT_TRUE(matches(
      Fragment,
      nestedNameSpecifier(specifiesType(asString("struct a::A")),
                          has(nestedNameSpecifier(
                              specifiesNamespace(hasName("a")))))));

  // Not really useful because a NestedNameSpecifier can af at most one child,
  // but to complete the interface.
  EXPECT_TRUE(matchAndVerifyResultTrue(
      Fragment,
      nestedNameSpecifier(specifiesType(asString("struct a::A::B")),
                          forEach(nestedNameSpecifier().bind("x"))),
      new VerifyIdIsBoundTo<NestedNameSpecifier>("x", 1)));
}

TEST(NNS, NestedNameSpecifiersAsDescendants) {
  std::string Fragment =
      "namespace a { struct A { struct B { struct C {}; }; }; };"
      "void f() { a::A::B::C c; }";
  EXPECT_TRUE(matches(
      Fragment,
      decl(hasDescendant(nestedNameSpecifier(specifiesType(
          asString("struct a::A")))))));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      Fragment,
      functionDecl(hasName("f"),
                   forEachDescendant(nestedNameSpecifier().bind("x"))),
      // Nested names: a, a::A and a::A::B.
      new VerifyIdIsBoundTo<NestedNameSpecifier>("x", 3)));
}

TEST(NNSLoc, DescendantsOfNestedNameSpecifierLocs) {
  std::string Fragment =
      "namespace a { struct A { struct B { struct C {}; }; }; };"
      "void f() { a::A::B::C c; }";
  EXPECT_TRUE(matches(
      Fragment,
      nestedNameSpecifierLoc(loc(specifiesType(asString("struct a::A::B"))),
                             hasDescendant(loc(nestedNameSpecifier(
                                 specifiesNamespace(hasName("a"))))))));
  EXPECT_TRUE(notMatches(
      Fragment,
      nestedNameSpecifierLoc(loc(specifiesType(asString("struct a::A::B"))),
                             has(loc(nestedNameSpecifier(
                                 specifiesNamespace(hasName("a"))))))));
  EXPECT_TRUE(matches(
      Fragment,
      nestedNameSpecifierLoc(loc(specifiesType(asString("struct a::A"))),
                             has(loc(nestedNameSpecifier(
                                 specifiesNamespace(hasName("a"))))))));

  EXPECT_TRUE(matchAndVerifyResultTrue(
      Fragment,
      nestedNameSpecifierLoc(loc(specifiesType(asString("struct a::A::B"))),
                             forEach(nestedNameSpecifierLoc().bind("x"))),
      new VerifyIdIsBoundTo<NestedNameSpecifierLoc>("x", 1)));
}

TEST(NNSLoc, NestedNameSpecifierLocsAsDescendants) {
  std::string Fragment =
      "namespace a { struct A { struct B { struct C {}; }; }; };"
      "void f() { a::A::B::C c; }";
  EXPECT_TRUE(matches(
      Fragment,
      decl(hasDescendant(loc(nestedNameSpecifier(specifiesType(
          asString("struct a::A"))))))));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      Fragment,
      functionDecl(hasName("f"),
                   forEachDescendant(nestedNameSpecifierLoc().bind("x"))),
      // Nested names: a, a::A and a::A::B.
      new VerifyIdIsBoundTo<NestedNameSpecifierLoc>("x", 3)));
}

template <typename T> class VerifyMatchOnNode : public BoundNodesCallback {
public:
  VerifyMatchOnNode(StringRef Id, const internal::Matcher<T> &InnerMatcher,
                    StringRef InnerId)
      : Id(Id), InnerMatcher(InnerMatcher), InnerId(InnerId) {
  }

  bool run(const BoundNodes *Nodes) override { return false; }

  bool run(const BoundNodes *Nodes, ASTContext *Context) override {
    const T *Node = Nodes->getNodeAs<T>(Id);
    return selectFirst<T>(InnerId, match(InnerMatcher, *Node, *Context)) !=
           nullptr;
  }
private:
  std::string Id;
  internal::Matcher<T> InnerMatcher;
  std::string InnerId;
};

TEST(MatchFinder, CanMatchDeclarationsRecursively) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { class Y {}; };", recordDecl(hasName("::X")).bind("X"),
      new VerifyMatchOnNode<clang::Decl>(
          "X", decl(hasDescendant(recordDecl(hasName("X::Y")).bind("Y"))),
          "Y")));
  EXPECT_TRUE(matchAndVerifyResultFalse(
      "class X { class Y {}; };", recordDecl(hasName("::X")).bind("X"),
      new VerifyMatchOnNode<clang::Decl>(
          "X", decl(hasDescendant(recordDecl(hasName("X::Z")).bind("Z"))),
          "Z")));
}

TEST(MatchFinder, CanMatchStatementsRecursively) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f() { if (1) { for (;;) { } } }", ifStmt().bind("if"),
      new VerifyMatchOnNode<clang::Stmt>(
          "if", stmt(hasDescendant(forStmt().bind("for"))), "for")));
  EXPECT_TRUE(matchAndVerifyResultFalse(
      "void f() { if (1) { for (;;) { } } }", ifStmt().bind("if"),
      new VerifyMatchOnNode<clang::Stmt>(
          "if", stmt(hasDescendant(declStmt().bind("decl"))), "decl")));
}

TEST(MatchFinder, CanMatchSingleNodesRecursively) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { class Y {}; };", recordDecl(hasName("::X")).bind("X"),
      new VerifyMatchOnNode<clang::Decl>(
          "X", recordDecl(has(recordDecl(hasName("X::Y")).bind("Y"))), "Y")));
  EXPECT_TRUE(matchAndVerifyResultFalse(
      "class X { class Y {}; };", recordDecl(hasName("::X")).bind("X"),
      new VerifyMatchOnNode<clang::Decl>(
          "X", recordDecl(has(recordDecl(hasName("X::Z")).bind("Z"))), "Z")));
}

template <typename T>
class VerifyAncestorHasChildIsEqual : public BoundNodesCallback {
public:
  bool run(const BoundNodes *Nodes) override { return false; }

  bool run(const BoundNodes *Nodes, ASTContext *Context) override {
    const T *Node = Nodes->getNodeAs<T>("");
    return verify(*Nodes, *Context, Node);
  }

  bool verify(const BoundNodes &Nodes, ASTContext &Context, const Stmt *Node) {
    // Use the original typed pointer to verify we can pass pointers to subtypes
    // to equalsNode.
    const T *TypedNode = cast<T>(Node);
    return selectFirst<T>(
               "", match(stmt(hasParent(
                             stmt(has(stmt(equalsNode(TypedNode)))).bind(""))),
                         *Node, Context)) != nullptr;
  }
  bool verify(const BoundNodes &Nodes, ASTContext &Context, const Decl *Node) {
    // Use the original typed pointer to verify we can pass pointers to subtypes
    // to equalsNode.
    const T *TypedNode = cast<T>(Node);
    return selectFirst<T>(
               "", match(decl(hasParent(
                             decl(has(decl(equalsNode(TypedNode)))).bind(""))),
                         *Node, Context)) != nullptr;
  }
  bool verify(const BoundNodes &Nodes, ASTContext &Context, const Type *Node) {
    // Use the original typed pointer to verify we can pass pointers to subtypes
    // to equalsNode.
    const T *TypedNode = cast<T>(Node);
    const auto *Dec = Nodes.getNodeAs<FieldDecl>("decl");
    return selectFirst<T>(
               "", match(fieldDecl(hasParent(decl(has(fieldDecl(
                             hasType(type(equalsNode(TypedNode)).bind(""))))))),
                         *Dec, Context)) != nullptr;
  }
};

TEST(IsEqualTo, MatchesNodesByIdentity) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { class Y {}; };", recordDecl(hasName("::X::Y")).bind(""),
      new VerifyAncestorHasChildIsEqual<CXXRecordDecl>()));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f() { if (true) if(true) {} }", ifStmt().bind(""),
      new VerifyAncestorHasChildIsEqual<IfStmt>()));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { class Y {} y; };",
      fieldDecl(hasName("y"), hasType(type().bind(""))).bind("decl"),
      new VerifyAncestorHasChildIsEqual<Type>()));
}

TEST(MatchFinder, CheckProfiling) {
  MatchFinder::MatchFinderOptions Options;
  llvm::StringMap<llvm::TimeRecord> Records;
  Options.CheckProfiling.emplace(Records);
  MatchFinder Finder(std::move(Options));

  struct NamedCallback : public MatchFinder::MatchCallback {
    void run(const MatchFinder::MatchResult &Result) override {}
    StringRef getID() const override { return "MyID"; }
  } Callback;
  Finder.addMatcher(decl(), &Callback);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  ASSERT_TRUE(tooling::runToolOnCode(Factory->create(), "int x;"));

  EXPECT_EQ(1u, Records.size());
  EXPECT_EQ("MyID", Records.begin()->getKey());
}

class VerifyStartOfTranslationUnit : public MatchFinder::MatchCallback {
public:
  VerifyStartOfTranslationUnit() : Called(false) {}
  void run(const MatchFinder::MatchResult &Result) override {
    EXPECT_TRUE(Called);
  }
  void onStartOfTranslationUnit() override { Called = true; }
  bool Called;
};

TEST(MatchFinder, InterceptsStartOfTranslationUnit) {
  MatchFinder Finder;
  VerifyStartOfTranslationUnit VerifyCallback;
  Finder.addMatcher(decl(), &VerifyCallback);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  ASSERT_TRUE(tooling::runToolOnCode(Factory->create(), "int x;"));
  EXPECT_TRUE(VerifyCallback.Called);

  VerifyCallback.Called = false;
  std::unique_ptr<ASTUnit> AST(tooling::buildASTFromCode("int x;"));
  ASSERT_TRUE(AST.get());
  Finder.matchAST(AST->getASTContext());
  EXPECT_TRUE(VerifyCallback.Called);
}

class VerifyEndOfTranslationUnit : public MatchFinder::MatchCallback {
public:
  VerifyEndOfTranslationUnit() : Called(false) {}
  void run(const MatchFinder::MatchResult &Result) override {
    EXPECT_FALSE(Called);
  }
  void onEndOfTranslationUnit() override { Called = true; }
  bool Called;
};

TEST(MatchFinder, InterceptsEndOfTranslationUnit) {
  MatchFinder Finder;
  VerifyEndOfTranslationUnit VerifyCallback;
  Finder.addMatcher(decl(), &VerifyCallback);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  ASSERT_TRUE(tooling::runToolOnCode(Factory->create(), "int x;"));
  EXPECT_TRUE(VerifyCallback.Called);

  VerifyCallback.Called = false;
  std::unique_ptr<ASTUnit> AST(tooling::buildASTFromCode("int x;"));
  ASSERT_TRUE(AST.get());
  Finder.matchAST(AST->getASTContext());
  EXPECT_TRUE(VerifyCallback.Called);
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
      new VerifyIdIsBoundTo<VarDecl>("decl", 2)));
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
      new VerifyIdIsBoundTo<VarDecl>("d", 5)));
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
      new VerifyIdIsBoundTo<Expr>("data", 1)));

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

TEST(TypeDefDeclMatcher, Match) {
  EXPECT_TRUE(matches("typedef int typedefDeclTest;",
                      typedefDecl(hasName("typedefDeclTest"))));
}

TEST(IsInlineMatcher, IsInline) {
  EXPECT_TRUE(matches("void g(); inline void f();",
                      functionDecl(isInline(), hasName("f"))));
  EXPECT_TRUE(matches("namespace n { inline namespace m {} }",
                      namespaceDecl(isInline(), hasName("m"))));
}

// FIXME: Figure out how to specify paths so the following tests pass on
// Windows.
#ifndef LLVM_ON_WIN32

TEST(Matcher, IsExpansionInMainFileMatcher) {
  EXPECT_TRUE(matches("class X {};",
                      recordDecl(hasName("X"), isExpansionInMainFile())));
  EXPECT_TRUE(notMatches("", recordDecl(isExpansionInMainFile())));
  FileContentMappings M;
  M.push_back(std::make_pair("/other", "class X {};"));
  EXPECT_TRUE(matchesConditionally("#include <other>\n",
                                   recordDecl(isExpansionInMainFile()), false,
                                   "-isystem/", M));
}

TEST(Matcher, IsExpansionInSystemHeader) {
  FileContentMappings M;
  M.push_back(std::make_pair("/other", "class X {};"));
  EXPECT_TRUE(matchesConditionally(
      "#include \"other\"\n", recordDecl(isExpansionInSystemHeader()), true,
      "-isystem/", M));
  EXPECT_TRUE(matchesConditionally("#include \"other\"\n",
                                   recordDecl(isExpansionInSystemHeader()),
                                   false, "-I/", M));
  EXPECT_TRUE(notMatches("class X {};",
                         recordDecl(isExpansionInSystemHeader())));
  EXPECT_TRUE(notMatches("", recordDecl(isExpansionInSystemHeader())));
}

TEST(Matcher, IsExpansionInFileMatching) {
  FileContentMappings M;
  M.push_back(std::make_pair("/foo", "class A {};"));
  M.push_back(std::make_pair("/bar", "class B {};"));
  EXPECT_TRUE(matchesConditionally(
      "#include <foo>\n"
      "#include <bar>\n"
      "class X {};",
      recordDecl(isExpansionInFileMatching("b.*"), hasName("B")), true,
      "-isystem/", M));
  EXPECT_TRUE(matchesConditionally(
      "#include <foo>\n"
      "#include <bar>\n"
      "class X {};",
      recordDecl(isExpansionInFileMatching("f.*"), hasName("X")), false,
      "-isystem/", M));
}

#endif // LLVM_ON_WIN32

  
TEST(ObjCMessageExprMatcher, SimpleExprs) {
  // don't find ObjCMessageExpr where none are present
  EXPECT_TRUE(notMatchesObjC("", objcMessageExpr(anything())));
 
  std::string Objc1String =
  "@interface Str "
  " - (Str *)uppercaseString:(Str *)str;"
  "@end "
  "@interface foo "
  "- (void)meth:(Str *)text;"
  "@end "
  " "
  "@implementation foo "
  "- (void) meth:(Str *)text { "
  "  [self contents];"
  "  Str *up = [text uppercaseString];"
  "} "
  "@end ";
  EXPECT_TRUE(matchesObjC(
      Objc1String,
      objcMessageExpr(anything())));
  EXPECT_TRUE(matchesObjC(
      Objc1String,
      objcMessageExpr(hasSelector("contents"))));
  EXPECT_TRUE(matchesObjC(
      Objc1String,
      objcMessageExpr(matchesSelector("cont*"))));
  EXPECT_FALSE(matchesObjC(
      Objc1String,
      objcMessageExpr(matchesSelector("?cont*"))));
  EXPECT_TRUE(notMatchesObjC(
      Objc1String,
      objcMessageExpr(hasSelector("contents"), hasNullSelector())));
  EXPECT_TRUE(matchesObjC(
      Objc1String,
      objcMessageExpr(hasSelector("contents"), hasUnarySelector())));
  EXPECT_TRUE(matchesObjC(
      Objc1String,
      objcMessageExpr(hasSelector("contents"), numSelectorArgs(0))));
  EXPECT_TRUE(matchesObjC(
      Objc1String,
      objcMessageExpr(matchesSelector("uppercase*"),
                      argumentCountIs(0)
                      )));
  
}

} // end namespace ast_matchers
} // end namespace clang
