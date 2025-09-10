//== unittests/ASTMatchers/ASTMatchersNodeTest.cpp - AST matcher unit tests ==//
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
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

TEST_P(ASTMatchersTest, Decl_CXX) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `decl()` that does not depend on C++.
    return;
  }
  EXPECT_TRUE(notMatches("", decl(usingDecl())));
  EXPECT_TRUE(
      matches("namespace x { class X {}; } using x::X;", decl(usingDecl())));
}

TEST_P(ASTMatchersTest, NameableDeclaration_MatchesVariousDecls) {
  DeclarationMatcher NamedX = namedDecl(hasName("X"));
  EXPECT_TRUE(matches("typedef int X;", NamedX));
  EXPECT_TRUE(matches("int X;", NamedX));
  EXPECT_TRUE(matches("void foo() { int X; }", NamedX));
  EXPECT_TRUE(matches("enum X { A, B, C };", NamedX));

  EXPECT_TRUE(notMatches("#define X 1", NamedX));
}

TEST_P(ASTMatchersTest, NamedDecl_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  DeclarationMatcher NamedX = namedDecl(hasName("X"));
  EXPECT_TRUE(matches("class foo { virtual void X(); };", NamedX));
  EXPECT_TRUE(matches("void foo() try { } catch(int X) { }", NamedX));
  EXPECT_TRUE(matches("namespace X { }", NamedX));
}

TEST_P(ASTMatchersTest, MatchesNameRE) {
  DeclarationMatcher NamedX = namedDecl(matchesName("::X"));
  EXPECT_TRUE(matches("typedef int Xa;", NamedX));
  EXPECT_TRUE(matches("int Xb;", NamedX));
  EXPECT_TRUE(matches("void foo() { int Xgh; }", NamedX));
  EXPECT_TRUE(matches("enum X { A, B, C };", NamedX));

  EXPECT_TRUE(notMatches("#define Xkl 1", NamedX));

  DeclarationMatcher StartsWithNo = namedDecl(matchesName("::no"));
  EXPECT_TRUE(matches("int no_foo;", StartsWithNo));

  DeclarationMatcher Abc = namedDecl(matchesName("a.*b.*c"));
  EXPECT_TRUE(matches("int abc;", Abc));
  EXPECT_TRUE(matches("int aFOObBARc;", Abc));
  EXPECT_TRUE(notMatches("int cab;", Abc));
  EXPECT_TRUE(matches("int cabc;", Abc));

  DeclarationMatcher StartsWithK = namedDecl(matchesName(":k[^:]*$"));
  EXPECT_TRUE(matches("int k;", StartsWithK));
  EXPECT_TRUE(matches("int kAbc;", StartsWithK));
}

TEST_P(ASTMatchersTest, MatchesNameRE_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  DeclarationMatcher NamedX = namedDecl(matchesName("::X"));
  EXPECT_TRUE(matches("class foo { virtual void Xc(); };", NamedX));
  EXPECT_TRUE(matches("void foo() try { } catch(int Xdef) { }", NamedX));
  EXPECT_TRUE(matches("namespace Xij { }", NamedX));

  DeclarationMatcher StartsWithNo = namedDecl(matchesName("::no"));
  EXPECT_TRUE(matches("class foo { virtual void nobody(); };", StartsWithNo));

  DeclarationMatcher StartsWithK = namedDecl(matchesName(":k[^:]*$"));
  EXPECT_TRUE(matches("namespace x { int kTest; }", StartsWithK));
  EXPECT_TRUE(matches("class C { int k; };", StartsWithK));
  EXPECT_TRUE(notMatches("class C { int ckc; };", StartsWithK));
  EXPECT_TRUE(notMatches("int K;", StartsWithK));

  DeclarationMatcher StartsWithKIgnoreCase =
      namedDecl(matchesName(":k[^:]*$", llvm::Regex::IgnoreCase));
  EXPECT_TRUE(matches("int k;", StartsWithKIgnoreCase));
  EXPECT_TRUE(matches("int K;", StartsWithKIgnoreCase));
}

TEST_P(ASTMatchersTest, DeclarationMatcher_MatchClass) {
  if (!GetParam().isCXX()) {
    return;
  }

  DeclarationMatcher ClassX = recordDecl(recordDecl(hasName("X")));
  EXPECT_TRUE(matches("class X;", ClassX));
  EXPECT_TRUE(matches("class X {};", ClassX));
  EXPECT_TRUE(matches("template<class T> class X {};", ClassX));
  EXPECT_TRUE(notMatches("", ClassX));
}

TEST_P(ASTMatchersTest, TranslationUnitDecl) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `translationUnitDecl()` that does not depend on
    // C++.
    return;
  }
  StringRef Code = "int MyVar1;\n"
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

TEST_P(ASTMatchersTest, LinkageSpecDecl) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("extern \"C\" { void foo() {}; }", linkageSpecDecl()));
  EXPECT_TRUE(notMatches("void foo() {};", linkageSpecDecl()));
}

TEST_P(ASTMatchersTest, ClassTemplateDecl_DoesNotMatchClass) {
  if (!GetParam().isCXX()) {
    return;
  }
  DeclarationMatcher ClassX = classTemplateDecl(hasName("X"));
  EXPECT_TRUE(notMatches("class X;", ClassX));
  EXPECT_TRUE(notMatches("class X {};", ClassX));
}

TEST_P(ASTMatchersTest, ClassTemplateDecl_MatchesClassTemplate) {
  if (!GetParam().isCXX()) {
    return;
  }
  DeclarationMatcher ClassX = classTemplateDecl(hasName("X"));
  EXPECT_TRUE(matches("template<typename T> class X {};", ClassX));
  EXPECT_TRUE(matches("class Z { template<class T> class X {}; };", ClassX));
}

TEST_P(ASTMatchersTest,
       ClassTemplateDecl_DoesNotMatchClassTemplateExplicitSpecialization) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches(
      "template<typename T> class X { };"
      "template<> class X<int> { int a; };",
      classTemplateDecl(hasName("X"), hasDescendant(fieldDecl(hasName("a"))))));
}

TEST_P(ASTMatchersTest,
       ClassTemplateDecl_DoesNotMatchClassTemplatePartialSpecialization) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches(
      "template<typename T, typename U> class X { };"
      "template<typename T> class X<T, int> { int a; };",
      classTemplateDecl(hasName("X"), hasDescendant(fieldDecl(hasName("a"))))));
}

TEST(ASTMatchersTestCUDA, CUDAKernelCallExpr) {
  EXPECT_TRUE(matchesWithCuda("__global__ void f() { }"
                              "void g() { f<<<1, 2>>>(); }",
                              cudaKernelCallExpr()));
  EXPECT_TRUE(notMatchesWithCuda("void f() {}", cudaKernelCallExpr()));
}

TEST(ASTMatchersTestCUDA, HasAttrCUDA) {
  EXPECT_TRUE(matchesWithCuda("__attribute__((device)) void f() {}",
                              hasAttr(clang::attr::CUDADevice)));
  EXPECT_FALSE(notMatchesWithCuda("__attribute__((global)) void f() {}",
                                  hasAttr(clang::attr::CUDAGlobal)));
}

TEST_P(ASTMatchersTest, ExportDecl) {
  if (!GetParam().isCXX20OrLater()) {
    return;
  }
  const std::string moduleHeader = "module;export module ast_matcher_test;";
  EXPECT_TRUE(matches(moduleHeader + "export void foo();",
                      exportDecl(has(functionDecl()))));
  EXPECT_TRUE(matches(moduleHeader + "export { void foo(); int v; }",
                      exportDecl(has(functionDecl()))));
  EXPECT_TRUE(matches(moduleHeader + "export { void foo(); int v; }",
                      exportDecl(has(varDecl()))));
  EXPECT_TRUE(matches(moduleHeader + "export namespace aa { void foo(); }",
                      exportDecl(has(namespaceDecl()))));
}

TEST_P(ASTMatchersTest, ValueDecl) {
  if (!GetParam().isCXX()) {
    // FIXME: Fix this test in non-C++ language modes.
    return;
  }
  EXPECT_TRUE(matches("enum EnumType { EnumValue };",
                      valueDecl(hasType(asString("enum EnumType")))));
  EXPECT_TRUE(matches("void FunctionDecl();",
                      valueDecl(hasType(asString("void (void)")))));
}

TEST_P(ASTMatchersTest, FriendDecl) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class Y { friend class X; };",
                      friendDecl(hasType(asString("class X")))));
  EXPECT_TRUE(matches("class Y { friend class X; };",
                      friendDecl(hasType(recordDecl(hasName("X"))))));

  EXPECT_TRUE(matches("class Y { friend void f(); };",
                      functionDecl(hasName("f"), hasParent(friendDecl()))));
}

TEST_P(ASTMatchersTest, EnumDecl_DoesNotMatchClasses) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class X {};", enumDecl(hasName("X"))));
}

TEST_P(ASTMatchersTest, EnumDecl_MatchesEnums) {
  if (!GetParam().isCXX()) {
    // FIXME: Fix this test in non-C++ language modes.
    return;
  }
  EXPECT_TRUE(matches("enum X {};", enumDecl(hasName("X"))));
}

TEST_P(ASTMatchersTest, EnumConstantDecl) {
  if (!GetParam().isCXX()) {
    // FIXME: Fix this test in non-C++ language modes.
    return;
  }
  DeclarationMatcher Matcher = enumConstantDecl(hasName("A"));
  EXPECT_TRUE(matches("enum X{ A };", Matcher));
  EXPECT_TRUE(notMatches("enum X{ B };", Matcher));
  EXPECT_TRUE(notMatches("enum X {};", Matcher));
}

TEST_P(ASTMatchersTest, TagDecl) {
  if (!GetParam().isCXX()) {
    // FIXME: Fix this test in non-C++ language modes.
    return;
  }
  EXPECT_TRUE(matches("struct X {};", tagDecl(hasName("X"))));
  EXPECT_TRUE(matches("union U {};", tagDecl(hasName("U"))));
  EXPECT_TRUE(matches("enum E {};", tagDecl(hasName("E"))));
}

TEST_P(ASTMatchersTest, TagDecl_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class C {};", tagDecl(hasName("C"))));
}

TEST_P(ASTMatchersTest, UnresolvedLookupExpr) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("template<typename T>"
                      "T foo() { T a; return a; }"
                      "template<typename T>"
                      "void bar() {"
                      "  foo<T>();"
                      "}",
                      unresolvedLookupExpr()));
}

TEST_P(ASTMatchersTest, UsesADL) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher ADLMatch = callExpr(usesADL());
  StatementMatcher ADLMatchOper = cxxOperatorCallExpr(usesADL());
  StringRef NS_Str = R"cpp(
  namespace NS {
    struct X {};
    void f(X);
    void operator+(X, X);
  }
  struct MyX {};
  void f(...);
  void operator+(MyX, MyX);
)cpp";

  auto MkStr = [&](StringRef Body) {
    return (NS_Str + "void test_fn() { " + Body + " }").str();
  };

  EXPECT_TRUE(matches(MkStr("NS::X x; f(x);"), ADLMatch));
  EXPECT_TRUE(notMatches(MkStr("NS::X x; NS::f(x);"), ADLMatch));
  EXPECT_TRUE(notMatches(MkStr("MyX x; f(x);"), ADLMatch));
  EXPECT_TRUE(notMatches(MkStr("NS::X x; using NS::f; f(x);"), ADLMatch));

  // Operator call expressions
  EXPECT_TRUE(matches(MkStr("NS::X x; x + x;"), ADLMatch));
  EXPECT_TRUE(matches(MkStr("NS::X x; x + x;"), ADLMatchOper));
  EXPECT_TRUE(notMatches(MkStr("MyX x; x + x;"), ADLMatch));
  EXPECT_TRUE(notMatches(MkStr("MyX x; x + x;"), ADLMatchOper));
  EXPECT_TRUE(matches(MkStr("NS::X x; operator+(x, x);"), ADLMatch));
  EXPECT_TRUE(notMatches(MkStr("NS::X x; NS::operator+(x, x);"), ADLMatch));
}

TEST_P(ASTMatchersTest, CallExpr_CXX) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `callExpr()` that does not depend on C++.
    return;
  }
  // FIXME: Do we want to overload Call() to directly take
  // Matcher<Decl>, too?
  StatementMatcher MethodX =
      callExpr(hasDeclaration(cxxMethodDecl(hasName("x"))));

  EXPECT_TRUE(matches("class Y { void x() { x(); } };", MethodX));
  EXPECT_TRUE(notMatches("class Y { void x() {} };", MethodX));

  StatementMatcher MethodOnY =
      cxxMemberCallExpr(on(hasType(recordDecl(hasName("Y")))));

  EXPECT_TRUE(matches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
                      MethodOnY));
  EXPECT_TRUE(matches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
                      MethodOnY));
  EXPECT_TRUE(notMatches(
      "class Y { public: void x(); }; void z(Y *&y) { y->x(); }", MethodOnY));
  EXPECT_TRUE(notMatches(
      "class Y { public: void x(); }; void z(Y y[]) { y->x(); }", MethodOnY));
  EXPECT_TRUE(notMatches(
      "class Y { public: void x(); }; void z() { Y *y; y->x(); }", MethodOnY));

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

TEST_P(ASTMatchersTest, LambdaExpr) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("auto f = [] (int i) { return i; };", lambdaExpr()));
}

TEST_P(ASTMatchersTest, CXXForRangeStmt) {
  EXPECT_TRUE(
      notMatches("void f() { for (int i; i<5; ++i); }", cxxForRangeStmt()));
}

TEST_P(ASTMatchersTest, CXXForRangeStmt_CXX11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("int as[] = { 1, 2, 3 };"
                      "void f() { for (auto &a : as); }",
                      cxxForRangeStmt()));
}

TEST_P(ASTMatchersTest, SubstNonTypeTemplateParmExpr) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_FALSE(matches("template<int N>\n"
                       "struct A {  static const int n = 0; };\n"
                       "struct B : public A<42> {};",
                       traverse(TK_AsIs, substNonTypeTemplateParmExpr())));
  EXPECT_TRUE(matches("template<int N>\n"
                      "struct A {  static const int n = N; };\n"
                      "struct B : public A<42> {};",
                      traverse(TK_AsIs, substNonTypeTemplateParmExpr())));
}

TEST_P(ASTMatchersTest, NonTypeTemplateParmDecl) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("template <int N> void f();",
                      nonTypeTemplateParmDecl(hasName("N"))));
  EXPECT_TRUE(
      notMatches("template <typename T> void f();", nonTypeTemplateParmDecl()));
}

TEST_P(ASTMatchersTest, TemplateTypeParmDecl) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("template <typename T> void f();",
                      templateTypeParmDecl(hasName("T"))));
  EXPECT_TRUE(notMatches("template <int N> void f();", templateTypeParmDecl()));
}

TEST_P(ASTMatchersTest, TemplateTemplateParmDecl) {
  if (!GetParam().isCXX())
    return;
  EXPECT_TRUE(matches("template <template <typename> class Z> void f();",
                      templateTemplateParmDecl(hasName("Z"))));
  EXPECT_TRUE(notMatches("template <typename, int> void f();",
                         templateTemplateParmDecl()));
}

TEST_P(ASTMatchersTest, UserDefinedLiteral) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("constexpr char operator \"\" _inc (const char i) {"
                      "  return i + 1;"
                      "}"
                      "char c = 'a'_inc;",
                      userDefinedLiteral()));
}

TEST_P(ASTMatchersTest, FlowControl) {
  EXPECT_TRUE(matches("void f() { while(1) { break; } }", breakStmt()));
  EXPECT_TRUE(matches("void f() { while(1) { continue; } }", continueStmt()));
  EXPECT_TRUE(matches("void f() { goto FOO; FOO: ;}", gotoStmt()));
  EXPECT_TRUE(matches("void f() { goto FOO; FOO: ;}",
                      labelStmt(hasDeclaration(labelDecl(hasName("FOO"))))));
  EXPECT_TRUE(matches("void f() { FOO: ; void *ptr = &&FOO; goto *ptr; }",
                      addrLabelExpr()));
  EXPECT_TRUE(matches("void f() { return; }", returnStmt()));
}

TEST_P(ASTMatchersTest, CXXOperatorCallExpr) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher OpCall = cxxOperatorCallExpr();
  // Unary operator
  EXPECT_TRUE(matches("class Y { }; "
                      "bool operator!(Y x) { return false; }; "
                      "Y y; bool c = !y;",
                      OpCall));
  // No match -- special operators like "new", "delete"
  // FIXME: operator new takes size_t, for which we need stddef.h, for which
  // we need to figure out include paths in the test.
  // EXPECT_TRUE(NotMatches("#include <stddef.h>\n"
  //             "class Y { }; "
  //             "void *operator new(size_t size) { return 0; } "
  //             "Y *y = new Y;", OpCall));
  EXPECT_TRUE(notMatches("class Y { }; "
                         "void operator delete(void *p) { } "
                         "void a() {Y *y = new Y; delete y;}",
                         OpCall));
  // Binary operator
  EXPECT_TRUE(matches("class Y { }; "
                      "bool operator&&(Y x, Y y) { return true; }; "
                      "Y a; Y b; bool c = a && b;",
                      OpCall));
  // No match -- normal operator, not an overloaded one.
  EXPECT_TRUE(notMatches("bool x = true, y = true; bool t = x && y;", OpCall));
  EXPECT_TRUE(notMatches("int t = 5 << 2;", OpCall));
}

TEST_P(ASTMatchersTest, FoldExpr) {
  if (!GetParam().isCXX() || !GetParam().isCXX17OrLater()) {
    return;
  }

  EXPECT_TRUE(matches("template <typename... Args> auto sum(Args... args) { "
                      "return (0 + ... + args); }",
                      cxxFoldExpr()));
  EXPECT_TRUE(matches("template <typename... Args> auto sum(Args... args) { "
                      "return (args + ...); }",
                      cxxFoldExpr()));
}

TEST_P(ASTMatchersTest, ThisPointerType) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher MethodOnY = traverse(
      TK_AsIs, cxxMemberCallExpr(thisPointerType(recordDecl(hasName("Y")))));

  EXPECT_TRUE(matches("class Y { public: void x(); }; void z() { Y y; y.x(); }",
                      MethodOnY));
  EXPECT_TRUE(matches("class Y { public: void x(); }; void z(Y &y) { y.x(); }",
                      MethodOnY));
  EXPECT_TRUE(matches(
      "class Y { public: void x(); }; void z(Y *&y) { y->x(); }", MethodOnY));
  EXPECT_TRUE(matches(
      "class Y { public: void x(); }; void z(Y y[]) { y->x(); }", MethodOnY));
  EXPECT_TRUE(matches(
      "class Y { public: void x(); }; void z() { Y *y; y->x(); }", MethodOnY));

  EXPECT_TRUE(matches("class Y {"
                      "  public: virtual void x();"
                      "};"
                      "class X : public Y {"
                      "  public: virtual void x();"
                      "};"
                      "void z() { X *x; x->Y::x(); }",
                      MethodOnY));
}

TEST_P(ASTMatchersTest, DeclRefExpr) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `declRefExpr()` that does not depend on C++.
    return;
  }
  StatementMatcher Reference = declRefExpr(to(varDecl(hasInitializer(
      cxxMemberCallExpr(thisPointerType(recordDecl(hasName("Y"))))))));

  EXPECT_TRUE(matches("class Y {"
                      " public:"
                      "  bool x() const;"
                      "};"
                      "void z(const Y &y) {"
                      "  bool b = y.x();"
                      "  if (b) {}"
                      "}",
                      Reference));

  EXPECT_TRUE(notMatches("class Y {"
                         " public:"
                         "  bool x() const;"
                         "};"
                         "void z(const Y &y) {"
                         "  bool b = y.x();"
                         "}",
                         Reference));
}

TEST_P(ASTMatchersTest, DependentScopeDeclRefExpr) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }

  EXPECT_TRUE(matches("template <class T> class X : T { void f() { T::v; } };",
                      dependentScopeDeclRefExpr()));

  EXPECT_TRUE(
      matches("template <typename T> struct S { static T Foo; };"
              "template <typename T> void declToImport() { (void)S<T>::Foo; }",
              dependentScopeDeclRefExpr()));
}

TEST_P(ASTMatchersTest, CXXMemberCallExpr) {
  if (!GetParam().isCXX()) {
    return;
  }
  StatementMatcher CallOnVariableY =
      cxxMemberCallExpr(on(declRefExpr(to(varDecl(hasName("y"))))));

  EXPECT_TRUE(matches("class Y { public: void x() { Y y; y.x(); } };",
                      CallOnVariableY));
  EXPECT_TRUE(matches("class Y { public: void x() const { Y y; y.x(); } };",
                      CallOnVariableY));
  EXPECT_TRUE(matches("class Y { public: void x(); };"
                      "class X : public Y { void z() { X y; y.x(); } };",
                      CallOnVariableY));
  EXPECT_TRUE(matches("class Y { public: void x(); };"
                      "class X : public Y { void z() { X *y; y->x(); } };",
                      CallOnVariableY));
  EXPECT_TRUE(notMatches(
      "class Y { public: void x(); };"
      "class X : public Y { void z() { unsigned long y; ((X*)y)->x(); } };",
      CallOnVariableY));
}

TEST_P(ASTMatchersTest, UnaryExprOrTypeTraitExpr) {
  EXPECT_TRUE(
      matches("void x() { int a = sizeof(a); }", unaryExprOrTypeTraitExpr()));
}

TEST_P(ASTMatchersTest, AlignOfExpr) {
  EXPECT_TRUE(
      notMatches("void x() { int a = sizeof(a); }", alignOfExpr(anything())));
  // FIXME: Uncomment once alignof is enabled.
  // EXPECT_TRUE(matches("void x() { int a = alignof(a); }",
  //                     unaryExprOrTypeTraitExpr()));
  // EXPECT_TRUE(notMatches("void x() { int a = alignof(a); }",
  //                        sizeOfExpr()));
}

TEST_P(ASTMatchersTest, MemberExpr_DoesNotMatchClasses) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class Y { void x() {} };", memberExpr()));
  EXPECT_TRUE(notMatches("class Y { void x() {} };", unresolvedMemberExpr()));
  EXPECT_TRUE(
      notMatches("class Y { void x() {} };", cxxDependentScopeMemberExpr()));
}

TEST_P(ASTMatchersTest, MemberExpr_MatchesMemberFunctionCall) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }
  EXPECT_TRUE(matches("class Y { void x() { x(); } };", memberExpr()));
  EXPECT_TRUE(matches("class Y { template <class T> void x() { x<T>(); } };",
                      unresolvedMemberExpr()));
  EXPECT_TRUE(matches("template <class T> void x() { T t; t.f(); }",
                      cxxDependentScopeMemberExpr()));
}

TEST_P(ASTMatchersTest, MemberExpr_MatchesVariable) {
  if (!GetParam().isCXX() || GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    return;
  }
  EXPECT_TRUE(
      matches("class Y { void x() { this->y; } int y; };", memberExpr()));
  EXPECT_TRUE(matches("class Y { void x() { y; } int y; };", memberExpr()));
  EXPECT_TRUE(
      matches("class Y { void x() { Y y; y.y; } int y; };", memberExpr()));
  EXPECT_TRUE(matches("template <class T>"
                      "class X : T { void f() { this->T::v; } };",
                      cxxDependentScopeMemberExpr()));
  EXPECT_TRUE(matches("template <class T> class X : T { void f() { T::v; } };",
                      dependentScopeDeclRefExpr()));
  EXPECT_TRUE(matches("template <class T> void x() { T t; t.v; }",
                      cxxDependentScopeMemberExpr()));
}

TEST_P(ASTMatchersTest, MemberExpr_MatchesStaticVariable) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class Y { void x() { this->y; } static int y; };",
                      memberExpr()));
  EXPECT_TRUE(
      notMatches("class Y { void x() { y; } static int y; };", memberExpr()));
  EXPECT_TRUE(notMatches("class Y { void x() { Y::y; } static int y; };",
                         memberExpr()));
}

TEST_P(ASTMatchersTest, FunctionDecl) {
  StatementMatcher CallFunctionF = callExpr(callee(functionDecl(hasName("f"))));

  EXPECT_TRUE(matches("void f() { f(); }", CallFunctionF));
  EXPECT_TRUE(notMatches("void f() { }", CallFunctionF));

  EXPECT_TRUE(notMatches("void f(int);", functionDecl(isVariadic())));
  EXPECT_TRUE(notMatches("void f();", functionDecl(isVariadic())));
  EXPECT_TRUE(matches("void f(int, ...);", functionDecl(parameterCountIs(1))));
}

TEST_P(ASTMatchersTest, FunctionDecl_C) {
  if (!GetParam().isC()) {
    return;
  }
  EXPECT_TRUE(notMatches("void f();", functionDecl(isVariadic())));
  EXPECT_TRUE(matches("void f();", functionDecl(parameterCountIs(0))));
}

TEST_P(ASTMatchersTest, FunctionDecl_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher CallFunctionF = callExpr(callee(functionDecl(hasName("f"))));

  if (!GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Fix this test to work with delayed template parsing.
    // Dependent contexts, but a non-dependent call.
    EXPECT_TRUE(
        matches("void f(); template <int N> void g() { f(); }", CallFunctionF));
    EXPECT_TRUE(
        matches("void f(); template <int N> struct S { void g() { f(); } };",
                CallFunctionF));
  }

  // Dependent calls don't match.
  EXPECT_TRUE(
      notMatches("void f(int); template <typename T> void g(T t) { f(t); }",
                 CallFunctionF));
  EXPECT_TRUE(
      notMatches("void f(int);"
                 "template <typename T> struct S { void g(T t) { f(t); } };",
                 CallFunctionF));

  EXPECT_TRUE(matches("void f(...);", functionDecl(isVariadic())));
  EXPECT_TRUE(matches("void f(...);", functionDecl(parameterCountIs(0))));
}

TEST_P(ASTMatchersTest, FunctionDecl_CXX11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  EXPECT_TRUE(notMatches("template <typename... Ts> void f(Ts...);",
                         functionDecl(isVariadic())));
}

TEST_P(ASTMatchersTest,
       FunctionTemplateDecl_MatchesFunctionTemplateDeclarations) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("template <typename T> void f(T t) {}",
                      functionTemplateDecl(hasName("f"))));
}

TEST_P(ASTMatchersTest, FunctionTemplate_DoesNotMatchFunctionDeclarations) {
  EXPECT_TRUE(
      notMatches("void f(double d);", functionTemplateDecl(hasName("f"))));
  EXPECT_TRUE(
      notMatches("void f(int t) {}", functionTemplateDecl(hasName("f"))));
}

TEST_P(ASTMatchersTest,
       FunctionTemplateDecl_DoesNotMatchFunctionTemplateSpecializations) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches(
      "void g(); template <typename T> void f(T t) {}"
      "template <> void f(int t) { g(); }",
      functionTemplateDecl(hasName("f"), hasDescendant(declRefExpr(to(
                                             functionDecl(hasName("g"))))))));
}

TEST_P(ASTMatchersTest, ClassTemplateSpecializationDecl) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("template<typename T> struct A {};"
                      "template<> struct A<int> {};",
                      classTemplateSpecializationDecl()));
  EXPECT_TRUE(matches("template<typename T> struct A {}; A<int> a;",
                      classTemplateSpecializationDecl()));
  EXPECT_TRUE(notMatches("template<typename T> struct A {};",
                         classTemplateSpecializationDecl()));
}

TEST_P(ASTMatchersTest, DeclaratorDecl) {
  EXPECT_TRUE(matches("int x;", declaratorDecl()));
  EXPECT_TRUE(notMatches("struct A {};", declaratorDecl()));
}

TEST_P(ASTMatchersTest, DeclaratorDecl_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("class A {};", declaratorDecl()));
}

TEST_P(ASTMatchersTest, ParmVarDecl) {
  EXPECT_TRUE(matches("void f(int x);", parmVarDecl()));
  EXPECT_TRUE(notMatches("void f();", parmVarDecl()));
}

TEST_P(ASTMatchersTest, StaticAssertDecl) {
  if (!GetParam().isCXX11OrLater())
    return;

  EXPECT_TRUE(matches("static_assert(true, \"\");", staticAssertDecl()));
  EXPECT_TRUE(
      notMatches("constexpr bool staticassert(bool B, const char *M) "
                 "{ return true; };\n void f() { staticassert(true, \"\"); }",
                 staticAssertDecl()));
}

TEST_P(ASTMatchersTest, Matcher_ConstructorCall) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher Constructor = traverse(TK_AsIs, cxxConstructExpr());

  EXPECT_TRUE(
      matches("class X { public: X(); }; void x() { X x; }", Constructor));
  EXPECT_TRUE(matches("class X { public: X(); }; void x() { X x = X(); }",
                      Constructor));
  EXPECT_TRUE(matches("class X { public: X(int); }; void x() { X x = 0; }",
                      Constructor));
  EXPECT_TRUE(matches("class X {}; void x(int) { X x; }", Constructor));
}

TEST_P(ASTMatchersTest, Match_ConstructorInitializers) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class C { int i; public: C(int ii) : i(ii) {} };",
                      cxxCtorInitializer(forField(hasName("i")))));
}

TEST_P(ASTMatchersTest, Matcher_ThisExpr) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(
      matches("struct X { int a; int f () { return a; } };", cxxThisExpr()));
  EXPECT_TRUE(
      notMatches("struct X { int f () { int a; return a; } };", cxxThisExpr()));
}

TEST_P(ASTMatchersTest, Matcher_BindTemporaryExpression) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher TempExpression = traverse(TK_AsIs, cxxBindTemporaryExpr());

  StringRef ClassString = "class string { public: string(); ~string(); }; ";

  EXPECT_TRUE(matches(
      ClassString + "string GetStringByValue();"
                    "void FunctionTakesString(string s);"
                    "void run() { FunctionTakesString(GetStringByValue()); }",
      TempExpression));

  EXPECT_TRUE(notMatches(ClassString +
                             "string* GetStringPointer(); "
                             "void FunctionTakesStringPtr(string* s);"
                             "void run() {"
                             "  string* s = GetStringPointer();"
                             "  FunctionTakesStringPtr(GetStringPointer());"
                             "  FunctionTakesStringPtr(s);"
                             "}",
                         TempExpression));

  EXPECT_TRUE(notMatches("class no_dtor {};"
                         "no_dtor GetObjByValue();"
                         "void ConsumeObj(no_dtor param);"
                         "void run() { ConsumeObj(GetObjByValue()); }",
                         TempExpression));
}

TEST_P(ASTMatchersTest, MaterializeTemporaryExpr_MatchesTemporaryCXX11CXX14) {
  if (GetParam().Language != Lang_CXX11 && GetParam().Language != Lang_CXX14) {
    return;
  }

  StatementMatcher TempExpression =
      traverse(TK_AsIs, materializeTemporaryExpr());

  EXPECT_TRUE(matches("class string { public: string(); }; "
                      "string GetStringByValue();"
                      "void FunctionTakesString(string s);"
                      "void run() { FunctionTakesString(GetStringByValue()); }",
                      TempExpression));
}

TEST_P(ASTMatchersTest, MaterializeTemporaryExpr_MatchesTemporary) {
  if (!GetParam().isCXX()) {
    return;
  }

  StringRef ClassString = "class string { public: string(); int length(); }; ";
  StatementMatcher TempExpression =
      traverse(TK_AsIs, materializeTemporaryExpr());

  EXPECT_TRUE(notMatches(ClassString +
                             "string* GetStringPointer(); "
                             "void FunctionTakesStringPtr(string* s);"
                             "void run() {"
                             "  string* s = GetStringPointer();"
                             "  FunctionTakesStringPtr(GetStringPointer());"
                             "  FunctionTakesStringPtr(s);"
                             "}",
                         TempExpression));

  EXPECT_TRUE(matches(ClassString +
                          "string GetStringByValue();"
                          "void run() { int k = GetStringByValue().length(); }",
                      TempExpression));

  EXPECT_TRUE(notMatches(ClassString + "string GetStringByValue();"
                                       "void run() { GetStringByValue(); }",
                         TempExpression));
}

TEST_P(ASTMatchersTest, Matcher_NewExpression) {
  if (!GetParam().isCXX()) {
    return;
  }

  StatementMatcher New = cxxNewExpr();

  EXPECT_TRUE(matches("class X { public: X(); }; void x() { new X; }", New));
  EXPECT_TRUE(matches("class X { public: X(); }; void x() { new X(); }", New));
  EXPECT_TRUE(
      matches("class X { public: X(int); }; void x() { new X(0); }", New));
  EXPECT_TRUE(matches("class X {}; void x(int) { new X; }", New));
}

TEST_P(ASTMatchersTest, Matcher_DeleteExpression) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(
      matches("struct A {}; void f(A* a) { delete a; }", cxxDeleteExpr()));
}

TEST_P(ASTMatchersTest, Matcher_NoexceptExpression) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  StatementMatcher NoExcept = cxxNoexceptExpr();
  EXPECT_TRUE(matches("void foo(); bool bar = noexcept(foo());", NoExcept));
  EXPECT_TRUE(
      matches("void foo() noexcept; bool bar = noexcept(foo());", NoExcept));
  EXPECT_TRUE(notMatches("void foo() noexcept;", NoExcept));
  EXPECT_TRUE(notMatches("void foo() noexcept(0+1);", NoExcept));
  EXPECT_TRUE(matches("void foo() noexcept(noexcept(1+1));", NoExcept));
}

TEST_P(ASTMatchersTest, Matcher_DefaultArgument) {
  if (!GetParam().isCXX()) {
    return;
  }
  StatementMatcher Arg = cxxDefaultArgExpr();
  EXPECT_TRUE(matches("void x(int, int = 0) { int y; x(y); }", Arg));
  EXPECT_TRUE(
      matches("class X { void x(int, int = 0) { int y; x(y); } };", Arg));
  EXPECT_TRUE(notMatches("void x(int, int = 0) { int y; x(y, 0); }", Arg));
}

TEST_P(ASTMatchersTest, StringLiteral) {
  StatementMatcher Literal = stringLiteral();
  EXPECT_TRUE(matches("const char *s = \"string\";", Literal));
  // with escaped characters
  EXPECT_TRUE(matches("const char *s = \"\x05five\";", Literal));
  // no matching -- though the data type is the same, there is no string literal
  EXPECT_TRUE(notMatches("const char s[1] = {'a'};", Literal));
}

TEST_P(ASTMatchersTest, StringLiteral_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("const wchar_t *s = L\"string\";", stringLiteral()));
}

TEST_P(ASTMatchersTest, CharacterLiteral) {
  EXPECT_TRUE(matches("const char c = 'c';", characterLiteral()));
  EXPECT_TRUE(notMatches("const char c = 0x1;", characterLiteral()));
}

TEST_P(ASTMatchersTest, CharacterLiteral_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  // wide character
  EXPECT_TRUE(matches("const char c = L'c';", characterLiteral()));
  // wide character, Hex encoded, NOT MATCHED!
  EXPECT_TRUE(notMatches("const wchar_t c = 0x2126;", characterLiteral()));
}

TEST_P(ASTMatchersTest, IntegerLiteral) {
  StatementMatcher HasIntLiteral = integerLiteral();
  EXPECT_TRUE(matches("int i = 10;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 0x1AB;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 10L;", HasIntLiteral));
  EXPECT_TRUE(matches("int i = 10U;", HasIntLiteral));

  // Non-matching cases (character literals, float and double)
  EXPECT_TRUE(notMatches("int i = L'a';",
                         HasIntLiteral)); // this is actually a character
  // literal cast to int
  EXPECT_TRUE(notMatches("int i = 'a';", HasIntLiteral));
  EXPECT_TRUE(notMatches("int i = 1e10;", HasIntLiteral));
  EXPECT_TRUE(notMatches("int i = 10.0;", HasIntLiteral));

  // Negative integers.
  EXPECT_TRUE(
      matches("int i = -10;",
              unaryOperator(hasOperatorName("-"),
                            hasUnaryOperand(integerLiteral(equals(10))))));
}

TEST_P(ASTMatchersTest, FloatLiteral) {
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

TEST_P(ASTMatchersTest, FixedPointLiterals) {
  StatementMatcher HasFixedPointLiteral = fixedPointLiteral();
  EXPECT_TRUE(matchesWithFixedpoint("_Fract i = 0.25r;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Fract i = 0.25hr;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Fract i = 0.25uhr;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Fract i = 0.25ur;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Fract i = 0.25lr;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Fract i = 0.25ulr;", HasFixedPointLiteral));
  EXPECT_TRUE(matchesWithFixedpoint("_Accum i = 1.25k;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Accum i = 1.25hk;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Accum i = 1.25uhk;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Accum i = 1.25uk;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Accum i = 1.25lk;", HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Accum i = 1.25ulk;", HasFixedPointLiteral));
  EXPECT_TRUE(matchesWithFixedpoint("_Accum decexp1 = 1.575e1k;",
                                    HasFixedPointLiteral));
  EXPECT_TRUE(
      matchesWithFixedpoint("_Accum hex = 0x1.25fp2k;", HasFixedPointLiteral));
  EXPECT_TRUE(matchesWithFixedpoint("_Sat long _Fract i = 0.25r;",
                                    HasFixedPointLiteral));
  EXPECT_TRUE(matchesWithFixedpoint("_Sat short _Accum i = 256.0k;",
                                    HasFixedPointLiteral));
  EXPECT_TRUE(matchesWithFixedpoint(
      "_Accum i = 256.0k;",
      fixedPointLiteral(equals(llvm::APInt(32, 0x800000, true)))));
  EXPECT_TRUE(matchesWithFixedpoint(
      "_Fract i = 0.25ulr;",
      fixedPointLiteral(equals(llvm::APInt(32, 0x40000000, false)))));
  EXPECT_TRUE(matchesWithFixedpoint(
      "_Fract i = 0.5hr;",
      fixedPointLiteral(equals(llvm::APInt(8, 0x40, true)))));

  EXPECT_TRUE(
      notMatchesWithFixedpoint("short _Accum i = 2u;", HasFixedPointLiteral));
  EXPECT_TRUE(
      notMatchesWithFixedpoint("short _Accum i = 2;", HasFixedPointLiteral));
  EXPECT_TRUE(
      notMatchesWithFixedpoint("_Accum i = 1.25;", HasFixedPointLiteral));
  EXPECT_TRUE(notMatchesWithFixedpoint("_Accum i = (double)(1.25 *  4.5i);",
                                       HasFixedPointLiteral));
  EXPECT_TRUE(notMatchesWithFixedpoint(
      "_Accum i = 256.0k;",
      fixedPointLiteral(equals(llvm::APInt(32, 0x800001, true)))));
  EXPECT_TRUE(notMatchesWithFixedpoint(
      "_Fract i = 0.25ulr;",
      fixedPointLiteral(equals(llvm::APInt(32, 0x40000001, false)))));
  EXPECT_TRUE(notMatchesWithFixedpoint(
      "_Fract i = 0.5hr;",
      fixedPointLiteral(equals(llvm::APInt(8, 0x41, true)))));
}

TEST_P(ASTMatchersTest, CXXNullPtrLiteralExpr) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("int* i = nullptr;", cxxNullPtrLiteralExpr()));
}

TEST_P(ASTMatchersTest, ChooseExpr) {
  EXPECT_TRUE(matches("void f() { (void)__builtin_choose_expr(1, 2, 3); }",
                      chooseExpr()));
}

TEST_P(ASTMatchersTest, ConvertVectorExpr) {
  EXPECT_TRUE(matches(
      "typedef double vector4double __attribute__((__vector_size__(32)));"
      "typedef float  vector4float  __attribute__((__vector_size__(16)));"
      "vector4float vf;"
      "void f() { (void)__builtin_convertvector(vf, vector4double); }",
      convertVectorExpr()));
  EXPECT_TRUE(notMatches("void f() { (void)__builtin_choose_expr(1, 2, 3); }",
                         convertVectorExpr()));
}

TEST_P(ASTMatchersTest, GNUNullExpr) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("int* i = __null;", gnuNullExpr()));
}

TEST_P(ASTMatchersTest, GenericSelectionExpr) {
  EXPECT_TRUE(matches("void f() { (void)_Generic(1, int: 1, float: 2.0); }",
                      genericSelectionExpr()));
}

TEST_P(ASTMatchersTest, AtomicExpr) {
  EXPECT_TRUE(matches("void foo() { int *ptr; __atomic_load_n(ptr, 1); }",
                      atomicExpr()));
}

TEST_P(ASTMatchersTest, Initializers_C99) {
  if (!GetParam().isC99OrLater()) {
    return;
  }
  EXPECT_TRUE(matches(
      "void foo() { struct point { double x; double y; };"
      "  struct point ptarray[10] = "
      "      { [2].y = 1.0, [2].x = 2.0, [0].x = 1.0 }; }",
      initListExpr(hasSyntacticForm(initListExpr(
          has(designatedInitExpr(designatorCountIs(2),
                                 hasDescendant(floatLiteral(equals(1.0))),
                                 hasDescendant(integerLiteral(equals(2))))),
          has(designatedInitExpr(designatorCountIs(2),
                                 hasDescendant(floatLiteral(equals(2.0))),
                                 hasDescendant(integerLiteral(equals(2))))),
          has(designatedInitExpr(
              designatorCountIs(2), hasDescendant(floatLiteral(equals(1.0))),
              hasDescendant(integerLiteral(equals(0))))))))));
}

TEST_P(ASTMatchersTest, Initializers_CXX) {
  if (GetParam().Language != Lang_CXX03) {
    // FIXME: Make this test pass with other C++ standard versions.
    return;
  }
  EXPECT_TRUE(matches(
      "void foo() { struct point { double x; double y; };"
      "  struct point ptarray[10] = "
      "      { [2].y = 1.0, [2].x = 2.0, [0].x = 1.0 }; }",
      initListExpr(
          has(cxxConstructExpr(requiresZeroInitialization())),
          has(initListExpr(
              hasType(asString("struct point")), has(floatLiteral(equals(1.0))),
              has(implicitValueInitExpr(hasType(asString("double")))))),
          has(initListExpr(hasType(asString("struct point")),
                           has(floatLiteral(equals(2.0))),
                           has(floatLiteral(equals(1.0))))))));
}

TEST_P(ASTMatchersTest, ParenListExpr) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(
      matches("template<typename T> class foo { void bar() { foo X(*this); } };"
              "template class foo<int>;",
              varDecl(hasInitializer(parenListExpr(has(unaryOperator()))))));
}

TEST_P(ASTMatchersTest, StmtExpr) {
  EXPECT_TRUE(matches("void declToImport() { int C = ({int X=4; X;}); }",
                      varDecl(hasInitializer(stmtExpr()))));
}

TEST_P(ASTMatchersTest, PredefinedExpr) {
  // __func__ expands as StringLiteral("foo")
  EXPECT_TRUE(matches("void foo() { __func__; }",
                      predefinedExpr(hasType(asString("const char[4]")),
                                     has(stringLiteral()))));
}

TEST_P(ASTMatchersTest, AsmStatement) {
  EXPECT_TRUE(matches("void foo() { __asm(\"mov al, 2\"); }", asmStmt()));
}

TEST_P(ASTMatchersTest, HasConditionVariableStatement) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `hasConditionVariableStatement()` that does not
    // depend on C++.
    return;
  }

  StatementMatcher IfCondition =
      ifStmt(hasConditionVariableStatement(declStmt()));

  EXPECT_TRUE(matches("void x() { if (int* a = 0) {} }", IfCondition));
  EXPECT_TRUE(notMatches("void x() { if (true) {} }", IfCondition));
  EXPECT_TRUE(notMatches("void x() { int x; if ((x = 42)) {} }", IfCondition));

  StatementMatcher SwitchCondition =
      switchStmt(hasConditionVariableStatement(declStmt()));

  EXPECT_TRUE(matches("void x() { switch (int a = 0) {} }", SwitchCondition));
  if (GetParam().isCXX17OrLater()) {
    EXPECT_TRUE(
        notMatches("void x() { switch (int a = 0; a) {} }", SwitchCondition));
  }

  StatementMatcher ForCondition =
      forStmt(hasConditionVariableStatement(declStmt()));

  EXPECT_TRUE(matches("void x() { for (; int a = 0; ) {} }", ForCondition));
  EXPECT_TRUE(notMatches("void x() { for (int a = 0; ; ) {} }", ForCondition));

  EXPECT_TRUE(matches("void x() { while (int a = 0) {} }",
                      whileStmt(hasConditionVariableStatement(declStmt()))));
}

TEST_P(ASTMatchersTest, HasCondition) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `hasCondition()` that does not depend on C++.
    return;
  }

  StatementMatcher Condition =
      ifStmt(hasCondition(cxxBoolLiteral(equals(true))));

  EXPECT_TRUE(matches("void x() { if (true) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (false) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { bool a = true; if (a) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (true || false) {} }", Condition));
  EXPECT_TRUE(notMatches("void x() { if (1) {} }", Condition));
}

TEST_P(ASTMatchersTest, ConditionalOperator) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a test for `conditionalOperator()` that does not depend on
    // C++.
    return;
  }

  StatementMatcher Conditional =
      conditionalOperator(hasCondition(cxxBoolLiteral(equals(true))),
                          hasTrueExpression(cxxBoolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true ? false : true; }", Conditional));
  EXPECT_TRUE(notMatches("void x() { false ? false : true; }", Conditional));
  EXPECT_TRUE(notMatches("void x() { true ? true : false; }", Conditional));

  StatementMatcher ConditionalFalse =
      conditionalOperator(hasFalseExpression(cxxBoolLiteral(equals(false))));

  EXPECT_TRUE(matches("void x() { true ? true : false; }", ConditionalFalse));
  EXPECT_TRUE(
      notMatches("void x() { true ? false : true; }", ConditionalFalse));

  EXPECT_TRUE(matches("void x() { true ? true : false; }", ConditionalFalse));
  EXPECT_TRUE(
      notMatches("void x() { true ? false : true; }", ConditionalFalse));
}

TEST_P(ASTMatchersTest, BinaryConditionalOperator) {
  if (!GetParam().isCXX()) {
    // FIXME: This test should work in non-C++ language modes.
    return;
  }

  StatementMatcher AlwaysOne = traverse(
      TK_AsIs, binaryConditionalOperator(
                   hasCondition(implicitCastExpr(has(opaqueValueExpr(
                       hasSourceExpression((integerLiteral(equals(1)))))))),
                   hasFalseExpression(integerLiteral(equals(0)))));

  EXPECT_TRUE(matches("void x() { 1 ?: 0; }", AlwaysOne));

  StatementMatcher FourNotFive = binaryConditionalOperator(
      hasTrueExpression(
          opaqueValueExpr(hasSourceExpression((integerLiteral(equals(4)))))),
      hasFalseExpression(integerLiteral(equals(5))));

  EXPECT_TRUE(matches("void x() { 4 ?: 5; }", FourNotFive));
}

TEST_P(ASTMatchersTest, ArraySubscriptExpr) {
  EXPECT_TRUE(
      matches("int i[2]; void f() { i[1] = 1; }", arraySubscriptExpr()));
  EXPECT_TRUE(notMatches("int i; void f() { i = 1; }", arraySubscriptExpr()));
}

TEST_P(ASTMatchersTest, ForStmt) {
  EXPECT_TRUE(matches("void f() { for(;;); }", forStmt()));
  EXPECT_TRUE(matches("void f() { if(1) for(;;); }", forStmt()));
}

TEST_P(ASTMatchersTest, ForStmt_CXX11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(notMatches("int as[] = { 1, 2, 3 };"
                         "void f() { for (auto &a : as); }",
                         forStmt()));
}

TEST_P(ASTMatchersTest, ForStmt_NoFalsePositives) {
  EXPECT_TRUE(notMatches("void f() { ; }", forStmt()));
  EXPECT_TRUE(notMatches("void f() { if(1); }", forStmt()));
}

TEST_P(ASTMatchersTest, CompoundStatement) {
  EXPECT_TRUE(notMatches("void f();", compoundStmt()));
  EXPECT_TRUE(matches("void f() {}", compoundStmt()));
  EXPECT_TRUE(matches("void f() {{}}", compoundStmt()));
}

TEST_P(ASTMatchersTest, CompoundStatement_DoesNotMatchEmptyStruct) {
  if (!GetParam().isCXX()) {
    // FIXME: Add a similar test that does not depend on C++.
    return;
  }
  // It's not a compound statement just because there's "{}" in the source
  // text. This is an AST search, not grep.
  EXPECT_TRUE(notMatches("namespace n { struct S {}; }", compoundStmt()));
  EXPECT_TRUE(
      matches("namespace n { struct S { void f() {{}} }; }", compoundStmt()));
}

TEST_P(ASTMatchersTest, CastExpr_MatchesExplicitCasts) {
  EXPECT_TRUE(matches("void *p = (void *)(&p);", castExpr()));
}

TEST_P(ASTMatchersTest, CastExpr_MatchesExplicitCasts_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("char *p = reinterpret_cast<char *>(&p);", castExpr()));
  EXPECT_TRUE(matches("char q, *p = const_cast<char *>(&q);", castExpr()));
  EXPECT_TRUE(matches("char c = char(0);", castExpr()));
}

TEST_P(ASTMatchersTest, CastExpression_MatchesImplicitCasts) {
  // This test creates an implicit cast from int to char.
  EXPECT_TRUE(matches("char c = 0;", traverse(TK_AsIs, castExpr())));
  // This test creates an implicit cast from lvalue to rvalue.
  EXPECT_TRUE(matches("void f() { char c = 0, d = c; }",
                      traverse(TK_AsIs, castExpr())));
}

TEST_P(ASTMatchersTest, CastExpr_DoesNotMatchNonCasts) {
  if (GetParam().isC()) {
    // This does have a cast in C
    EXPECT_TRUE(matches("char c = '0';", implicitCastExpr()));
  } else {
    EXPECT_TRUE(notMatches("char c = '0';", castExpr()));
  }
  EXPECT_TRUE(notMatches("int i = (0);", castExpr()));
  EXPECT_TRUE(notMatches("int i = 0;", castExpr()));
}

TEST_P(ASTMatchersTest, CastExpr_DoesNotMatchNonCasts_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("char c, &q = c;", castExpr()));
}

TEST_P(ASTMatchersTest, CXXReinterpretCastExpr) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("char* p = reinterpret_cast<char*>(&p);",
                      cxxReinterpretCastExpr()));
}

TEST_P(ASTMatchersTest, CXXReinterpretCastExpr_DoesNotMatchOtherCasts) {
  if (!GetParam().isCXX()) {
    return;
  }
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

TEST_P(ASTMatchersTest, CXXFunctionalCastExpr_MatchesSimpleCase) {
  if (!GetParam().isCXX()) {
    return;
  }
  StringRef foo_class = "class Foo { public: Foo(const char*); };";
  EXPECT_TRUE(matches(foo_class + "void r() { Foo f = Foo(\"hello world\"); }",
                      cxxFunctionalCastExpr()));
}

TEST_P(ASTMatchersTest, CXXFunctionalCastExpr_DoesNotMatchOtherCasts) {
  if (!GetParam().isCXX()) {
    return;
  }
  StringRef FooClass = "class Foo { public: Foo(const char*); };";
  EXPECT_TRUE(
      notMatches(FooClass + "void r() { Foo f = (Foo) \"hello world\"; }",
                 cxxFunctionalCastExpr()));
  EXPECT_TRUE(notMatches(FooClass + "void r() { Foo f = \"hello world\"; }",
                         cxxFunctionalCastExpr()));
}

TEST_P(ASTMatchersTest, CXXDynamicCastExpr) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("struct B { virtual ~B() {} }; struct D : B {};"
                      "B b;"
                      "D* p = dynamic_cast<D*>(&b);",
                      cxxDynamicCastExpr()));
}

TEST_P(ASTMatchersTest, CXXStaticCastExpr_MatchesSimpleCase) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("void* p(static_cast<void*>(&p));", cxxStaticCastExpr()));
}

TEST_P(ASTMatchersTest, CXXStaticCastExpr_DoesNotMatchOtherCasts) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("char* p = (char*)(&p);", cxxStaticCastExpr()));
  EXPECT_TRUE(
      notMatches("char q, *p = const_cast<char*>(&q);", cxxStaticCastExpr()));
  EXPECT_TRUE(notMatches("void* p = reinterpret_cast<char*>(&p);",
                         cxxStaticCastExpr()));
  EXPECT_TRUE(notMatches("struct B { virtual ~B() {} }; struct D : B {};"
                         "B b;"
                         "D* p = dynamic_cast<D*>(&b);",
                         cxxStaticCastExpr()));
}

TEST_P(ASTMatchersTest, CStyleCastExpr_MatchesSimpleCase) {
  EXPECT_TRUE(matches("int i = (int) 2.2f;", cStyleCastExpr()));
}

TEST_P(ASTMatchersTest, CStyleCastExpr_DoesNotMatchOtherCasts) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("char* p = static_cast<char*>(0);"
                         "char q, *r = const_cast<char*>(&q);"
                         "void* s = reinterpret_cast<char*>(&s);"
                         "struct B { virtual ~B() {} }; struct D : B {};"
                         "B b;"
                         "D* t = dynamic_cast<D*>(&b);",
                         cStyleCastExpr()));
}

TEST_P(ASTMatchersTest, ImplicitCastExpr_MatchesSimpleCase) {
  // This test creates an implicit const cast.
  EXPECT_TRUE(
      matches("void f() { int x = 0; const int y = x; }",
              traverse(TK_AsIs, varDecl(hasInitializer(implicitCastExpr())))));
  // This test creates an implicit cast from int to char.
  EXPECT_TRUE(
      matches("char c = 0;",
              traverse(TK_AsIs, varDecl(hasInitializer(implicitCastExpr())))));
  // This test creates an implicit array-to-pointer cast.
  EXPECT_TRUE(
      matches("int arr[6]; int *p = arr;",
              traverse(TK_AsIs, varDecl(hasInitializer(implicitCastExpr())))));
}

TEST_P(ASTMatchersTest, ImplicitCastExpr_DoesNotMatchIncorrectly) {
  // This test verifies that implicitCastExpr() matches exactly when implicit
  // casts are present, and that it ignores explicit and paren casts.

  // These two test cases have no casts.
  EXPECT_TRUE(
      notMatches("int x = 0;", varDecl(hasInitializer(implicitCastExpr()))));
  EXPECT_TRUE(
      notMatches("int x = (0);", varDecl(hasInitializer(implicitCastExpr()))));
  EXPECT_TRUE(notMatches("void f() { int x = 0; double d = (double) x; }",
                         varDecl(hasInitializer(implicitCastExpr()))));
}

TEST_P(ASTMatchersTest, ImplicitCastExpr_DoesNotMatchIncorrectly_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches("int x = 0, &y = x;",
                         varDecl(hasInitializer(implicitCastExpr()))));
  EXPECT_TRUE(notMatches("const int *p; int *q = const_cast<int *>(p);",
                         varDecl(hasInitializer(implicitCastExpr()))));
}

TEST_P(ASTMatchersTest, Stmt_DoesNotMatchDeclarations) {
  EXPECT_TRUE(notMatches("struct X {};", stmt()));
}

TEST_P(ASTMatchersTest, Stmt_MatchesCompoundStatments) {
  EXPECT_TRUE(matches("void x() {}", stmt()));
}

TEST_P(ASTMatchersTest, DeclStmt_DoesNotMatchCompoundStatements) {
  EXPECT_TRUE(notMatches("void x() {}", declStmt()));
}

TEST_P(ASTMatchersTest, DeclStmt_MatchesVariableDeclarationStatements) {
  EXPECT_TRUE(matches("void x() { int a; }", declStmt()));
}

TEST_P(ASTMatchersTest, ExprWithCleanups_MatchesExprWithCleanups) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(
      matches("struct Foo { ~Foo(); };"
              "const Foo f = Foo();",
              traverse(TK_AsIs, varDecl(hasInitializer(exprWithCleanups())))));
  EXPECT_FALSE(
      matches("struct Foo { }; Foo a;"
              "const Foo f = a;",
              traverse(TK_AsIs, varDecl(hasInitializer(exprWithCleanups())))));
}

TEST_P(ASTMatchersTest, InitListExpr) {
  EXPECT_TRUE(matches("int a[] = { 1, 2 };",
                      initListExpr(hasType(asString("int[2]")))));
  EXPECT_TRUE(matches("struct B { int x, y; }; struct B b = { 5, 6 };",
                      initListExpr(hasType(recordDecl(hasName("B"))))));
  EXPECT_TRUE(
      matches("int i[1] = {42, [0] = 43};", integerLiteral(equals(42))));
}

TEST_P(ASTMatchersTest, InitListExpr_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("struct S { S(void (*a)()); };"
                      "void f();"
                      "S s[1] = { &f };",
                      declRefExpr(to(functionDecl(hasName("f"))))));
}

TEST_P(ASTMatchersTest,
       CXXStdInitializerListExpression_MatchesCXXStdInitializerListExpression) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  StringRef code = "namespace std {"
                   "template <typename E> class initializer_list {"
                   "  public: const E *a, *b;"
                   "};"
                   "}"
                   "struct A {"
                   "  A(std::initializer_list<int>) {}"
                   "};";
  EXPECT_TRUE(matches(
      code + "A a{0};",
      traverse(TK_AsIs, cxxConstructExpr(has(cxxStdInitializerListExpr()),
                                         hasDeclaration(cxxConstructorDecl(
                                             ofClass(hasName("A"))))))));
  EXPECT_TRUE(matches(
      code + "A a = {0};",
      traverse(TK_AsIs, cxxConstructExpr(has(cxxStdInitializerListExpr()),
                                         hasDeclaration(cxxConstructorDecl(
                                             ofClass(hasName("A"))))))));

  EXPECT_TRUE(notMatches("int a[] = { 1, 2 };", cxxStdInitializerListExpr()));
  EXPECT_TRUE(notMatches("struct B { int x, y; }; B b = { 5, 6 };",
                         cxxStdInitializerListExpr()));
}

TEST_P(ASTMatchersTest, UsingDecl_MatchesUsingDeclarations) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("namespace X { int x; } using X::x;", usingDecl()));
}

TEST_P(ASTMatchersTest, UsingDecl_MatchesShadowUsingDelcarations) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("namespace f { int a; } using f::a;",
                      usingDecl(hasAnyUsingShadowDecl(hasName("a")))));
}

TEST_P(ASTMatchersTest, UsingEnumDecl_MatchesUsingEnumDeclarations) {
  if (!GetParam().isCXX20OrLater()) {
    return;
  }
  EXPECT_TRUE(
      matches("namespace X { enum x {}; } using enum X::x;", usingEnumDecl()));
}

TEST_P(ASTMatchersTest, UsingEnumDecl_MatchesShadowUsingDeclarations) {
  if (!GetParam().isCXX20OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("namespace f { enum a {b}; } using enum f::a;",
                      usingEnumDecl(hasAnyUsingShadowDecl(hasName("b")))));
}

TEST_P(ASTMatchersTest, UsingDirectiveDecl_MatchesUsingNamespace) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("namespace X { int x; } using namespace X;",
                      usingDirectiveDecl()));
  EXPECT_FALSE(
      matches("namespace X { int x; } using X::x;", usingDirectiveDecl()));
}

TEST_P(ASTMatchersTest, WhileStmt) {
  EXPECT_TRUE(notMatches("void x() {}", whileStmt()));
  EXPECT_TRUE(matches("void x() { while(1); }", whileStmt()));
  EXPECT_TRUE(notMatches("void x() { do {} while(1); }", whileStmt()));
}

TEST_P(ASTMatchersTest, DoStmt_MatchesDoLoops) {
  EXPECT_TRUE(matches("void x() { do {} while(1); }", doStmt()));
  EXPECT_TRUE(matches("void x() { do ; while(0); }", doStmt()));
}

TEST_P(ASTMatchersTest, DoStmt_DoesNotMatchWhileLoops) {
  EXPECT_TRUE(notMatches("void x() { while(1) {} }", doStmt()));
}

TEST_P(ASTMatchersTest, SwitchCase_MatchesCase) {
  EXPECT_TRUE(matches("void x() { switch(42) { case 42:; } }", switchCase()));
  EXPECT_TRUE(matches("void x() { switch(42) { default:; } }", switchCase()));
  EXPECT_TRUE(matches("void x() { switch(42) default:; }", switchCase()));
  EXPECT_TRUE(notMatches("void x() { switch(42) {} }", switchCase()));
}

TEST_P(ASTMatchersTest, SwitchCase_MatchesSwitch) {
  EXPECT_TRUE(matches("void x() { switch(42) { case 42:; } }", switchStmt()));
  EXPECT_TRUE(matches("void x() { switch(42) { default:; } }", switchStmt()));
  EXPECT_TRUE(matches("void x() { switch(42) default:; }", switchStmt()));
  EXPECT_TRUE(notMatches("void x() {}", switchStmt()));
}

TEST_P(ASTMatchersTest, CxxExceptionHandling_SimpleCases) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("void foo() try { } catch(int X) { }", cxxCatchStmt()));
  EXPECT_TRUE(matches("void foo() try { } catch(int X) { }", cxxTryStmt()));
  EXPECT_TRUE(
      notMatches("void foo() try { } catch(int X) { }", cxxThrowExpr()));
  EXPECT_TRUE(
      matches("void foo() try { throw; } catch(int X) { }", cxxThrowExpr()));
  EXPECT_TRUE(
      matches("void foo() try { throw 5;} catch(int X) { }", cxxThrowExpr()));
  EXPECT_TRUE(matches("void foo() try { throw; } catch(...) { }",
                      cxxCatchStmt(isCatchAll())));
  EXPECT_TRUE(notMatches("void foo() try { throw; } catch(int) { }",
                         cxxCatchStmt(isCatchAll())));
  EXPECT_TRUE(matches("void foo() try {} catch(int X) { }",
                      varDecl(isExceptionVariable())));
  EXPECT_TRUE(notMatches("void foo() try { int X; } catch (...) { }",
                         varDecl(isExceptionVariable())));
}

TEST_P(ASTMatchersTest, ParenExpr_SimpleCases) {
  EXPECT_TRUE(matches("int i = (3);", traverse(TK_AsIs, parenExpr())));
  EXPECT_TRUE(matches("int i = (3 + 7);", traverse(TK_AsIs, parenExpr())));
  EXPECT_TRUE(notMatches("int i = 3;", traverse(TK_AsIs, parenExpr())));
  EXPECT_TRUE(notMatches("int f() { return 1; }; void g() { int a = f(); }",
                         traverse(TK_AsIs, parenExpr())));
}

TEST_P(ASTMatchersTest, IgnoringParens) {
  EXPECT_FALSE(matches("const char* str = (\"my-string\");",
                       traverse(TK_AsIs, implicitCastExpr(hasSourceExpression(
                                             stringLiteral())))));
  EXPECT_TRUE(
      matches("const char* str = (\"my-string\");",
              traverse(TK_AsIs, implicitCastExpr(hasSourceExpression(
                                    ignoringParens(stringLiteral()))))));
}

TEST_P(ASTMatchersTest, QualType) {
  EXPECT_TRUE(matches("struct S {};", qualType().bind("loc")));
}

TEST_P(ASTMatchersTest, ConstantArrayType) {
  EXPECT_TRUE(matches("int a[2];", constantArrayType()));
  EXPECT_TRUE(notMatches("void f() { int a[] = { 2, 3 }; int b[a[0]]; }",
                         constantArrayType(hasElementType(builtinType()))));

  EXPECT_TRUE(matches("int a[42];", constantArrayType(hasSize(42))));
  EXPECT_TRUE(matches("int b[2*21];", constantArrayType(hasSize(42))));
  EXPECT_TRUE(notMatches("int c[41], d[43];", constantArrayType(hasSize(42))));
}

TEST_P(ASTMatchersTest, DependentSizedArrayType) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(
      matches("template <typename T, int Size> class array { T data[Size]; };",
              dependentSizedArrayType()));
  EXPECT_TRUE(
      notMatches("int a[42]; int b[] = { 2, 3 }; void f() { int c[b[0]]; }",
                 dependentSizedArrayType()));
}

TEST_P(ASTMatchersTest, DependentSizedExtVectorType) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("template<typename T, int Size>"
                      "class vector {"
                      "  typedef T __attribute__((ext_vector_type(Size))) type;"
                      "};",
                      dependentSizedExtVectorType()));
  EXPECT_TRUE(
      notMatches("int a[42]; int b[] = { 2, 3 }; void f() { int c[b[0]]; }",
                 dependentSizedExtVectorType()));
}

TEST_P(ASTMatchersTest, IncompleteArrayType) {
  EXPECT_TRUE(matches("int a[] = { 2, 3 };", incompleteArrayType()));
  EXPECT_TRUE(matches("void f(int a[]) {}", incompleteArrayType()));

  EXPECT_TRUE(notMatches("int a[42]; void f() { int b[a[0]]; }",
                         incompleteArrayType()));
}

TEST_P(ASTMatchersTest, VariableArrayType) {
  EXPECT_TRUE(matches("void f(int b) { int a[b]; }", variableArrayType()));
  EXPECT_TRUE(notMatches("int a[] = {2, 3}; int b[42];", variableArrayType()));

  EXPECT_TRUE(matches("void f(int b) { int a[b]; }",
                      variableArrayType(hasSizeExpr(ignoringImpCasts(
                          declRefExpr(to(varDecl(hasName("b")))))))));
}

TEST_P(ASTMatchersTest, AtomicType) {
  if (llvm::Triple(llvm::sys::getDefaultTargetTriple()).getOS() !=
      llvm::Triple::Win32) {
    // FIXME: Make this work for MSVC.
    EXPECT_TRUE(matches("_Atomic(int) i;", atomicType()));

    EXPECT_TRUE(
        matches("_Atomic(int) i;", atomicType(hasValueType(isInteger()))));
    EXPECT_TRUE(
        notMatches("_Atomic(float) f;", atomicType(hasValueType(isInteger()))));
  }
}

TEST_P(ASTMatchersTest, AutoType) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("auto i = 2;", autoType()));
  EXPECT_TRUE(matches("int v[] = { 2, 3 }; void f() { for (int i : v) {} }",
                      autoType()));

  EXPECT_TRUE(matches("auto i = 2;", varDecl(hasType(isInteger()))));
  EXPECT_TRUE(matches("struct X{}; auto x = X{};",
                      varDecl(hasType(recordDecl(hasName("X"))))));

  // FIXME: Matching against the type-as-written can't work here, because the
  //        type as written was not deduced.
  // EXPECT_TRUE(matches("auto a = 1;",
  //                    autoType(hasDeducedType(isInteger()))));
  // EXPECT_TRUE(notMatches("auto b = 2.0;",
  //                       autoType(hasDeducedType(isInteger()))));
}

TEST_P(ASTMatchersTest, DecltypeType) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("decltype(1 + 1) sum = 1 + 1;", decltypeType()));
  EXPECT_TRUE(matches("decltype(1 + 1) sum = 1 + 1;",
                      decltypeType(hasUnderlyingType(isInteger()))));
}

TEST_P(ASTMatchersTest, FunctionType) {
  EXPECT_TRUE(matches("int (*f)(int);", functionType()));
  EXPECT_TRUE(matches("void f(int i) {}", functionType()));
}

TEST_P(ASTMatchersTest, IgnoringParens_Type) {
  EXPECT_TRUE(
      notMatches("void (*fp)(void);", pointerType(pointee(functionType()))));
  EXPECT_TRUE(matches("void (*fp)(void);",
                      pointerType(pointee(ignoringParens(functionType())))));
}

TEST_P(ASTMatchersTest, FunctionProtoType) {
  EXPECT_TRUE(matches("int (*f)(int);", functionProtoType()));
  EXPECT_TRUE(matches("void f(int i);", functionProtoType()));
  EXPECT_TRUE(matches("void f(void);", functionProtoType(parameterCountIs(0))));
}

TEST_P(ASTMatchersTest, FunctionProtoType_C) {
  if (!GetParam().isCOrEarlier(17)) {
    return;
  }
  EXPECT_TRUE(notMatches("void f();", functionProtoType()));
}

TEST_P(ASTMatchersTest, FunctionProtoType_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("void f();", functionProtoType(parameterCountIs(0))));
}

TEST_P(ASTMatchersTest, ParenType) {
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

TEST_P(ASTMatchersTest, PointerType) {
  // FIXME: Reactive when these tests can be more specific (not matching
  // implicit code on certain platforms), likely when we have hasDescendant for
  // Types/TypeLocs.
  // EXPECT_TRUE(matchAndVerifyResultTrue(
  //    "int* a;",
  //    pointerTypeLoc(pointeeLoc(typeLoc().bind("loc"))),
  //    std::make_unique<VerifyIdIsBoundTo<TypeLoc>>("loc", 1)));
  // EXPECT_TRUE(matchAndVerifyResultTrue(
  //    "int* a;",
  //    pointerTypeLoc().bind("loc"),
  //    std::make_unique<VerifyIdIsBoundTo<TypeLoc>>("loc", 1)));
  EXPECT_TRUE(matches("int** a;", loc(pointerType(pointee(qualType())))));
  EXPECT_TRUE(matches("int** a;", loc(pointerType(pointee(pointerType())))));
  EXPECT_TRUE(matches("int* b; int* * const a = &b;",
                      loc(qualType(isConstQualified(), pointerType()))));

  StringRef Fragment = "int *ptr;";
  EXPECT_TRUE(notMatches(Fragment,
                         varDecl(hasName("ptr"), hasType(blockPointerType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("ptr"), hasType(memberPointerType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("ptr"), hasType(pointerType()))));
  EXPECT_TRUE(
      notMatches(Fragment, varDecl(hasName("ptr"), hasType(referenceType()))));
}

TEST_P(ASTMatchersTest, PointerType_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  StringRef Fragment = "struct A { int i; }; int A::* ptr = &A::i;";
  EXPECT_TRUE(notMatches(Fragment,
                         varDecl(hasName("ptr"), hasType(blockPointerType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("ptr"), hasType(memberPointerType()))));
  EXPECT_TRUE(
      notMatches(Fragment, varDecl(hasName("ptr"), hasType(pointerType()))));
  EXPECT_TRUE(
      notMatches(Fragment, varDecl(hasName("ptr"), hasType(referenceType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("ptr"), hasType(lValueReferenceType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("ptr"), hasType(rValueReferenceType()))));

  Fragment = "int a; int &ref = a;";
  EXPECT_TRUE(notMatches(Fragment,
                         varDecl(hasName("ref"), hasType(blockPointerType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("ref"), hasType(memberPointerType()))));
  EXPECT_TRUE(
      notMatches(Fragment, varDecl(hasName("ref"), hasType(pointerType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("ref"), hasType(referenceType()))));
  EXPECT_TRUE(matches(Fragment,
                      varDecl(hasName("ref"), hasType(lValueReferenceType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("ref"), hasType(rValueReferenceType()))));
}

TEST_P(ASTMatchersTest, PointerType_CXX11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  StringRef Fragment = "int &&ref = 2;";
  EXPECT_TRUE(notMatches(Fragment,
                         varDecl(hasName("ref"), hasType(blockPointerType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("ref"), hasType(memberPointerType()))));
  EXPECT_TRUE(
      notMatches(Fragment, varDecl(hasName("ref"), hasType(pointerType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("ref"), hasType(referenceType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("ref"), hasType(lValueReferenceType()))));
  EXPECT_TRUE(matches(Fragment,
                      varDecl(hasName("ref"), hasType(rValueReferenceType()))));
}

TEST_P(ASTMatchersTest, AutoRefTypes) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  StringRef Fragment = "auto a = 1;"
                       "auto b = a;"
                       "auto &c = a;"
                       "auto &&d = c;"
                       "auto &&e = 2;";
  EXPECT_TRUE(
      notMatches(Fragment, varDecl(hasName("a"), hasType(referenceType()))));
  EXPECT_TRUE(
      notMatches(Fragment, varDecl(hasName("b"), hasType(referenceType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("c"), hasType(referenceType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("c"), hasType(lValueReferenceType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("c"), hasType(rValueReferenceType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("d"), hasType(referenceType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("d"), hasType(lValueReferenceType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("d"), hasType(rValueReferenceType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("e"), hasType(referenceType()))));
  EXPECT_TRUE(notMatches(
      Fragment, varDecl(hasName("e"), hasType(lValueReferenceType()))));
  EXPECT_TRUE(
      matches(Fragment, varDecl(hasName("e"), hasType(rValueReferenceType()))));
}

TEST_P(ASTMatchersTest, EnumType) {
  EXPECT_TRUE(
      matches("enum Color { Green }; enum Color color;", loc(enumType())));
}

TEST_P(ASTMatchersTest, EnumType_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("enum Color { Green }; Color color;", loc(enumType())));
}

TEST_P(ASTMatchersTest, EnumType_CXX11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(
      matches("enum class Color { Green }; Color color;", loc(enumType())));
}

TEST_P(ASTMatchersTest, PointerType_MatchesPointersToConstTypes) {
  EXPECT_TRUE(matches("int b; int * const a = &b;", loc(pointerType())));
  EXPECT_TRUE(matches("int b; int * const a = &b;", loc(pointerType())));
  EXPECT_TRUE(matches("int b; const int * a = &b;",
                      loc(pointerType(pointee(builtinType())))));
  EXPECT_TRUE(matches("int b; const int * a = &b;",
                      pointerType(pointee(builtinType()))));
}

TEST_P(ASTMatchersTest, TypedefType) {
  EXPECT_TRUE(matches("typedef int X; X a;",
                      varDecl(hasName("a"), hasType(typedefType()))));
}

TEST_P(ASTMatchersTest, MacroQualifiedType) {
  EXPECT_TRUE(matches(
      R"(
        #define CDECL __attribute__((cdecl))
        typedef void (CDECL *X)();
      )",
      typedefDecl(hasType(pointerType(pointee(macroQualifiedType()))))));
  EXPECT_TRUE(notMatches(
      R"(
        typedef void (__attribute__((cdecl)) *Y)();
      )",
      typedefDecl(hasType(pointerType(pointee(macroQualifiedType()))))));
}

TEST_P(ASTMatchersTest, TemplateSpecializationType) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("template <typename T> class A{}; A<int> a;",
                      templateSpecializationType()));
}

TEST_P(ASTMatchersTest, DeducedTemplateSpecializationType) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }
  EXPECT_TRUE(
      matches("template <typename T> class A{ public: A(T) {} }; A a(1);",
              deducedTemplateSpecializationType()));
}

TEST_P(ASTMatchersTest, DependentNameType) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches(
      R"(
        template <typename T> struct declToImport {
          typedef typename T::type dependent_name;
        };
      )",
      dependentNameType()));
}

TEST_P(ASTMatchersTest, DependentTemplateSpecializationType) {
  if (!GetParam().isCXX()) {
    return;
  }

  EXPECT_TRUE(matches(
      R"(
        template<typename T> struct A;
        template<typename T> struct declToImport {
          typename A<T>::template B<T> a;
        };
      )",
      dependentTemplateSpecializationType()));
}

TEST_P(ASTMatchersTest, RecordType) {
  EXPECT_TRUE(matches("struct S {}; struct S s;",
                      recordType(hasDeclaration(recordDecl(hasName("S"))))));
  EXPECT_TRUE(notMatches("int i;",
                         recordType(hasDeclaration(recordDecl(hasName("S"))))));
}

TEST_P(ASTMatchersTest, RecordType_CXX) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("class C {}; C c;", recordType()));
  EXPECT_TRUE(matches("struct S {}; S s;",
                      recordType(hasDeclaration(recordDecl(hasName("S"))))));
}

TEST_P(ASTMatchersTest, SubstTemplateTypeParmType) {
  if (!GetParam().isCXX()) {
    return;
  }
  StringRef code = "template <typename T>"
                   "int F() {"
                   "  return 1 + T();"
                   "}"
                   "int i = F<int>();";
  EXPECT_FALSE(matches(code, binaryOperator(hasLHS(
                                 expr(hasType(substTemplateTypeParmType()))))));
  EXPECT_TRUE(matches(code, binaryOperator(hasRHS(
                                expr(hasType(substTemplateTypeParmType()))))));
}

TEST_P(ASTMatchersTest, NestedNameSpecifier) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(
      matches("namespace ns { struct A {}; } ns::A a;", nestedNameSpecifier()));
  EXPECT_TRUE(matches("template <typename T> class A { typename T::B b; };",
                      nestedNameSpecifier()));
  EXPECT_TRUE(
      matches("struct A { void f(); }; void A::f() {}", nestedNameSpecifier()));
  EXPECT_TRUE(matches("namespace a { namespace b {} } namespace ab = a::b;",
                      nestedNameSpecifier()));

  EXPECT_TRUE(matches("struct A { static void f() {} }; void g() { A::f(); }",
                      nestedNameSpecifier()));
  EXPECT_TRUE(
      notMatches("struct A { static void f() {} }; void g(A* a) { a->f(); }",
                 nestedNameSpecifier()));
}

TEST_P(ASTMatchersTest, Attr) {
  // Windows adds some implicit attributes.
  bool AutomaticAttributes = StringRef(GetParam().Target).contains("win32");
  if (GetParam().isCXX11OrLater()) {
    EXPECT_TRUE(matches("struct [[clang::warn_unused_result]] F{};", attr()));

    // Unknown attributes are not parsed into an AST node.
    if (!AutomaticAttributes) {
      EXPECT_TRUE(notMatches("int x [[unknownattr]];", attr()));
    }
  }
  if (GetParam().isCXX17OrLater()) {
    EXPECT_TRUE(matches("struct [[nodiscard]] F{};", attr()));
  }
  EXPECT_TRUE(matches("int x(int * __attribute__((nonnull)) );", attr()));
  if (!AutomaticAttributes) {
    EXPECT_TRUE(notMatches("struct F{}; int x(int *);", attr()));
    // Some known attributes are not parsed into an AST node.
    EXPECT_TRUE(notMatches("typedef int x __attribute__((ext_vector_type(1)));",
                           attr()));
  }
}

TEST_P(ASTMatchersTest, NullStmt) {
  EXPECT_TRUE(matches("void f() {int i;;}", nullStmt()));
  EXPECT_TRUE(notMatches("void f() {int i;}", nullStmt()));
}

TEST_P(ASTMatchersTest, NamespaceAliasDecl) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches("namespace test {} namespace alias = ::test;",
                      namespaceAliasDecl(hasName("alias"))));
}

TEST_P(ASTMatchersTest, NestedNameSpecifier_MatchesTypes) {
  if (!GetParam().isCXX()) {
    return;
  }
  NestedNameSpecifierMatcher Matcher = nestedNameSpecifier(
      specifiesType(hasDeclaration(recordDecl(hasName("A")))));
  EXPECT_TRUE(matches("struct A { struct B {}; }; A::B b;", Matcher));
  EXPECT_TRUE(
      matches("struct A { struct B { struct C {}; }; }; A::B::C c;", Matcher));
  EXPECT_TRUE(notMatches("namespace A { struct B {}; } A::B b;", Matcher));
}

TEST_P(ASTMatchersTest, NestedNameSpecifier_MatchesNamespaceDecls) {
  if (!GetParam().isCXX()) {
    return;
  }
  NestedNameSpecifierMatcher Matcher =
      nestedNameSpecifier(specifiesNamespace(hasName("ns")));
  EXPECT_TRUE(matches("namespace ns { struct A {}; } ns::A a;", Matcher));
  EXPECT_TRUE(notMatches("namespace xx { struct A {}; } xx::A a;", Matcher));
  EXPECT_TRUE(notMatches("struct ns { struct A {}; }; ns::A a;", Matcher));
}

TEST_P(ASTMatchersTest,
       NestedNameSpecifier_MatchesNestedNameSpecifierPrefixes) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(
      matches("struct A { struct B { struct C {}; }; }; A::B::C c;",
              nestedNameSpecifier(hasPrefix(specifiesType(asString("A"))))));
  EXPECT_TRUE(matches("struct A { struct B { struct C {}; }; }; A::B::C c;",
                      nestedNameSpecifierLoc(hasPrefix(
                          specifiesTypeLoc(loc(qualType(asString("A"))))))));
  EXPECT_TRUE(matches(
      "namespace N { struct A { struct B { struct C {}; }; }; } N::A::B::C c;",
      nestedNameSpecifierLoc(
          hasPrefix(specifiesTypeLoc(loc(qualType(asString("N::A"))))))));
}

template <typename T>
class VerifyAncestorHasChildIsEqual : public BoundNodesCallback {
public:
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

TEST_P(ASTMatchersTest, IsEqualTo_MatchesNodesByIdentity) {
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "void f() { if (1) if(1) {} }", ifStmt().bind(""),
      std::make_unique<VerifyAncestorHasChildIsEqual<IfStmt>>()));
}

TEST_P(ASTMatchersTest, IsEqualTo_MatchesNodesByIdentity_Cxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { class Y {}; };", recordDecl(hasName("::X::Y")).bind(""),
      std::make_unique<VerifyAncestorHasChildIsEqual<CXXRecordDecl>>()));
  EXPECT_TRUE(matchAndVerifyResultTrue(
      "class X { class Y {} y; };",
      fieldDecl(hasName("y"), hasType(type().bind(""))).bind("decl"),
      std::make_unique<VerifyAncestorHasChildIsEqual<Type>>()));
}

TEST_P(ASTMatchersTest, TypedefDecl) {
  EXPECT_TRUE(matches("typedef int typedefDeclTest;",
                      typedefDecl(hasName("typedefDeclTest"))));
}

TEST_P(ASTMatchersTest, TypedefDecl_Cxx) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(notMatches("using typedefDeclTest = int;",
                         typedefDecl(hasName("typedefDeclTest"))));
}

TEST_P(ASTMatchersTest, TypeAliasDecl) {
  EXPECT_TRUE(notMatches("typedef int typeAliasTest;",
                         typeAliasDecl(hasName("typeAliasTest"))));
}

TEST_P(ASTMatchersTest, TypeAliasDecl_CXX) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("using typeAliasTest = int;",
                      typeAliasDecl(hasName("typeAliasTest"))));
}

TEST_P(ASTMatchersTest, TypedefNameDecl) {
  EXPECT_TRUE(matches("typedef int typedefNameDeclTest1;",
                      typedefNameDecl(hasName("typedefNameDeclTest1"))));
}

TEST_P(ASTMatchersTest, TypedefNameDecl_CXX) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("using typedefNameDeclTest = int;",
                      typedefNameDecl(hasName("typedefNameDeclTest"))));
}

TEST_P(ASTMatchersTest, TypeAliasTemplateDecl) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  StringRef Code = R"(
    template <typename T>
    class X { T t; };

    template <typename T>
    using typeAliasTemplateDecl = X<T>;

    using typeAliasDecl = X<int>;
  )";
  EXPECT_TRUE(
      matches(Code, typeAliasTemplateDecl(hasName("typeAliasTemplateDecl"))));
  EXPECT_TRUE(
      notMatches(Code, typeAliasTemplateDecl(hasName("typeAliasDecl"))));
}

TEST_P(ASTMatchersTest, QualifiedTypeLocTest_BindsToConstIntVarDecl) {
  EXPECT_TRUE(matches("const int x = 0;",
                      qualifiedTypeLoc(loc(asString("const int")))));
}

TEST_P(ASTMatchersTest, QualifiedTypeLocTest_BindsToConstIntFunctionDecl) {
  EXPECT_TRUE(matches("const int f() { return 5; }",
                      qualifiedTypeLoc(loc(asString("const int")))));
}

TEST_P(ASTMatchersTest, QualifiedTypeLocTest_DoesNotBindToUnqualifiedVarDecl) {
  EXPECT_TRUE(notMatches("int x = 0;", qualifiedTypeLoc(loc(asString("int")))));
}

TEST_P(ASTMatchersTest, QualifiedTypeLocTest_IntDoesNotBindToConstIntDecl) {
  EXPECT_TRUE(
      notMatches("const int x = 0;", qualifiedTypeLoc(loc(asString("int")))));
}

TEST_P(ASTMatchersTest, QualifiedTypeLocTest_IntDoesNotBindToConstFloatDecl) {
  EXPECT_TRUE(
      notMatches("const float x = 0;", qualifiedTypeLoc(loc(asString("int")))));
}

TEST_P(ASTMatchersTest, PointerTypeLocTest_BindsToAnyPointerTypeLoc) {
  auto matcher = varDecl(hasName("x"), hasTypeLoc(pointerTypeLoc()));
  EXPECT_TRUE(matches("int* x;", matcher));
  EXPECT_TRUE(matches("float* x;", matcher));
  EXPECT_TRUE(matches("char* x;", matcher));
  EXPECT_TRUE(matches("void* x;", matcher));
}

TEST_P(ASTMatchersTest, PointerTypeLocTest_DoesNotBindToNonPointerTypeLoc) {
  auto matcher = varDecl(hasName("x"), hasTypeLoc(pointerTypeLoc()));
  EXPECT_TRUE(notMatches("int x;", matcher));
  EXPECT_TRUE(notMatches("float x;", matcher));
  EXPECT_TRUE(notMatches("char x;", matcher));
}

TEST_P(ASTMatchersTest, ReferenceTypeLocTest_BindsToAnyReferenceTypeLoc) {
  if (!GetParam().isCXX()) {
    return;
  }
  auto matcher = varDecl(hasName("r"), hasTypeLoc(referenceTypeLoc()));
  EXPECT_TRUE(matches("int rr = 3; int& r = rr;", matcher));
  EXPECT_TRUE(matches("int rr = 3; auto& r = rr;", matcher));
  EXPECT_TRUE(matches("int rr = 3; const int& r = rr;", matcher));
  EXPECT_TRUE(matches("float rr = 3.0; float& r = rr;", matcher));
  EXPECT_TRUE(matches("char rr = 'a'; char& r = rr;", matcher));
}

TEST_P(ASTMatchersTest, ReferenceTypeLocTest_DoesNotBindToNonReferenceTypeLoc) {
  auto matcher = varDecl(hasName("r"), hasTypeLoc(referenceTypeLoc()));
  EXPECT_TRUE(notMatches("int r;", matcher));
  EXPECT_TRUE(notMatches("int r = 3;", matcher));
  EXPECT_TRUE(notMatches("const int r = 3;", matcher));
  EXPECT_TRUE(notMatches("int* r;", matcher));
  EXPECT_TRUE(notMatches("float r;", matcher));
  EXPECT_TRUE(notMatches("char r;", matcher));
}

TEST_P(ASTMatchersTest, ReferenceTypeLocTest_BindsToAnyRvalueReferenceTypeLoc) {
  if (!GetParam().isCXX()) {
    return;
  }
  auto matcher = varDecl(hasName("r"), hasTypeLoc(referenceTypeLoc()));
  EXPECT_TRUE(matches("int&& r = 3;", matcher));
  EXPECT_TRUE(matches("auto&& r = 3;", matcher));
  EXPECT_TRUE(matches("float&& r = 3.0;", matcher));
}

TEST_P(ASTMatchersTest,
       TemplateSpecializationTypeLocTest_BindsToVarDeclTemplateSpecialization) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(matches(
      "template <typename T> class C {}; C<char> var;",
      varDecl(hasName("var"), hasTypeLoc(templateSpecializationTypeLoc()))));
}

TEST_P(
    ASTMatchersTest,
    TemplateSpecializationTypeLocTest_DoesNotBindToNonTemplateSpecialization) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(notMatches(
      "class C {}; C var;",
      varDecl(hasName("var"), hasTypeLoc(templateSpecializationTypeLoc()))));
}

TEST_P(ASTMatchersTest, LambdaCaptureTest) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(matches("int main() { int cc; auto f = [cc](){ return cc; }; }",
                      lambdaExpr(hasAnyCapture(lambdaCapture()))));
}

TEST_P(ASTMatchersTest, LambdaCaptureTest_BindsToCaptureOfVarDecl) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  auto matcher = lambdaExpr(
      hasAnyCapture(lambdaCapture(capturesVar(varDecl(hasName("cc"))))));
  EXPECT_TRUE(matches("int main() { int cc; auto f = [cc](){ return cc; }; }",
                      matcher));
  EXPECT_TRUE(matches("int main() { int cc; auto f = [&cc](){ return cc; }; }",
                      matcher));
  EXPECT_TRUE(
      matches("int main() { int cc; auto f = [=](){ return cc; }; }", matcher));
  EXPECT_TRUE(
      matches("int main() { int cc; auto f = [&](){ return cc; }; }", matcher));
  EXPECT_TRUE(matches(
      "void f(int a) { int cc[a]; auto f = [&](){ return cc;}; }", matcher));
}

TEST_P(ASTMatchersTest, LambdaCaptureTest_BindsToCaptureWithInitializer) {
  if (!GetParam().isCXX14OrLater()) {
    return;
  }
  auto matcher = lambdaExpr(hasAnyCapture(lambdaCapture(capturesVar(
      varDecl(hasName("cc"), hasInitializer(integerLiteral(equals(1))))))));
  EXPECT_TRUE(
      matches("int main() { auto lambda = [cc = 1] {return cc;}; }", matcher));
  EXPECT_TRUE(
      matches("int main() { int cc = 2; auto lambda = [cc = 1] {return cc;}; }",
              matcher));
}

TEST_P(ASTMatchersTest, LambdaCaptureTest_DoesNotBindToCaptureOfVarDecl) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  auto matcher = lambdaExpr(
      hasAnyCapture(lambdaCapture(capturesVar(varDecl(hasName("cc"))))));
  EXPECT_FALSE(matches("int main() { auto f = [](){ return 5; }; }", matcher));
  EXPECT_FALSE(matches("int main() { int xx; auto f = [xx](){ return xx; }; }",
                       matcher));
}

TEST_P(ASTMatchersTest,
       LambdaCaptureTest_DoesNotBindToCaptureWithInitializerAndDifferentName) {
  if (!GetParam().isCXX14OrLater()) {
    return;
  }
  EXPECT_FALSE(matches(
      "int main() { auto lambda = [xx = 1] {return xx;}; }",
      lambdaExpr(hasAnyCapture(lambdaCapture(capturesVar(varDecl(
          hasName("cc"), hasInitializer(integerLiteral(equals(1))))))))));
}

TEST_P(ASTMatchersTest, LambdaCaptureTest_BindsToCaptureOfReferenceType) {
  if (!GetParam().isCXX20OrLater()) {
    return;
  }
  auto matcher = lambdaExpr(hasAnyCapture(
      lambdaCapture(capturesVar(varDecl(hasType(referenceType()))))));
  EXPECT_TRUE(matches("template <class ...T> void f(T &...args) {"
                      "  [&...args = args] () mutable {"
                      "  }();"
                      "}"
                      "int main() {"
                      "  int a;"
                      "  f(a);"
                      "}", matcher));
  EXPECT_FALSE(matches("template <class ...T> void f(T &...args) {"
                       "  [...args = args] () mutable {"
                       "  }();"
                       "}"
                       "int main() {"
                       "  int a;"
                       "  f(a);"
                       "}", matcher));
}

TEST_P(ASTMatchersTest, IsDerivedFromRecursion) {
  if (!GetParam().isCXX11OrLater())
    return;

  // Check we don't crash on cycles in the traversal and inheritance hierarchy.
  // Clang will normally enforce there are no cycles, but matchers opted to
  // traverse primary template for dependent specializations, spuriously
  // creating the cycles.
  DeclarationMatcher matcher = cxxRecordDecl(isDerivedFrom("X"));
  EXPECT_TRUE(notMatches(R"cpp(
      template <typename T1, typename T2>
      struct M;

      template <typename T1>
      struct M<T1, void> {};

      template <typename T1, typename T2>
      struct L : M<T1, T2> {};

      template <typename T1, typename T2>
      struct M : L<M<T1, T2>, M<T1, T2>> {};
  )cpp",
                         matcher));

  // Check the running time is not exponential. The number of subojects to
  // traverse grows as fibonacci numbers even though the number of bases to
  // traverse is quadratic.
  // The test will hang if implementation of matchers traverses all subojects.
  EXPECT_TRUE(notMatches(R"cpp(
    template <class T> struct A0 {};
    template <class T> struct A1 : A0<T> {};
    template <class T> struct A2 : A1<T>, A0<T> {};
    template <class T> struct A3 : A2<T>, A1<T> {};
    template <class T> struct A4 : A3<T>, A2<T> {};
    template <class T> struct A5 : A4<T>, A3<T> {};
    template <class T> struct A6 : A5<T>, A4<T> {};
    template <class T> struct A7 : A6<T>, A5<T> {};
    template <class T> struct A8 : A7<T>, A6<T> {};
    template <class T> struct A9 : A8<T>, A7<T> {};
    template <class T> struct A10 : A9<T>, A8<T> {};
    template <class T> struct A11 : A10<T>, A9<T> {};
    template <class T> struct A12 : A11<T>, A10<T> {};
    template <class T> struct A13 : A12<T>, A11<T> {};
    template <class T> struct A14 : A13<T>, A12<T> {};
    template <class T> struct A15 : A14<T>, A13<T> {};
    template <class T> struct A16 : A15<T>, A14<T> {};
    template <class T> struct A17 : A16<T>, A15<T> {};
    template <class T> struct A18 : A17<T>, A16<T> {};
    template <class T> struct A19 : A18<T>, A17<T> {};
    template <class T> struct A20 : A19<T>, A18<T> {};
    template <class T> struct A21 : A20<T>, A19<T> {};
    template <class T> struct A22 : A21<T>, A20<T> {};
    template <class T> struct A23 : A22<T>, A21<T> {};
    template <class T> struct A24 : A23<T>, A22<T> {};
    template <class T> struct A25 : A24<T>, A23<T> {};
    template <class T> struct A26 : A25<T>, A24<T> {};
    template <class T> struct A27 : A26<T>, A25<T> {};
    template <class T> struct A28 : A27<T>, A26<T> {};
    template <class T> struct A29 : A28<T>, A27<T> {};
    template <class T> struct A30 : A29<T>, A28<T> {};
    template <class T> struct A31 : A30<T>, A29<T> {};
    template <class T> struct A32 : A31<T>, A30<T> {};
    template <class T> struct A33 : A32<T>, A31<T> {};
    template <class T> struct A34 : A33<T>, A32<T> {};
    template <class T> struct A35 : A34<T>, A33<T> {};
    template <class T> struct A36 : A35<T>, A34<T> {};
    template <class T> struct A37 : A36<T>, A35<T> {};
    template <class T> struct A38 : A37<T>, A36<T> {};
    template <class T> struct A39 : A38<T>, A37<T> {};
    template <class T> struct A40 : A39<T>, A38<T> {};
)cpp",
                         matcher));
}

TEST(ASTMatchersTestObjC, ObjCMessageCalees) {
  StatementMatcher MessagingFoo =
      objcMessageExpr(callee(objcMethodDecl(hasName("foo"))));

  EXPECT_TRUE(matchesObjC("@interface I"
                          "+ (void)foo;"
                          "@end\n"
                          "int main() {"
                          "  [I foo];"
                          "}",
                          MessagingFoo));
  EXPECT_TRUE(notMatchesObjC("@interface I"
                             "+ (void)foo;"
                             "+ (void)bar;"
                             "@end\n"
                             "int main() {"
                             "  [I bar];"
                             "}",
                             MessagingFoo));
  EXPECT_TRUE(matchesObjC("@interface I"
                          "- (void)foo;"
                          "- (void)bar;"
                          "@end\n"
                          "int main() {"
                          "  I *i;"
                          "  [i foo];"
                          "}",
                          MessagingFoo));
  EXPECT_TRUE(notMatchesObjC("@interface I"
                             "- (void)foo;"
                             "- (void)bar;"
                             "@end\n"
                             "int main() {"
                             "  I *i;"
                             "  [i bar];"
                             "}",
                             MessagingFoo));
}

TEST(ASTMatchersTestObjC, ObjCMessageExpr) {
  // Don't find ObjCMessageExpr where none are present.
  EXPECT_TRUE(notMatchesObjC("", objcMessageExpr(anything())));

  StringRef Objc1String = "@interface Str "
                          " - (Str *)uppercaseString;"
                          "@end "
                          "@interface foo "
                          "- (void)contents;"
                          "- (void)meth:(Str *)text;"
                          "@end "
                          " "
                          "@implementation foo "
                          "- (void) meth:(Str *)text { "
                          "  [self contents];"
                          "  Str *up = [text uppercaseString];"
                          "} "
                          "@end ";
  EXPECT_TRUE(matchesObjC(Objc1String, objcMessageExpr(anything())));
  EXPECT_TRUE(matchesObjC(Objc1String,
                          objcMessageExpr(hasAnySelector({"contents", "meth:"}))

                              ));
  EXPECT_TRUE(
      matchesObjC(Objc1String, objcMessageExpr(hasSelector("contents"))));
  EXPECT_TRUE(matchesObjC(
      Objc1String, objcMessageExpr(hasAnySelector("contents", "contentsA"))));
  EXPECT_FALSE(matchesObjC(
      Objc1String, objcMessageExpr(hasAnySelector("contentsB", "contentsC"))));
  EXPECT_TRUE(
      matchesObjC(Objc1String, objcMessageExpr(matchesSelector("cont*"))));
  EXPECT_FALSE(
      matchesObjC(Objc1String, objcMessageExpr(matchesSelector("?cont*"))));
  EXPECT_TRUE(
      notMatchesObjC(Objc1String, objcMessageExpr(hasSelector("contents"),
                                                  hasNullSelector())));
  EXPECT_TRUE(matchesObjC(Objc1String, objcMessageExpr(hasSelector("contents"),
                                                       hasUnarySelector())));
  EXPECT_TRUE(matchesObjC(Objc1String, objcMessageExpr(hasSelector("contents"),
                                                       numSelectorArgs(0))));
  EXPECT_TRUE(
      matchesObjC(Objc1String, objcMessageExpr(matchesSelector("uppercase*"),
                                               argumentCountIs(0))));
}

TEST(ASTMatchersTestObjC, ObjCStringLiteral) {

  StringRef Objc1String = "@interface NSObject "
                          "@end "
                          "@interface NSString "
                          "@end "
                          "@interface Test : NSObject "
                          "+ (void)someFunction:(NSString *)Desc; "
                          "@end "
                          "@implementation Test "
                          "+ (void)someFunction:(NSString *)Desc { "
                          "    return; "
                          "} "
                          "- (void) foo { "
                          "    [Test someFunction:@\"Ola!\"]; "
                          "}\n"
                          "@end ";
    EXPECT_TRUE(matchesObjC(Objc1String, objcStringLiteral()));
}

TEST(ASTMatchersTestObjC, ObjCDecls) {
  StringRef ObjCString = "@protocol Proto "
                         "- (void)protoDidThing; "
                         "@end "
                         "@interface Thing "
                         "@property int enabled; "
                         "@end "
                         "@interface Thing (ABC) "
                         "- (void)abc_doThing; "
                         "@end "
                         "@implementation Thing "
                         "{ id _ivar; } "
                         "- (void)anything {} "
                         "@end "
                         "@implementation Thing (ABC) "
                         "- (void)abc_doThing {} "
                         "@end ";

  EXPECT_TRUE(matchesObjC(ObjCString, objcProtocolDecl(hasName("Proto"))));
  EXPECT_TRUE(
      matchesObjC(ObjCString, objcImplementationDecl(hasName("Thing"))));
  EXPECT_TRUE(matchesObjC(ObjCString, objcCategoryDecl(hasName("ABC"))));
  EXPECT_TRUE(matchesObjC(ObjCString, objcCategoryImplDecl(hasName("ABC"))));
  EXPECT_TRUE(
      matchesObjC(ObjCString, objcMethodDecl(hasName("protoDidThing"))));
  EXPECT_TRUE(matchesObjC(ObjCString, objcMethodDecl(hasName("abc_doThing"))));
  EXPECT_TRUE(matchesObjC(ObjCString, objcMethodDecl(hasName("anything"))));
  EXPECT_TRUE(matchesObjC(ObjCString, objcIvarDecl(hasName("_ivar"))));
  EXPECT_TRUE(matchesObjC(ObjCString, objcPropertyDecl(hasName("enabled"))));
}

TEST(ASTMatchersTestObjC, ObjCExceptionStmts) {
  StringRef ObjCString = "void f(id obj) {"
                         "  @try {"
                         "    @throw obj;"
                         "  } @catch (...) {"
                         "  } @finally {}"
                         "}";

  EXPECT_TRUE(matchesObjC(ObjCString, objcTryStmt()));
  EXPECT_TRUE(matchesObjC(ObjCString, objcThrowStmt()));
  EXPECT_TRUE(matchesObjC(ObjCString, objcCatchStmt()));
  EXPECT_TRUE(matchesObjC(ObjCString, objcFinallyStmt()));
}

TEST(ASTMatchersTest, DecompositionDecl) {
  StringRef Code = R"cpp(
void foo()
{
    int arr[3];
    auto &[f, s, t] = arr;

    f = 42;
}
  )cpp";
  EXPECT_TRUE(matchesConditionally(
      Code, decompositionDecl(hasBinding(0, bindingDecl(hasName("f")))), true,
      {"-std=c++17"}));
  EXPECT_FALSE(matchesConditionally(
      Code, decompositionDecl(hasBinding(42, bindingDecl(hasName("f")))), true,
      {"-std=c++17"}));
  EXPECT_FALSE(matchesConditionally(
      Code, decompositionDecl(hasBinding(0, bindingDecl(hasName("s")))), true,
      {"-std=c++17"}));
  EXPECT_TRUE(matchesConditionally(
      Code, decompositionDecl(hasBinding(1, bindingDecl(hasName("s")))), true,
      {"-std=c++17"}));

  EXPECT_TRUE(matchesConditionally(
      Code,
      bindingDecl(decl().bind("self"), hasName("f"),
                  forDecomposition(decompositionDecl(
                      hasAnyBinding(bindingDecl(equalsBoundNode("self")))))),
      true, {"-std=c++17"}));
}

TEST(ASTMatchersTestObjC, ObjCAutoreleasePoolStmt) {
  StringRef ObjCString = "void f() {"
                         "@autoreleasepool {"
                         "  int x = 1;"
                         "}"
                         "}";
  EXPECT_TRUE(matchesObjC(ObjCString, autoreleasePoolStmt()));
  StringRef ObjCStringNoPool = "void f() { int x = 1; }";
  EXPECT_FALSE(matchesObjC(ObjCStringNoPool, autoreleasePoolStmt()));
}

TEST(ASTMatchersTestOpenMP, OMPExecutableDirective) {
  auto Matcher = stmt(ompExecutableDirective());

  StringRef Source0 = R"(
void x() {
#pragma omp parallel
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source0, Matcher));

  StringRef Source1 = R"(
void x() {
#pragma omp taskyield
;
})";
  EXPECT_TRUE(matchesWithOpenMP(Source1, Matcher));

  StringRef Source2 = R"(
void x() {
;
})";
  EXPECT_TRUE(notMatchesWithOpenMP(Source2, Matcher));
}

TEST(ASTMatchersTestOpenMP, OMPDefaultClause) {
  auto Matcher = ompExecutableDirective(hasAnyClause(ompDefaultClause()));

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
  EXPECT_TRUE(notMatchesWithOpenMP(Source5, Matcher));
}

TEST(ASTMatchersTest, Finder_DynamicOnlyAcceptsSomeMatchers) {
  MatchFinder Finder;
  EXPECT_TRUE(Finder.addDynamicMatcher(decl(), nullptr));
  EXPECT_TRUE(Finder.addDynamicMatcher(callExpr(), nullptr));
  EXPECT_TRUE(
      Finder.addDynamicMatcher(constantArrayType(hasSize(42)), nullptr));

  // Do not accept non-toplevel matchers.
  EXPECT_FALSE(Finder.addDynamicMatcher(isMain(), nullptr));
  EXPECT_FALSE(Finder.addDynamicMatcher(hasName("x"), nullptr));
}

TEST(MatchFinderAPI, MatchesDynamic) {
  StringRef SourceCode = "struct A { void f() {} };";
  auto Matcher = functionDecl(isDefinition()).bind("method");

  auto astUnit = tooling::buildASTFromCode(SourceCode);

  auto GlobalBoundNodes = matchDynamic(Matcher, astUnit->getASTContext());

  EXPECT_EQ(GlobalBoundNodes.size(), 1u);
  EXPECT_EQ(GlobalBoundNodes[0].getMap().size(), 1u);

  auto GlobalMethodNode = GlobalBoundNodes[0].getNodeAs<FunctionDecl>("method");
  EXPECT_TRUE(GlobalMethodNode != nullptr);

  auto MethodBoundNodes =
      matchDynamic(Matcher, *GlobalMethodNode, astUnit->getASTContext());
  EXPECT_EQ(MethodBoundNodes.size(), 1u);
  EXPECT_EQ(MethodBoundNodes[0].getMap().size(), 1u);

  auto MethodNode = MethodBoundNodes[0].getNodeAs<FunctionDecl>("method");
  EXPECT_EQ(MethodNode, GlobalMethodNode);
}

static std::vector<TestClangConfig> allTestClangConfigs() {
  std::vector<TestClangConfig> all_configs;
  for (TestLanguage lang : {
#define TESTLANGUAGE(lang, version, std_flag, version_index)                   \
  Lang_##lang##version,
#include "clang/Testing/TestLanguage.def"
       }) {
    TestClangConfig config;
    config.Language = lang;

    // Use an unknown-unknown triple so we don't instantiate the full system
    // toolchain.  On Linux, instantiating the toolchain involves stat'ing
    // large portions of /usr/lib, and this slows down not only this test, but
    // all other tests, via contention in the kernel.
    //
    // FIXME: This is a hack to work around the fact that there's no way to do
    // the equivalent of runToolOnCodeWithArgs without instantiating a full
    // Driver.  We should consider having a function, at least for tests, that
    // invokes cc1.
    config.Target = "i386-unknown-unknown";
    all_configs.push_back(config);

    // Windows target is interesting to test because it enables
    // `-fdelayed-template-parsing`.
    config.Target = "x86_64-pc-win32-msvc";
    all_configs.push_back(config);
  }
  return all_configs;
}

INSTANTIATE_TEST_SUITE_P(
    ASTMatchersTests, ASTMatchersTest, testing::ValuesIn(allTestClangConfigs()),
    [](const testing::TestParamInfo<TestClangConfig> &Info) {
      return Info.param.toShortString();
    });

} // namespace ast_matchers
} // namespace clang
