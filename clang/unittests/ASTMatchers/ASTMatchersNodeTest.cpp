//== unittests/ASTMatchers/ASTMatchersNodeTest.cpp - AST matcher unit tests ==//
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

TEST(Finder, DynamicOnlyAcceptsSomeMatchers) {
  MatchFinder Finder;
  EXPECT_TRUE(Finder.addDynamicMatcher(decl(), nullptr));
  EXPECT_TRUE(Finder.addDynamicMatcher(callExpr(), nullptr));
  EXPECT_TRUE(Finder.addDynamicMatcher(constantArrayType(hasSize(42)),
                                       nullptr));

  // Do not accept non-toplevel matchers.
  EXPECT_FALSE(Finder.addDynamicMatcher(isArrow(), nullptr));
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

  // This passes on Windows only because we explicitly pass -target
  // i386-unknown-unknown.  If we were to compile with the default target
  // triple, we'd want to EXPECT_TRUE if it's Win32 or MSVC.
  EXPECT_FALSE(matches("", ClassMatcher));

  DeclarationMatcher ClassX = recordDecl(recordDecl(hasName("X")));
  EXPECT_TRUE(matches("class X;", ClassX));
  EXPECT_TRUE(matches("class X {};", ClassX));
  EXPECT_TRUE(matches("template<class T> class X {};", ClassX));
  EXPECT_TRUE(notMatches("", ClassX));
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

TEST(Matcher, UnresolvedLookupExpr) {
  // FIXME: The test is known to be broken on Windows with delayed template
  // parsing.
  EXPECT_TRUE(matchesConditionally("template<typename T>"
                                   "T foo() { T a; return a; }"
                                   "template<typename T>"
                                   "void bar() {"
                                   "  foo<T>();"
                                   "}",
                                   unresolvedLookupExpr(),
                                   /*ExpectMatch=*/true,
                                   "-fno-delayed-template-parsing"));
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
  EXPECT_TRUE(matches("void f() { goto FOO; FOO: ;}",
                      labelStmt(
                        hasDeclaration(
                          labelDecl(hasName("FOO"))))));
  EXPECT_TRUE(matches("void f() { FOO: ; void *ptr = &&FOO; goto *ptr; }",
                      addrLabelExpr()));
  EXPECT_TRUE(matches("void f() { return; }", returnStmt()));
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

TEST(Match, ConstructorInitializers) {
  EXPECT_TRUE(matches("class C { int i; public: C(int ii) : i(ii) {} };",
                      cxxCtorInitializer(forField(hasName("i")))));
}

TEST(Matcher, ThisExpr) {
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
    matches(ClassString +
                 "string GetStringByValue();"
                   "void run() { int k = GetStringByValue().length(); }",
               materializeTemporaryExpr()));

  EXPECT_TRUE(
    notMatches(ClassString +
                 "string GetStringByValue();"
                   "void run() { GetStringByValue(); }",
               materializeTemporaryExpr()));
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

TEST(Matcher, AtomicExpr) {
  EXPECT_TRUE(matches("void foo() { int *ptr; __atomic_load_n(ptr, 1); }",
                      atomicExpr()));
}

TEST(Matcher, Initializers) {
  const char *ToMatch = "void foo() { struct point { double x; double y; };"
    "  struct point ptarray[10] = "
    "      { [2].y = 1.0, [2].x = 2.0, [0].x = 1.0 }; }";
  EXPECT_TRUE(matchesConditionally(
    ToMatch,
    initListExpr(
      has(
        cxxConstructExpr(
          requiresZeroInitialization())),
      has(
        initListExpr(
          hasType(asString("struct point")),
          has(floatLiteral(equals(1.0))),
          has(implicitValueInitExpr(
            hasType(asString("double")))))),
      has(
        initListExpr(
          hasType(asString("struct point")),
          has(floatLiteral(equals(2.0))),
          has(floatLiteral(equals(1.0)))))
    ), true, "-std=gnu++98"));

  EXPECT_TRUE(matchesC99(ToMatch,
                         initListExpr(
                           hasSyntacticForm(
                             initListExpr(
                               has(
                                 designatedInitExpr(
                                   designatorCountIs(2),
                                   has(floatLiteral(
                                     equals(1.0))),
                                   has(integerLiteral(
                                     equals(2))))),
                               has(
                                 designatedInitExpr(
                                   designatorCountIs(2),
                                   has(floatLiteral(
                                     equals(2.0))),
                                   has(integerLiteral(
                                     equals(2))))),
                               has(
                                 designatedInitExpr(
                                   designatorCountIs(2),
                                   has(floatLiteral(
                                     equals(1.0))),
                                   has(integerLiteral(
                                     equals(0)))))
                             )))));
}

TEST(Matcher, ParenListExpr) {
  EXPECT_TRUE(
    matches("template<typename T> class foo { void bar() { foo X(*this); } };"
              "template class foo<int>;",
            varDecl(hasInitializer(parenListExpr(has(unaryOperator()))))));
}

TEST(Matcher, StmtExpr) {
  EXPECT_TRUE(matches("void declToImport() { int C = ({int X=4; X;}); }",
                      varDecl(hasInitializer(stmtExpr()))));
}

TEST(Matcher, ImportPredefinedExpr) {
  // __func__ expands as StringLiteral("foo")
  EXPECT_TRUE(matches("void foo() { __func__; }",
                      predefinedExpr(
                        hasType(asString("const char [4]")),
                        has(stringLiteral()))));
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

  EXPECT_TRUE(matches("void x() { true ? true : false; }", ConditionalFalse));
  EXPECT_TRUE(
    notMatches("void x() { true ? false : true; }", ConditionalFalse));
}

TEST(Matcher, BinaryConditionalOperator) {
  StatementMatcher AlwaysOne = binaryConditionalOperator(
    hasCondition(implicitCastExpr(
      has(
        opaqueValueExpr(
          hasSourceExpression((integerLiteral(equals(1)))))))),
    hasFalseExpression(integerLiteral(equals(0))));

  EXPECT_TRUE(matches("void x() { 1 ?: 0; }", AlwaysOne));

  StatementMatcher FourNotFive = binaryConditionalOperator(
    hasTrueExpression(opaqueValueExpr(
      hasSourceExpression((integerLiteral(equals(4)))))),
    hasFalseExpression(integerLiteral(equals(5))));

  EXPECT_TRUE(matches("void x() { 4 ?: 5; }", FourNotFive));
}

TEST(ArraySubscriptMatchers, ArraySubscripts) {
  EXPECT_TRUE(matches("int i[2]; void f() { i[1] = 1; }",
                      arraySubscriptExpr()));
  EXPECT_TRUE(notMatches("int i; void f() { i = 1; }",
                         arraySubscriptExpr()));
}

TEST(For, FindsForLoops) {
  EXPECT_TRUE(matches("void f() { for(;;); }", forStmt()));
  EXPECT_TRUE(matches("void f() { if(true) for(;;); }", forStmt()));
  EXPECT_TRUE(notMatches("int as[] = { 1, 2, 3 };"
                           "void f() { for (auto &a : as); }",
                         forStmt()));
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
  EXPECT_FALSE(matches("struct Foo { }; Foo a;"
                       "const Foo f = a;",
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

TEST(CXXStdInitializerListExpression, MatchesCXXStdInitializerListExpression) {
  const std::string code = "namespace std {"
                           "template <typename> class initializer_list {"
                           "  public: initializer_list() noexcept {}"
                           "};"
                           "}"
                           "struct A {"
                           "  A(std::initializer_list<int>) {}"
                           "};";
  EXPECT_TRUE(matches(code + "A a{0};",
                      cxxConstructExpr(has(cxxStdInitializerListExpr()),
                                       hasDeclaration(cxxConstructorDecl(
                                           ofClass(hasName("A")))))));
  EXPECT_TRUE(matches(code + "A a = {0};",
                      cxxConstructExpr(has(cxxStdInitializerListExpr()),
                                       hasDeclaration(cxxConstructorDecl(
                                           ofClass(hasName("A")))))));

  EXPECT_TRUE(notMatches("int a[] = { 1, 2 };", cxxStdInitializerListExpr()));
  EXPECT_TRUE(notMatches("struct B { int x, y; }; B b = { 5, 6 };",
                         cxxStdInitializerListExpr()));
}

TEST(UsingDeclaration, MatchesUsingDeclarations) {
  EXPECT_TRUE(matches("namespace X { int x; } using X::x;",
                      usingDecl()));
}

TEST(UsingDeclaration, MatchesShadowUsingDelcarations) {
  EXPECT_TRUE(matches("namespace f { int a; } using f::a;",
                      usingDecl(hasAnyUsingShadowDecl(hasName("a")))));
}

TEST(UsingDirectiveDeclaration, MatchesUsingNamespace) {
  EXPECT_TRUE(matches("namespace X { int x; } using namespace X;",
                      usingDirectiveDecl()));
  EXPECT_FALSE(
    matches("namespace X { int x; } using X::x;", usingDirectiveDecl()));
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

TEST(TypeMatching, MatchesTypes) {
  EXPECT_TRUE(matches("struct S {};", qualType().bind("loc")));
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

TEST(TypeMatching, IgnoringParens) {
  EXPECT_TRUE(
      notMatches("void (*fp)(void);", pointerType(pointee(functionType()))));
  EXPECT_TRUE(matches("void (*fp)(void);",
                      pointerType(pointee(ignoringParens(functionType())))));
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
  //    llvm::make_unique<VerifyIdIsBoundTo<TypeLoc>>("loc", 1)));
  //EXPECT_TRUE(matchAndVerifyResultTrue(
  //    "int* a;",
  //    pointerTypeLoc().bind("loc"),
  //    llvm::make_unique<VerifyIdIsBoundTo<TypeLoc>>("loc", 1)));
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

TEST(TypeMatching, MatchesEnumTypes) {
  EXPECT_TRUE(matches("enum Color { Green }; Color color;",
                      loc(enumType())));
  EXPECT_TRUE(matches("enum class Color { Green }; Color color;",
                      loc(enumType())));
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

TEST(NNS, MatchesNestedNameSpecifierPrefixes) {
  EXPECT_TRUE(matches(
    "struct A { struct B { struct C {}; }; }; A::B::C c;",
    nestedNameSpecifier(hasPrefix(specifiesType(asString("struct A"))))));
  EXPECT_TRUE(matches(
    "struct A { struct B { struct C {}; }; }; A::B::C c;",
    nestedNameSpecifierLoc(hasPrefix(
      specifiesTypeLoc(loc(qualType(asString("struct A"))))))));
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
    llvm::make_unique<VerifyAncestorHasChildIsEqual<CXXRecordDecl>>()));
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "void f() { if (true) if(true) {} }", ifStmt().bind(""),
    llvm::make_unique<VerifyAncestorHasChildIsEqual<IfStmt>>()));
  EXPECT_TRUE(matchAndVerifyResultTrue(
    "class X { class Y {} y; };",
    fieldDecl(hasName("y"), hasType(type().bind(""))).bind("decl"),
    llvm::make_unique<VerifyAncestorHasChildIsEqual<Type>>()));
}

TEST(TypedefDeclMatcher, Match) {
  EXPECT_TRUE(matches("typedef int typedefDeclTest;",
                      typedefDecl(hasName("typedefDeclTest"))));
  EXPECT_TRUE(notMatches("using typedefDeclTest2 = int;",
                         typedefDecl(hasName("typedefDeclTest2"))));
}

TEST(TypeAliasDeclMatcher, Match) {
  EXPECT_TRUE(matches("using typeAliasTest2 = int;",
                      typeAliasDecl(hasName("typeAliasTest2"))));
  EXPECT_TRUE(notMatches("typedef int typeAliasTest;",
                         typeAliasDecl(hasName("typeAliasTest"))));
}

TEST(TypedefNameDeclMatcher, Match) {
  EXPECT_TRUE(matches("typedef int typedefNameDeclTest1;",
                      typedefNameDecl(hasName("typedefNameDeclTest1"))));
  EXPECT_TRUE(matches("using typedefNameDeclTest2 = int;",
                      typedefNameDecl(hasName("typedefNameDeclTest2"))));
}

TEST(TypeAliasTemplateDeclMatcher, Match) {
  std::string Code = R"(
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

TEST(ObjCMessageExprMatcher, SimpleExprs) {
  // don't find ObjCMessageExpr where none are present
  EXPECT_TRUE(notMatchesObjC("", objcMessageExpr(anything())));

  std::string Objc1String =
    "@interface Str "
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

TEST(ObjCDeclMacher, CoreDecls) {
  std::string ObjCString =
    "@protocol Proto "
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
    ;

  EXPECT_TRUE(matchesObjC(
    ObjCString,
    objcProtocolDecl(hasName("Proto"))));
  EXPECT_TRUE(matchesObjC(
    ObjCString,
    objcCategoryDecl(hasName("ABC"))));
  EXPECT_TRUE(matchesObjC(
    ObjCString,
    objcMethodDecl(hasName("protoDidThing"))));
  EXPECT_TRUE(matchesObjC(
    ObjCString,
    objcMethodDecl(hasName("abc_doThing"))));
  EXPECT_TRUE(matchesObjC(
    ObjCString,
    objcMethodDecl(hasName("anything"))));
  EXPECT_TRUE(matchesObjC(
    ObjCString,
    objcIvarDecl(hasName("_ivar"))));
  EXPECT_TRUE(matchesObjC(
    ObjCString,
    objcPropertyDecl(hasName("enabled"))));
}

} // namespace ast_matchers
} // namespace clang
