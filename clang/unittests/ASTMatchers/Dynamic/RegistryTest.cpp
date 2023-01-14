//===- unittest/ASTMatchers/Dynamic/RegistryTest.cpp - Registry unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//

#include "../ASTMatchersTest.h"
#include "clang/ASTMatchers/Dynamic/Registry.h"
#include "gtest/gtest.h"
#include <optional>
#include <vector>

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

using ast_matchers::internal::Matcher;

class RegistryTest : public ::testing::Test {
public:
  std::vector<ParserValue> Args() { return std::vector<ParserValue>(); }
  std::vector<ParserValue> Args(const VariantValue &Arg1) {
    std::vector<ParserValue> Out(1);
    Out[0].Value = Arg1;
    return Out;
  }
  std::vector<ParserValue> Args(const VariantValue &Arg1,
                                const VariantValue &Arg2) {
    std::vector<ParserValue> Out(2);
    Out[0].Value = Arg1;
    Out[1].Value = Arg2;
    return Out;
  }

  llvm::Optional<MatcherCtor> lookupMatcherCtor(StringRef MatcherName) {
    return Registry::lookupMatcherCtor(MatcherName);
  }

  VariantMatcher constructMatcher(StringRef MatcherName,
                                  Diagnostics *Error = nullptr) {
    Diagnostics DummyError;
    if (!Error) Error = &DummyError;
    llvm::Optional<MatcherCtor> Ctor = lookupMatcherCtor(MatcherName);
    VariantMatcher Out;
    if (Ctor)
      Out = Registry::constructMatcher(*Ctor, SourceRange(), Args(), Error);
    EXPECT_EQ("", DummyError.toStringFull());
    return Out;
  }

  VariantMatcher constructMatcher(StringRef MatcherName,
                                  const VariantValue &Arg1,
                                  Diagnostics *Error = nullptr) {
    Diagnostics DummyError;
    if (!Error) Error = &DummyError;
    llvm::Optional<MatcherCtor> Ctor = lookupMatcherCtor(MatcherName);
    VariantMatcher Out;
    if (Ctor)
      Out = Registry::constructMatcher(*Ctor, SourceRange(), Args(Arg1), Error);
    EXPECT_EQ("", DummyError.toStringFull()) << MatcherName;
    return Out;
  }

  VariantMatcher constructMatcher(StringRef MatcherName,
                                  const VariantValue &Arg1,
                                  const VariantValue &Arg2,
                                  Diagnostics *Error = nullptr) {
    Diagnostics DummyError;
    if (!Error) Error = &DummyError;
    llvm::Optional<MatcherCtor> Ctor = lookupMatcherCtor(MatcherName);
    VariantMatcher Out;
    if (Ctor)
      Out = Registry::constructMatcher(*Ctor, SourceRange(), Args(Arg1, Arg2),
                                       Error);
    EXPECT_EQ("", DummyError.toStringFull());
    return Out;
  }

  typedef std::vector<MatcherCompletion> CompVector;

  CompVector getCompletions() {
    std::vector<std::pair<MatcherCtor, unsigned> > Context;
    return Registry::getMatcherCompletions(
        Registry::getAcceptedCompletionTypes(Context));
  }

  CompVector getCompletions(StringRef MatcherName1, unsigned ArgNo1) {
    std::vector<std::pair<MatcherCtor, unsigned> > Context;
    llvm::Optional<MatcherCtor> Ctor = lookupMatcherCtor(MatcherName1);
    if (!Ctor)
      return CompVector();
    Context.push_back(std::make_pair(*Ctor, ArgNo1));
    return Registry::getMatcherCompletions(
        Registry::getAcceptedCompletionTypes(Context));
  }

  CompVector getCompletions(StringRef MatcherName1, unsigned ArgNo1,
                            StringRef MatcherName2, unsigned ArgNo2) {
    std::vector<std::pair<MatcherCtor, unsigned> > Context;
    llvm::Optional<MatcherCtor> Ctor = lookupMatcherCtor(MatcherName1);
    if (!Ctor)
      return CompVector();
    Context.push_back(std::make_pair(*Ctor, ArgNo1));
    Ctor = lookupMatcherCtor(MatcherName2);
    if (!Ctor)
      return CompVector();
    Context.push_back(std::make_pair(*Ctor, ArgNo2));
    return Registry::getMatcherCompletions(
        Registry::getAcceptedCompletionTypes(Context));
  }

  bool hasCompletion(const CompVector &Comps, StringRef TypedText,
                     StringRef MatcherDecl = StringRef()) {
    for (CompVector::const_iterator I = Comps.begin(), E = Comps.end(); I != E;
         ++I) {
      if (I->TypedText == TypedText &&
          (MatcherDecl.empty() || I->MatcherDecl == MatcherDecl)) {
        return true;
      }
    }
    return false;
  }
};

TEST_F(RegistryTest, CanConstructNoArgs) {
  Matcher<Stmt> IsArrowValue = constructMatcher(
      "memberExpr", constructMatcher("isArrow")).getTypedMatcher<Stmt>();
  Matcher<Stmt> BoolValue =
      constructMatcher("cxxBoolLiteral").getTypedMatcher<Stmt>();

  const std::string ClassSnippet = "struct Foo { int x; };\n"
                                   "Foo *foo = new Foo;\n"
                                   "int i = foo->x;\n";
  const std::string BoolSnippet = "bool Foo = true;\n";

  EXPECT_TRUE(matches(ClassSnippet, IsArrowValue));
  EXPECT_TRUE(matches(BoolSnippet, BoolValue));
  EXPECT_FALSE(matches(ClassSnippet, BoolValue));
  EXPECT_FALSE(matches(BoolSnippet, IsArrowValue));
}

TEST_F(RegistryTest, ConstructWithSimpleArgs) {
  Matcher<Decl> Value = constructMatcher(
      "namedDecl", constructMatcher("hasName", StringRef("X")))
      .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("class X {};", Value));
  EXPECT_FALSE(matches("int x;", Value));

  Value = functionDecl(constructMatcher("parameterCountIs", 2)
                           .getTypedMatcher<FunctionDecl>());
  EXPECT_TRUE(matches("void foo(int,int);", Value));
  EXPECT_FALSE(matches("void foo(int);", Value));
}

TEST_F(RegistryTest, ConstructWithMatcherArgs) {
  Matcher<Decl> HasInitializerSimple = constructMatcher(
      "varDecl", constructMatcher("hasInitializer", constructMatcher("stmt")))
      .getTypedMatcher<Decl>();
  Matcher<Decl> HasInitializerComplex = constructMatcher(
      "varDecl",
      constructMatcher("hasInitializer", constructMatcher("callExpr")))
      .getTypedMatcher<Decl>();

  std::string code = "int i;";
  EXPECT_FALSE(matches(code, HasInitializerSimple));
  EXPECT_FALSE(matches(code, HasInitializerComplex));

  code = "int i = 1;";
  EXPECT_TRUE(matches(code, HasInitializerSimple));
  EXPECT_FALSE(matches(code, HasInitializerComplex));

  code = "int y(); int i = y();";
  EXPECT_TRUE(matches(code, HasInitializerSimple));
  EXPECT_TRUE(matches(code, HasInitializerComplex));

  Matcher<Decl> HasParameter =
      functionDecl(constructMatcher(
          "hasParameter", 1, constructMatcher("hasName", StringRef("x")))
                       .getTypedMatcher<FunctionDecl>());
  EXPECT_TRUE(matches("void f(int a, int x);", HasParameter));
  EXPECT_FALSE(matches("void f(int x, int a);", HasParameter));
}

TEST_F(RegistryTest, OverloadedMatchers) {
  Matcher<Stmt> CallExpr0 = constructMatcher(
      "callExpr",
      constructMatcher("callee", constructMatcher("memberExpr",
                                                  constructMatcher("isArrow"))))
      .getTypedMatcher<Stmt>();

  Matcher<Stmt> CallExpr1 = constructMatcher(
      "callExpr",
      constructMatcher(
          "callee",
          constructMatcher("cxxMethodDecl",
                           constructMatcher("hasName", StringRef("x")))))
      .getTypedMatcher<Stmt>();

  Matcher<Stmt> ObjCMsgExpr =
      constructMatcher(
          "objcMessageExpr",
          constructMatcher(
              "callee",
              constructMatcher("objcMethodDecl",
                               constructMatcher("hasName", StringRef("x")))))
          .getTypedMatcher<Stmt>();

  std::string Code = "class Y { public: void x(); }; void z() { Y y; y.x(); }";
  EXPECT_FALSE(matches(Code, CallExpr0));
  EXPECT_TRUE(matches(Code, CallExpr1));
  EXPECT_FALSE(matches(Code, ObjCMsgExpr));

  Code = "class Z { public: void z() { this->z(); } };";
  EXPECT_TRUE(matches(Code, CallExpr0));
  EXPECT_FALSE(matches(Code, CallExpr1));
  EXPECT_FALSE(matches(Code, ObjCMsgExpr));

  Code = "@interface I "
         "+ (void)x; "
         "@end\n"
         "int main() { [I x]; }";
  EXPECT_FALSE(matchesObjC(Code, CallExpr0));
  EXPECT_FALSE(matchesObjC(Code, CallExpr1));
  EXPECT_TRUE(matchesObjC(Code, ObjCMsgExpr));

  Matcher<Decl> DeclDecl = declaratorDecl(hasTypeLoc(
      constructMatcher(
          "loc", constructMatcher("asString", StringRef("const double *")))
          .getTypedMatcher<TypeLoc>()));

  Matcher<NestedNameSpecifierLoc> NNSL =
      constructMatcher(
          "loc", VariantMatcher::SingleMatcher(nestedNameSpecifier(
                     specifiesType(hasDeclaration(recordDecl(hasName("A")))))))
          .getTypedMatcher<NestedNameSpecifierLoc>();

  Code = "const double * x = 0;";
  EXPECT_TRUE(matches(Code, DeclDecl));
  EXPECT_FALSE(matches(Code, NNSL));

  Code = "struct A { struct B {}; }; A::B a_b;";
  EXPECT_FALSE(matches(Code, DeclDecl));
  EXPECT_TRUE(matches(Code, NNSL));
}

TEST_F(RegistryTest, PolymorphicMatchers) {
  const VariantMatcher IsDefinition = constructMatcher("isDefinition");
  Matcher<Decl> Var =
      constructMatcher("varDecl", IsDefinition).getTypedMatcher<Decl>();
  Matcher<Decl> Class =
      constructMatcher("recordDecl", IsDefinition).getTypedMatcher<Decl>();
  Matcher<Decl> Func =
      constructMatcher("functionDecl", IsDefinition).getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("int a;", Var));
  EXPECT_FALSE(matches("extern int a;", Var));
  EXPECT_TRUE(matches("class A {};", Class));
  EXPECT_FALSE(matches("class A;", Class));
  EXPECT_TRUE(matches("void f(){};", Func));
  EXPECT_FALSE(matches("void f();", Func));

  Matcher<Decl> Anything = constructMatcher("anything").getTypedMatcher<Decl>();
  Matcher<Decl> RecordDecl = constructMatcher(
      "recordDecl", constructMatcher("hasName", StringRef("Foo")),
      VariantMatcher::SingleMatcher(Anything)).getTypedMatcher<Decl>();

  EXPECT_TRUE(matches("int Foo;", Anything));
  EXPECT_TRUE(matches("class Foo {};", Anything));
  EXPECT_TRUE(matches("void Foo(){};", Anything));
  EXPECT_FALSE(matches("int Foo;", RecordDecl));
  EXPECT_TRUE(matches("class Foo {};", RecordDecl));
  EXPECT_FALSE(matches("void Foo(){};", RecordDecl));

  Matcher<Stmt> ConstructExpr = constructMatcher(
      "cxxConstructExpr",
      constructMatcher(
          "hasDeclaration",
          constructMatcher(
              "cxxMethodDecl",
              constructMatcher(
                  "ofClass", constructMatcher("hasName", StringRef("Foo"))))))
                                    .getTypedMatcher<Stmt>();
  EXPECT_FALSE(matches("class Foo { public: Foo(); };", ConstructExpr));
  EXPECT_TRUE(
      matches("class Foo { public: Foo(); }; Foo foo = Foo();", ConstructExpr));
}

TEST_F(RegistryTest, TemplateArgument) {
  Matcher<Decl> HasTemplateArgument = constructMatcher(
      "classTemplateSpecializationDecl",
      constructMatcher(
          "hasAnyTemplateArgument",
          constructMatcher("refersToType",
                           constructMatcher("asString", StringRef("int")))))
      .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("template<typename T> class A {}; A<int> a;",
                      HasTemplateArgument));
  EXPECT_FALSE(matches("template<typename T> class A {}; A<char> a;",
                       HasTemplateArgument));
}

TEST_F(RegistryTest, TypeTraversal) {
  Matcher<Type> M = constructMatcher(
      "pointerType",
      constructMatcher("pointee", constructMatcher("isConstQualified"),
                       constructMatcher("isInteger"))).getTypedMatcher<Type>();
  EXPECT_FALSE(matches("int *a;", M));
  EXPECT_TRUE(matches("int const *b;", M));

  M = constructMatcher(
      "arrayType",
      constructMatcher("hasElementType", constructMatcher("builtinType")))
      .getTypedMatcher<Type>();
  EXPECT_FALSE(matches("struct A{}; A a[7];;", M));
  EXPECT_TRUE(matches("int b[7];", M));
}

TEST_F(RegistryTest, CXXBaseSpecifier) {
  // TODO: rewrite with top-level cxxBaseSpecifier matcher when available
  DeclarationMatcher ClassHasAnyDirectBase =
      constructMatcher("cxxRecordDecl",
                       constructMatcher("hasDirectBase",
                                        constructMatcher("cxxBaseSpecifier")))
          .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("class X {}; class Y : X {};", ClassHasAnyDirectBase));
  EXPECT_TRUE(notMatches("class X {};", ClassHasAnyDirectBase));
}

TEST_F(RegistryTest, CXXCtorInitializer) {
  Matcher<Decl> CtorDecl = constructMatcher(
      "cxxConstructorDecl",
      constructMatcher(
          "hasAnyConstructorInitializer",
          constructMatcher("forField",
                           constructMatcher("hasName", StringRef("foo")))))
      .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("struct Foo { Foo() : foo(1) {} int foo; };", CtorDecl));
  EXPECT_FALSE(matches("struct Foo { Foo() {} int foo; };", CtorDecl));
  EXPECT_FALSE(matches("struct Foo { Foo() : bar(1) {} int bar; };", CtorDecl));
}

TEST_F(RegistryTest, Adaptative) {
  Matcher<Decl> D = constructMatcher(
      "recordDecl",
      constructMatcher(
          "has",
          constructMatcher("recordDecl",
                           constructMatcher("hasName", StringRef("X")))))
      .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("class X {};", D));
  EXPECT_TRUE(matches("class Y { class X {}; };", D));
  EXPECT_FALSE(matches("class Y { class Z {}; };", D));

  Matcher<Stmt> S = constructMatcher(
      "forStmt",
      constructMatcher(
          "hasDescendant",
          constructMatcher("varDecl",
                           constructMatcher("hasName", StringRef("X")))))
      .getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("void foo() { for(int X;;); }", S));
  EXPECT_TRUE(matches("void foo() { for(;;) { int X; } }", S));
  EXPECT_FALSE(matches("void foo() { for(;;); }", S));
  EXPECT_FALSE(matches("void foo() { if (int X = 0){} }", S));

  S = constructMatcher(
      "compoundStmt", constructMatcher("hasParent", constructMatcher("ifStmt")))
      .getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("void foo() { if (true) { int x = 42; } }", S));
  EXPECT_FALSE(matches("void foo() { if (true) return; }", S));
}

TEST_F(RegistryTest, VariadicOp) {
  Matcher<Decl> D = constructMatcher(
      "anyOf",
      constructMatcher("recordDecl",
                       constructMatcher("hasName", StringRef("Foo"))),
      constructMatcher("functionDecl",
                       constructMatcher("hasName", StringRef("foo"))))
      .getTypedMatcher<Decl>();

  EXPECT_TRUE(matches("void foo(){}", D));
  EXPECT_TRUE(matches("struct Foo{};", D));
  EXPECT_FALSE(matches("int i = 0;", D));

  D = constructMatcher(
      "allOf", constructMatcher("recordDecl"),
      constructMatcher(
          "namedDecl",
          constructMatcher("anyOf",
                           constructMatcher("hasName", StringRef("Foo")),
                           constructMatcher("hasName", StringRef("Bar")))))
      .getTypedMatcher<Decl>();

  EXPECT_FALSE(matches("void foo(){}", D));
  EXPECT_TRUE(matches("struct Foo{};", D));
  EXPECT_FALSE(matches("int i = 0;", D));
  EXPECT_TRUE(matches("class Bar{};", D));
  EXPECT_FALSE(matches("class OtherBar{};", D));

  D = recordDecl(
      has(fieldDecl(hasName("Foo"))),
      constructMatcher(
          "unless",
          constructMatcher("namedDecl",
                           constructMatcher("hasName", StringRef("Bar"))))
          .getTypedMatcher<Decl>());

  EXPECT_FALSE(matches("class Bar{ int Foo; };", D));
  EXPECT_TRUE(matches("class OtherBar{ int Foo; };", D));

  D = constructMatcher(
          "namedDecl", constructMatcher("hasName", StringRef("Foo")),
          constructMatcher("unless", constructMatcher("recordDecl")))
          .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("void Foo(){}", D));
  EXPECT_TRUE(notMatches("struct Foo {};", D));
}

TEST_F(RegistryTest, Errors) {
  // Incorrect argument count.
  std::unique_ptr<Diagnostics> Error(new Diagnostics());
  EXPECT_TRUE(constructMatcher("hasInitializer", Error.get()).isNull());
  EXPECT_EQ("Incorrect argument count. (Expected = 1) != (Actual = 0)",
            Error->toString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher("isArrow", StringRef(), Error.get()).isNull());
  EXPECT_EQ("Incorrect argument count. (Expected = 0) != (Actual = 1)",
            Error->toString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher("anyOf", Error.get()).isNull());
  EXPECT_EQ("Incorrect argument count. (Expected = (2, )) != (Actual = 0)",
            Error->toString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher("unless", StringRef(), StringRef(),
                               Error.get()).isNull());
  EXPECT_EQ("Incorrect argument count. (Expected = (1, 1)) != (Actual = 2)",
            Error->toString());

  // Bad argument type
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher("ofClass", StringRef(), Error.get()).isNull());
  EXPECT_EQ("Incorrect type for arg 1. (Expected = Matcher<CXXRecordDecl>) != "
            "(Actual = String)",
            Error->toString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(
      constructMatcher("cxxRecordDecl", constructMatcher("cxxRecordDecl"),
                       constructMatcher("parameterCountIs", 3), Error.get())
          .isNull());
  EXPECT_EQ("Incorrect type for arg 2. (Expected = Matcher<CXXRecordDecl>) != "
            "(Actual = Matcher<FunctionDecl|FunctionProtoType>)",
            Error->toString());

  // Bad argument type with variadic.
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher("anyOf", StringRef(), StringRef(),
                               Error.get()).isNull());
  EXPECT_EQ(
      "Incorrect type for arg 1. (Expected = Matcher<>) != (Actual = String)",
      Error->toString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher(
      "cxxRecordDecl",
      constructMatcher("allOf",
                       constructMatcher("isDerivedFrom", StringRef("FOO")),
                       constructMatcher("isArrow")),
      Error.get()).isNull());
  EXPECT_EQ("Incorrect type for arg 1. "
            "(Expected = Matcher<CXXRecordDecl>) != "
            "(Actual = Matcher<CXXRecordDecl|ObjCInterfaceDecl>&Matcher"
            "<MemberExpr|UnresolvedMemberExpr|CXXDependentScopeMemberExpr>)",
            Error->toString());
}

TEST_F(RegistryTest, Completion) {
  CompVector Comps = getCompletions();
  // Overloaded
  EXPECT_TRUE(hasCompletion(
      Comps, "hasParent(",
      "Matcher<NestedNameSpecifierLoc|TypeLoc|Decl|...> "
      "hasParent(Matcher<NestedNameSpecifierLoc|TypeLoc|Decl|...>)"));
  // Variadic.
  EXPECT_TRUE(hasCompletion(Comps, "whileStmt(",
                            "Matcher<Stmt> whileStmt(Matcher<WhileStmt>...)"));
  // Polymorphic.
  EXPECT_TRUE(hasCompletion(
      Comps, "hasDescendant(",
      "Matcher<NestedNameSpecifierLoc|QualType|TypeLoc|...> "
      "hasDescendant(Matcher<NestedNameSpecifierLoc|QualType|TypeLoc|...>)"));

  CompVector WhileComps = getCompletions("whileStmt", 0);

  EXPECT_TRUE(hasCompletion(WhileComps, "hasBody(",
                            "Matcher<WhileStmt> hasBody(Matcher<Stmt>)"));
  EXPECT_TRUE(hasCompletion(
      WhileComps, "hasParent(",
      "Matcher<Stmt> "
      "hasParent(Matcher<NestedNameSpecifierLoc|TypeLoc|Decl|...>)"));
  EXPECT_TRUE(
      hasCompletion(WhileComps, "allOf(", "Matcher<T> allOf(Matcher<T>...)"));

  EXPECT_FALSE(hasCompletion(WhileComps, "whileStmt("));
  EXPECT_FALSE(hasCompletion(WhileComps, "ifStmt("));

  CompVector AllOfWhileComps =
      getCompletions("allOf", 0, "whileStmt", 0);
  ASSERT_EQ(AllOfWhileComps.size(), WhileComps.size());
  EXPECT_TRUE(std::equal(WhileComps.begin(), WhileComps.end(),
                         AllOfWhileComps.begin()));

  CompVector DeclWhileComps =
      getCompletions("decl", 0, "whileStmt", 0);
  EXPECT_EQ(0u, DeclWhileComps.size());

  CompVector NamedDeclComps = getCompletions("namedDecl", 0);
  EXPECT_TRUE(
      hasCompletion(NamedDeclComps, "isPublic()", "Matcher<Decl> isPublic()"));
  EXPECT_TRUE(hasCompletion(NamedDeclComps, "hasName(\"",
                            "Matcher<NamedDecl> hasName(string)"));

  // Heterogeneous overloads.
  Comps = getCompletions("classTemplateSpecializationDecl", 0);
  EXPECT_TRUE(hasCompletion(
      Comps, "isSameOrDerivedFrom(",
      "Matcher<CXXRecordDecl> isSameOrDerivedFrom(string|Matcher<NamedDecl>)"));
}

TEST_F(RegistryTest, MatcherBuilder) {
  auto Ctor = *lookupMatcherCtor("mapAnyOf");
  EXPECT_TRUE(Registry::isBuilderMatcher(Ctor));
  auto BuiltCtor = Registry::buildMatcherCtor(Ctor, {}, Args(ASTNodeKind::getFromNodeKind<WhileStmt>(), ASTNodeKind::getFromNodeKind<ForStmt>()), nullptr);
  EXPECT_TRUE(BuiltCtor.get());
  auto LoopMatcher = Registry::constructMatcher(BuiltCtor.get(), SourceRange(), Args(), nullptr).getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("void f() { for (;;) {} }", LoopMatcher));
  EXPECT_TRUE(matches("void f() { while (true) {} }", LoopMatcher));
  EXPECT_FALSE(matches("void f() { if (true) {} }", LoopMatcher));

  auto NotBuiltCtor = Registry::buildMatcherCtor(Ctor, {}, Args(ASTNodeKind::getFromNodeKind<FunctionDecl>(), ASTNodeKind::getFromNodeKind<ForStmt>()), nullptr);
  EXPECT_FALSE(NotBuiltCtor.get());
}

TEST_F(RegistryTest, NodeType) {
  EXPECT_TRUE(Registry::nodeMatcherType(*lookupMatcherCtor("callExpr")).isSame(ASTNodeKind::getFromNodeKind<CallExpr>()));
  EXPECT_TRUE(Registry::nodeMatcherType(*lookupMatcherCtor("has")).isNone());
  EXPECT_TRUE(Registry::nodeMatcherType(*lookupMatcherCtor("allOf")).isNone());
}

TEST_F(RegistryTest, HasArgs) {
  Matcher<Decl> Value = constructMatcher(
      "decl", constructMatcher("hasAttr", StringRef("attr::WarnUnused")))
      .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("struct __attribute__((warn_unused)) X {};", Value));
  EXPECT_FALSE(matches("struct X {};", Value));
}

TEST_F(RegistryTest, ParenExpr) {
  Matcher<Stmt> Value = constructMatcher("parenExpr").getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("int i = (1);", traverse(TK_AsIs, Value)));
  EXPECT_FALSE(matches("int i = 1;", traverse(TK_AsIs, Value)));
}

TEST_F(RegistryTest, EqualsMatcher) {
  Matcher<Stmt> BooleanStmt = constructMatcher(
      "cxxBoolLiteral", constructMatcher("equals", VariantValue(true)))
      .getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("bool x = true;", BooleanStmt));
  EXPECT_FALSE(matches("bool x = false;", BooleanStmt));
  EXPECT_FALSE(matches("bool x = 0;", BooleanStmt));

  BooleanStmt = constructMatcher(
      "cxxBoolLiteral", constructMatcher("equals", VariantValue(0)))
      .getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("bool x = false;", BooleanStmt));
  EXPECT_FALSE(matches("bool x = true;", BooleanStmt));
  EXPECT_FALSE(matches("bool x = 0;", BooleanStmt));

  Matcher<Stmt> DoubleStmt = constructMatcher(
      "floatLiteral", constructMatcher("equals", VariantValue(1.2)))
      .getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("double x = 1.2;", DoubleStmt));
#if 0
  // FIXME floatLiteral matching should work regardless of suffix.
  EXPECT_TRUE(matches("double x = 1.2f;", DoubleStmt));
  EXPECT_TRUE(matches("double x = 1.2l;", DoubleStmt));
#endif
  EXPECT_TRUE(matches("double x = 12e-1;", DoubleStmt));
  EXPECT_FALSE(matches("double x = 1.23;", DoubleStmt));

  Matcher<Stmt> IntegerStmt = constructMatcher(
      "integerLiteral", constructMatcher("equals", VariantValue(42)))
      .getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("int x = 42;", IntegerStmt));
  EXPECT_FALSE(matches("int x = 1;", IntegerStmt));

  Matcher<Stmt> CharStmt = constructMatcher(
      "characterLiteral", constructMatcher("equals", VariantValue('x')))
      .getTypedMatcher<Stmt>();
  EXPECT_TRUE(matches("int x = 'x';", CharStmt));
  EXPECT_TRUE(matches("int x = L'x';", CharStmt));
  EXPECT_TRUE(matches("int x = u'x';", CharStmt));
  EXPECT_TRUE(matches("int x = U'x';", CharStmt));
  EXPECT_FALSE(matches("int x = 120;", CharStmt));
}

} // end anonymous namespace
} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
