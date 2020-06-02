#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Host.h"

#include "DeclMatcher.h"

#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

using std::get;

struct StructuralEquivalenceTest : ::testing::Test {
  std::unique_ptr<ASTUnit> AST0, AST1;
  std::string Code0, Code1; // Buffers for SourceManager

  // Get a pair of node pointers into the synthesized AST from the given code
  // snippets. To determine the returned node, a separate matcher is specified
  // for both snippets. The first matching node is returned.
  template <typename NodeType, typename MatcherType>
  std::tuple<NodeType *, NodeType *>
  makeDecls(const std::string &SrcCode0, const std::string &SrcCode1,
            TestLanguage Lang, const MatcherType &Matcher0,
            const MatcherType &Matcher1) {
    this->Code0 = SrcCode0;
    this->Code1 = SrcCode1;
    std::vector<std::string> Args = getCommandLineArgsForTesting(Lang);

    const char *const InputFileName = "input.cc";

    AST0 = tooling::buildASTFromCodeWithArgs(Code0, Args, InputFileName);
    AST1 = tooling::buildASTFromCodeWithArgs(Code1, Args, InputFileName);

    NodeType *D0 = FirstDeclMatcher<NodeType>().match(
        AST0->getASTContext().getTranslationUnitDecl(), Matcher0);
    NodeType *D1 = FirstDeclMatcher<NodeType>().match(
        AST1->getASTContext().getTranslationUnitDecl(), Matcher1);

    return std::make_tuple(D0, D1);
  }

  std::tuple<TranslationUnitDecl *, TranslationUnitDecl *>
  makeTuDecls(const std::string &SrcCode0, const std::string &SrcCode1,
              TestLanguage Lang) {
    this->Code0 = SrcCode0;
    this->Code1 = SrcCode1;
    std::vector<std::string> Args = getCommandLineArgsForTesting(Lang);

    const char *const InputFileName = "input.cc";

    AST0 = tooling::buildASTFromCodeWithArgs(Code0, Args, InputFileName);
    AST1 = tooling::buildASTFromCodeWithArgs(Code1, Args, InputFileName);

    return std::make_tuple(AST0->getASTContext().getTranslationUnitDecl(),
                           AST1->getASTContext().getTranslationUnitDecl());
  }

  // Get a pair of node pointers into the synthesized AST from the given code
  // snippets. The same matcher is used for both snippets.
  template <typename NodeType, typename MatcherType>
  std::tuple<NodeType *, NodeType *>
  makeDecls(const std::string &SrcCode0, const std::string &SrcCode1,
            TestLanguage Lang, const MatcherType &AMatcher) {
    return makeDecls<NodeType, MatcherType>(
          SrcCode0, SrcCode1, Lang, AMatcher, AMatcher);
  }

  // Get a pair of Decl pointers to the synthesized declarations from the given
  // code snippets. We search for the first NamedDecl with given name in both
  // snippets.
  std::tuple<NamedDecl *, NamedDecl *>
  makeNamedDecls(const std::string &SrcCode0, const std::string &SrcCode1,
                 TestLanguage Lang, const char *const Identifier = "foo") {
    auto Matcher = namedDecl(hasName(Identifier));
    return makeDecls<NamedDecl>(SrcCode0, SrcCode1, Lang, Matcher);
  }

  bool testStructuralMatch(Decl *D0, Decl *D1) {
    llvm::DenseSet<std::pair<Decl *, Decl *>> NonEquivalentDecls01;
    llvm::DenseSet<std::pair<Decl *, Decl *>> NonEquivalentDecls10;
    StructuralEquivalenceContext Ctx01(
        D0->getASTContext(), D1->getASTContext(),
        NonEquivalentDecls01, StructuralEquivalenceKind::Default, false, false);
    StructuralEquivalenceContext Ctx10(
        D1->getASTContext(), D0->getASTContext(),
        NonEquivalentDecls10, StructuralEquivalenceKind::Default, false, false);
    bool Eq01 = Ctx01.IsEquivalent(D0, D1);
    bool Eq10 = Ctx10.IsEquivalent(D1, D0);
    EXPECT_EQ(Eq01, Eq10);
    return Eq01;
  }

  bool testStructuralMatch(std::tuple<Decl *, Decl *> t) {
    return testStructuralMatch(get<0>(t), get<1>(t));
  }
};

TEST_F(StructuralEquivalenceTest, Int) {
  auto Decls = makeNamedDecls("int foo;", "int foo;", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, IntVsSignedInt) {
  auto Decls = makeNamedDecls("int foo;", "signed int foo;", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, Char) {
  auto Decls = makeNamedDecls("char foo;", "char foo;", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

// This test is disabled for now.
// FIXME Whether this is equivalent is dependendant on the target.
TEST_F(StructuralEquivalenceTest, DISABLED_CharVsSignedChar) {
  auto Decls = makeNamedDecls("char foo;", "signed char foo;", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, ForwardRecordDecl) {
  auto Decls = makeNamedDecls("struct foo;", "struct foo;", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, IntVsSignedIntInStruct) {
  auto Decls = makeNamedDecls("struct foo { int x; };",
                              "struct foo { signed int x; };", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, CharVsSignedCharInStruct) {
  auto Decls = makeNamedDecls("struct foo { char x; };",
                              "struct foo { signed char x; };", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, IntVsSignedIntTemplateSpec) {
  auto Decls = makeDecls<ClassTemplateSpecializationDecl>(
      R"(template <class T> struct foo; template<> struct foo<int>{};)",
      R"(template <class T> struct foo; template<> struct foo<signed int>{};)",
      Lang_CXX03, classTemplateSpecializationDecl());
  auto Spec0 = get<0>(Decls);
  auto Spec1 = get<1>(Decls);
  EXPECT_TRUE(testStructuralMatch(Spec0, Spec1));
}

TEST_F(StructuralEquivalenceTest, CharVsSignedCharTemplateSpec) {
  auto Decls = makeDecls<ClassTemplateSpecializationDecl>(
      R"(template <class T> struct foo; template<> struct foo<char>{};)",
      R"(template <class T> struct foo; template<> struct foo<signed char>{};)",
      Lang_CXX03, classTemplateSpecializationDecl());
  auto Spec0 = get<0>(Decls);
  auto Spec1 = get<1>(Decls);
  EXPECT_FALSE(testStructuralMatch(Spec0, Spec1));
}

TEST_F(StructuralEquivalenceTest, CharVsSignedCharTemplateSpecWithInheritance) {
  auto Decls = makeDecls<ClassTemplateSpecializationDecl>(
      R"(
      struct true_type{};
      template <class T> struct foo;
      template<> struct foo<char> : true_type {};
      )",
      R"(
      struct true_type{};
      template <class T> struct foo;
      template<> struct foo<signed char> : true_type {};
      )",
      Lang_CXX03, classTemplateSpecializationDecl());
  EXPECT_FALSE(testStructuralMatch(Decls));
}

// This test is disabled for now.
// FIXME Enable it, once the check is implemented.
TEST_F(StructuralEquivalenceTest, DISABLED_WrongOrderInNamespace) {
  auto Code =
      R"(
      namespace NS {
      template <class T> class Base {
          int a;
      };
      class Derived : Base<Derived> {
      };
      }
      void foo(NS::Derived &);
      )";
  auto Decls = makeNamedDecls(Code, Code, Lang_CXX03);

  NamespaceDecl *NS =
      LastDeclMatcher<NamespaceDecl>().match(get<1>(Decls), namespaceDecl());
  ClassTemplateDecl *TD = LastDeclMatcher<ClassTemplateDecl>().match(
      get<1>(Decls), classTemplateDecl(hasName("Base")));

  // Reorder the decls, move the TD to the last place in the DC.
  NS->removeDecl(TD);
  NS->addDeclInternal(TD);

  EXPECT_FALSE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, WrongOrderOfFieldsInClass) {
  auto Code = "class X { int a; int b; };";
  auto Decls = makeNamedDecls(Code, Code, Lang_CXX03, "X");

  CXXRecordDecl *RD = FirstDeclMatcher<CXXRecordDecl>().match(
      get<1>(Decls), cxxRecordDecl(hasName("X")));
  FieldDecl *FD =
      FirstDeclMatcher<FieldDecl>().match(get<1>(Decls), fieldDecl(hasName("a")));

  // Reorder the FieldDecls
  RD->removeDecl(FD);
  RD->addDeclInternal(FD);

  EXPECT_FALSE(testStructuralMatch(Decls));
}

struct StructuralEquivalenceFunctionTest : StructuralEquivalenceTest {
};

TEST_F(StructuralEquivalenceFunctionTest, TemplateVsNonTemplate) {
  auto t = makeNamedDecls("void foo();", "template<class T> void foo();",
                          Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, DifferentOperators) {
  auto t = makeDecls<FunctionDecl>(
      "struct X{}; bool operator<(X, X);", "struct X{}; bool operator==(X, X);",
      Lang_CXX03, functionDecl(hasOverloadedOperatorName("<")),
      functionDecl(hasOverloadedOperatorName("==")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, SameOperators) {
  auto t = makeDecls<FunctionDecl>(
      "struct X{}; bool operator<(X, X);", "struct X{}; bool operator<(X, X);",
      Lang_CXX03, functionDecl(hasOverloadedOperatorName("<")),
      functionDecl(hasOverloadedOperatorName("<")));
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, CtorVsDtor) {
  auto t = makeDecls<FunctionDecl>("struct X{ X(); };", "struct X{ ~X(); };",
                                   Lang_CXX03, cxxConstructorDecl(),
                                   cxxDestructorDecl());
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ParamConstWithRef) {
  auto t =
      makeNamedDecls("void foo(int&);", "void foo(const int&);", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ParamConstSimple) {
  auto t = makeNamedDecls("void foo(int);", "void foo(const int);", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(t));
  // consider this OK
}

TEST_F(StructuralEquivalenceFunctionTest, Throw) {
  auto t = makeNamedDecls("void foo();", "void foo() throw();", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, Noexcept) {
  auto t = makeNamedDecls("void foo();",
                          "void foo() noexcept;", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ThrowVsNoexcept) {
  auto t = makeNamedDecls("void foo() throw();",
                          "void foo() noexcept;", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ThrowVsNoexceptFalse) {
  auto t = makeNamedDecls("void foo() throw();",
                          "void foo() noexcept(false);", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ThrowVsNoexceptTrue) {
  auto t = makeNamedDecls("void foo() throw();",
                          "void foo() noexcept(true);", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, NoexceptNonMatch) {
  auto t = makeNamedDecls("void foo() noexcept(false);",
                          "void foo() noexcept(true);", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, NoexceptMatch) {
  auto t = makeNamedDecls("void foo() noexcept(false);",
                          "void foo() noexcept(false);", Lang_CXX11);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, NoexceptVsNoexceptFalse) {
  auto t = makeNamedDecls("void foo() noexcept;",
                          "void foo() noexcept(false);", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, NoexceptVsNoexceptTrue) {
  auto t = makeNamedDecls("void foo() noexcept;",
                          "void foo() noexcept(true);", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ReturnType) {
  auto t = makeNamedDecls("char foo();", "int foo();", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ReturnConst) {
  auto t = makeNamedDecls("char foo();", "const char foo();", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ReturnRef) {
  auto t = makeNamedDecls("char &foo();",
                          "char &&foo();", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ParamCount) {
  auto t = makeNamedDecls("void foo(int);", "void foo(int, int);", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ParamType) {
  auto t = makeNamedDecls("void foo(int);", "void foo(char);", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ParamName) {
  auto t = makeNamedDecls("void foo(int a);", "void foo(int b);", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, Variadic) {
  auto t =
      makeNamedDecls("void foo(int x...);", "void foo(int x);", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, ParamPtr) {
  auto t = makeNamedDecls("void foo(int *);", "void foo(int);", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, NameInParen) {
  auto t = makeNamedDecls("void ((foo))();", "void foo();", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, NameInParenWithExceptionSpec) {
  auto t = makeNamedDecls(
      "void (foo)() throw(int);",
      "void (foo)() noexcept;",
      Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, NameInParenWithConst) {
  auto t = makeNamedDecls(
      "struct A { void (foo)() const; };",
      "struct A { void (foo)(); };",
      Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, FunctionsWithDifferentNoreturnAttr) {
  auto t = makeNamedDecls("__attribute__((noreturn)) void foo();",
                          "                          void foo();", Lang_C99);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest,
    FunctionsWithDifferentCallingConventions) {
  // These attributes may not be available on certain platforms.
  if (llvm::Triple(llvm::sys::getDefaultTargetTriple()).getArch() !=
      llvm::Triple::x86_64)
    return;
  auto t = makeNamedDecls("__attribute__((preserve_all)) void foo();",
                          "__attribute__((ms_abi))   void foo();", Lang_C99);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceFunctionTest, FunctionsWithDifferentSavedRegsAttr) {
  if (llvm::Triple(llvm::sys::getDefaultTargetTriple()).getArch() !=
      llvm::Triple::x86_64)
    return;
  auto t = makeNamedDecls(
      "__attribute__((no_caller_saved_registers)) void foo();",
      "                                           void foo();", Lang_C99);
  EXPECT_FALSE(testStructuralMatch(t));
}

struct StructuralEquivalenceCXXMethodTest : StructuralEquivalenceTest {
};

TEST_F(StructuralEquivalenceCXXMethodTest, Virtual) {
  auto t = makeDecls<CXXMethodDecl>("struct X { void foo(); };",
                                    "struct X { virtual void foo(); };",
                                    Lang_CXX03, cxxMethodDecl(hasName("foo")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Pure) {
  auto t = makeNamedDecls("struct X { virtual void foo(); };",
                          "struct X { virtual void foo() = 0; };", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, DISABLED_Final) {
  // The final-ness is not checked yet.
  auto t =
      makeNamedDecls("struct X { virtual void foo(); };",
                     "struct X { virtual void foo() final; };", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Const) {
  auto t = makeNamedDecls("struct X { void foo(); };",
                          "struct X { void foo() const; };", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Static) {
  auto t = makeNamedDecls("struct X { void foo(); };",
                          "struct X { static void foo(); };", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Ref1) {
  auto t = makeNamedDecls("struct X { void foo(); };",
                          "struct X { void foo() &&; };", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Ref2) {
  auto t = makeNamedDecls("struct X { void foo() &; };",
                          "struct X { void foo() &&; };", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, AccessSpecifier) {
  auto t = makeDecls<CXXMethodDecl>("struct X { public: void foo(); };",
                                    "struct X { private: void foo(); };",
                                    Lang_CXX03, cxxMethodDecl(hasName("foo")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Delete) {
  auto t = makeNamedDecls("struct X { void foo(); };",
                          "struct X { void foo() = delete; };", Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Constructor) {
  auto t = makeDecls<FunctionDecl>("void foo();", "struct foo { foo(); };",
                                   Lang_CXX03, functionDecl(hasName("foo")),
                                   cxxConstructorDecl(hasName("foo")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, ConstructorParam) {
  auto t = makeDecls<CXXConstructorDecl>("struct X { X(); };",
                                         "struct X { X(int); };", Lang_CXX03,
                                         cxxConstructorDecl(hasName("X")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, ConstructorExplicit) {
  auto t = makeDecls<CXXConstructorDecl>("struct X { X(int); };",
                                         "struct X { explicit X(int); };",
                                         Lang_CXX11,
                                         cxxConstructorDecl(hasName("X")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, ConstructorDefault) {
  auto t = makeDecls<CXXConstructorDecl>("struct X { X(); };",
                                         "struct X { X() = default; };",
                                         Lang_CXX11,
                                         cxxConstructorDecl(hasName("X")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Conversion) {
  auto t = makeDecls<CXXConversionDecl>("struct X { operator bool(); };",
                                        "struct X { operator char(); };",
                                         Lang_CXX11,
                                         cxxConversionDecl());
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, Operator) {
  auto t =
      makeDecls<FunctionDecl>("struct X { int operator +(int); };",
                              "struct X { int operator -(int); };", Lang_CXX03,
                              functionDecl(hasOverloadedOperatorName("+")),
                              functionDecl(hasOverloadedOperatorName("-")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, OutOfClass1) {
  auto t = makeDecls<FunctionDecl>(
      "struct X { virtual void f(); }; void X::f() { }",
      "struct X { virtual void f() { }; };", Lang_CXX03,
      functionDecl(allOf(hasName("f"), isDefinition())));
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceCXXMethodTest, OutOfClass2) {
  auto t = makeDecls<FunctionDecl>(
      "struct X { virtual void f(); }; void X::f() { }",
      "struct X { void f(); }; void X::f() { }", Lang_CXX03,
      functionDecl(allOf(hasName("f"), isDefinition())));
  EXPECT_FALSE(testStructuralMatch(t));
}

struct StructuralEquivalenceRecordTest : StructuralEquivalenceTest {
  // FIXME Use a common getRecordDecl with ASTImporterTest.cpp!
  RecordDecl *getRecordDecl(FieldDecl *FD) {
    auto *ET = cast<ElaboratedType>(FD->getType().getTypePtr());
    return cast<RecordType>(ET->getNamedType().getTypePtr())->getDecl();
  };
};

TEST_F(StructuralEquivalenceRecordTest, Name) {
  auto t = makeDecls<CXXRecordDecl>("struct A{ };", "struct B{ };", Lang_CXX03,
                                    cxxRecordDecl(hasName("A")),
                                    cxxRecordDecl(hasName("B")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest, Fields) {
  auto t = makeNamedDecls("struct foo{ int x; };", "struct foo{ char x; };",
                          Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest, DISABLED_Methods) {
  // Currently, methods of a class are not checked at class equivalence.
  auto t = makeNamedDecls("struct foo{ int x(); };", "struct foo{ char x(); };",
                          Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest, Bases) {
  auto t = makeNamedDecls("struct A{ }; struct foo: A { };",
                          "struct B{ }; struct foo: B { };", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest, InheritanceVirtual) {
  auto t =
      makeNamedDecls("struct A{ }; struct foo: A { };",
                     "struct A{ }; struct foo: virtual A { };", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest, DISABLED_InheritanceType) {
  // Access specifier in inheritance is not checked yet.
  auto t =
      makeNamedDecls("struct A{ }; struct foo: public A { };",
                     "struct A{ }; struct foo: private A { };", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest, Match) {
  auto Code = R"(
      struct A{ };
      struct B{ };
      struct foo: A, virtual B {
        void x();
        int a;
      };
      )";
  auto t = makeNamedDecls(Code, Code, Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest, UnnamedRecordsShouldBeInequivalent) {
  auto t = makeTuDecls(
      R"(
      struct A {
        struct {
          struct A *next;
        } entry0;
        struct {
          struct A *next;
        } entry1;
      };
      )",
      "", Lang_C99);
  auto *TU = get<0>(t);
  auto *Entry0 =
      FirstDeclMatcher<FieldDecl>().match(TU, fieldDecl(hasName("entry0")));
  auto *Entry1 =
      FirstDeclMatcher<FieldDecl>().match(TU, fieldDecl(hasName("entry1")));
  auto *R0 = getRecordDecl(Entry0);
  auto *R1 = getRecordDecl(Entry1);

  ASSERT_NE(R0, R1);
  EXPECT_TRUE(testStructuralMatch(R0, R0));
  EXPECT_TRUE(testStructuralMatch(R1, R1));
  EXPECT_FALSE(testStructuralMatch(R0, R1));
}

TEST_F(StructuralEquivalenceRecordTest, AnonymousRecordsShouldBeInequivalent) {
  auto t = makeTuDecls(
      R"(
      struct X {
        struct {
          int a;
        };
        struct {
          int b;
        };
      };
      )",
      "", Lang_C99);
  auto *TU = get<0>(t);
  auto *A = FirstDeclMatcher<IndirectFieldDecl>().match(
      TU, indirectFieldDecl(hasName("a")));
  auto *FA = cast<FieldDecl>(A->chain().front());
  RecordDecl *RA = cast<RecordType>(FA->getType().getTypePtr())->getDecl();
  auto *B = FirstDeclMatcher<IndirectFieldDecl>().match(
      TU, indirectFieldDecl(hasName("b")));
  auto *FB = cast<FieldDecl>(B->chain().front());
  RecordDecl *RB = cast<RecordType>(FB->getType().getTypePtr())->getDecl();

  ASSERT_NE(RA, RB);
  EXPECT_TRUE(testStructuralMatch(RA, RA));
  EXPECT_TRUE(testStructuralMatch(RB, RB));
  EXPECT_FALSE(testStructuralMatch(RA, RB));
}

TEST_F(StructuralEquivalenceRecordTest,
       RecordsAreInequivalentIfOrderOfAnonRecordsIsDifferent) {
  auto t = makeTuDecls(
      R"(
      struct X {
        struct { int a; };
        struct { int b; };
      };
      )",
      R"(
      struct X { // The order is reversed.
        struct { int b; };
        struct { int a; };
      };
      )",
      Lang_C99);

  auto *TU = get<0>(t);
  auto *A = FirstDeclMatcher<IndirectFieldDecl>().match(
      TU, indirectFieldDecl(hasName("a")));
  auto *FA = cast<FieldDecl>(A->chain().front());
  RecordDecl *RA = cast<RecordType>(FA->getType().getTypePtr())->getDecl();

  auto *TU1 = get<1>(t);
  auto *A1 = FirstDeclMatcher<IndirectFieldDecl>().match(
      TU1, indirectFieldDecl(hasName("a")));
  auto *FA1 = cast<FieldDecl>(A1->chain().front());
  RecordDecl *RA1 = cast<RecordType>(FA1->getType().getTypePtr())->getDecl();

  RecordDecl *X =
      FirstDeclMatcher<RecordDecl>().match(TU, recordDecl(hasName("X")));
  RecordDecl *X1 =
      FirstDeclMatcher<RecordDecl>().match(TU1, recordDecl(hasName("X")));
  ASSERT_NE(X, X1);
  EXPECT_FALSE(testStructuralMatch(X, X1));

  ASSERT_NE(RA, RA1);
  EXPECT_TRUE(testStructuralMatch(RA, RA));
  EXPECT_TRUE(testStructuralMatch(RA1, RA1));
  EXPECT_FALSE(testStructuralMatch(RA1, RA));
}

TEST_F(StructuralEquivalenceRecordTest,
       UnnamedRecordsShouldBeInequivalentEvenIfTheSecondIsBeingDefined) {
  auto Code =
      R"(
      struct A {
        struct {
          struct A *next;
        } entry0;
        struct {
          struct A *next;
        } entry1;
      };
      )";
  auto t = makeTuDecls(Code, Code, Lang_C99);

  auto *FromTU = get<0>(t);
  auto *Entry1 =
      FirstDeclMatcher<FieldDecl>().match(FromTU, fieldDecl(hasName("entry1")));

  auto *ToTU = get<1>(t);
  auto *Entry0 =
      FirstDeclMatcher<FieldDecl>().match(ToTU, fieldDecl(hasName("entry0")));
  auto *A =
      FirstDeclMatcher<RecordDecl>().match(ToTU, recordDecl(hasName("A")));
  A->startDefinition(); // Set isBeingDefined, getDefinition() will return a
                        // nullptr. This may be the case during ASTImport.

  auto *R0 = getRecordDecl(Entry0);
  auto *R1 = getRecordDecl(Entry1);

  ASSERT_NE(R0, R1);
  EXPECT_TRUE(testStructuralMatch(R0, R0));
  EXPECT_TRUE(testStructuralMatch(R1, R1));
  EXPECT_FALSE(testStructuralMatch(R0, R1));
}

TEST_F(StructuralEquivalenceRecordTest, TemplateVsNonTemplate) {
  auto t = makeDecls<CXXRecordDecl>("struct A { };",
                                    "template<class T> struct A { };",
                                    Lang_CXX03, cxxRecordDecl(hasName("A")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest,
    FwdDeclRecordShouldBeEqualWithFwdDeclRecord) {
  auto t = makeNamedDecls("class foo;", "class foo;", Lang_CXX11);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest,
       FwdDeclRecordShouldBeEqualWithRecordWhichHasDefinition) {
  auto t =
      makeNamedDecls("class foo;", "class foo { int A; };", Lang_CXX11);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest,
       RecordShouldBeEqualWithRecordWhichHasDefinition) {
  auto t = makeNamedDecls("class foo { int A; };", "class foo { int A; };",
                          Lang_CXX11);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceRecordTest, RecordsWithDifferentBody) {
  auto t = makeNamedDecls("class foo { int B; };", "class foo { int A; };",
                          Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

struct StructuralEquivalenceLambdaTest : StructuralEquivalenceTest {};

TEST_F(StructuralEquivalenceLambdaTest, LambdaClassesWithDifferentMethods) {
  // Get the LambdaExprs, unfortunately we can't match directly the underlying
  // implicit CXXRecordDecl of the Lambda classes.
  auto t = makeDecls<LambdaExpr>(
      "void f() { auto L0 = [](int){}; }",
      "void f() { auto L1 = [](){}; }",
      Lang_CXX11,
      lambdaExpr(),
      lambdaExpr());
  CXXRecordDecl *L0 = get<0>(t)->getLambdaClass();
  CXXRecordDecl *L1 = get<1>(t)->getLambdaClass();
  EXPECT_FALSE(testStructuralMatch(L0, L1));
}

TEST_F(StructuralEquivalenceLambdaTest, LambdaClassesWithEqMethods) {
  auto t = makeDecls<LambdaExpr>(
      "void f() { auto L0 = [](int){}; }",
      "void f() { auto L1 = [](int){}; }",
      Lang_CXX11,
      lambdaExpr(),
      lambdaExpr());
  CXXRecordDecl *L0 = get<0>(t)->getLambdaClass();
  CXXRecordDecl *L1 = get<1>(t)->getLambdaClass();
  EXPECT_TRUE(testStructuralMatch(L0, L1));
}

TEST_F(StructuralEquivalenceLambdaTest, LambdaClassesWithDifferentFields) {
  auto t = makeDecls<LambdaExpr>(
      "void f() { char* X; auto L0 = [X](){}; }",
      "void f() { float X; auto L1 = [X](){}; }",
      Lang_CXX11,
      lambdaExpr(),
      lambdaExpr());
  CXXRecordDecl *L0 = get<0>(t)->getLambdaClass();
  CXXRecordDecl *L1 = get<1>(t)->getLambdaClass();
  EXPECT_FALSE(testStructuralMatch(L0, L1));
}

TEST_F(StructuralEquivalenceLambdaTest, LambdaClassesWithEqFields) {
  auto t = makeDecls<LambdaExpr>(
      "void f() { float X; auto L0 = [X](){}; }",
      "void f() { float X; auto L1 = [X](){}; }",
      Lang_CXX11,
      lambdaExpr(),
      lambdaExpr());
  CXXRecordDecl *L0 = get<0>(t)->getLambdaClass();
  CXXRecordDecl *L1 = get<1>(t)->getLambdaClass();
  EXPECT_TRUE(testStructuralMatch(L0, L1));
}

TEST_F(StructuralEquivalenceTest, CompareSameDeclWithMultiple) {
  auto t = makeNamedDecls("struct A{ }; struct B{ }; void foo(A a, A b);",
                          "struct A{ }; struct B{ }; void foo(A a, B b);",
                          Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceTest, ExplicitBoolDifferent) {
  auto Decls = makeNamedDecls("struct foo {explicit(false) foo(int);};",
                              "struct foo {explicit(true) foo(int);};", Lang_CXX20);
  CXXConstructorDecl *First = FirstDeclMatcher<CXXConstructorDecl>().match(
      get<0>(Decls), cxxConstructorDecl(hasName("foo")));
  CXXConstructorDecl *Second = FirstDeclMatcher<CXXConstructorDecl>().match(
      get<1>(Decls), cxxConstructorDecl(hasName("foo")));
  EXPECT_FALSE(testStructuralMatch(First, Second));
}

TEST_F(StructuralEquivalenceTest, ExplicitBoolSame) {
  auto Decls = makeNamedDecls("struct foo {explicit(true) foo(int);};",
                              "struct foo {explicit(true) foo(int);};", Lang_CXX20);
  CXXConstructorDecl *First = FirstDeclMatcher<CXXConstructorDecl>().match(
      get<0>(Decls), cxxConstructorDecl(hasName("foo")));
  CXXConstructorDecl *Second = FirstDeclMatcher<CXXConstructorDecl>().match(
      get<1>(Decls), cxxConstructorDecl(hasName("foo")));
  EXPECT_TRUE(testStructuralMatch(First, Second));
}

struct StructuralEquivalenceEnumTest : StructuralEquivalenceTest {};

TEST_F(StructuralEquivalenceEnumTest, FwdDeclEnumShouldBeEqualWithFwdDeclEnum) {
  auto t = makeNamedDecls("enum class foo;", "enum class foo;", Lang_CXX11);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceEnumTest,
       FwdDeclEnumShouldBeEqualWithEnumWhichHasDefinition) {
  auto t =
      makeNamedDecls("enum class foo;", "enum class foo { A };", Lang_CXX11);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceEnumTest,
       EnumShouldBeEqualWithEnumWhichHasDefinition) {
  auto t = makeNamedDecls("enum class foo { A };", "enum class foo { A };",
                          Lang_CXX11);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceEnumTest, EnumsWithDifferentBody) {
  auto t = makeNamedDecls("enum class foo { B };", "enum class foo { A };",
                          Lang_CXX11);
  EXPECT_FALSE(testStructuralMatch(t));
}

struct StructuralEquivalenceTemplateTest : StructuralEquivalenceTest {};

TEST_F(StructuralEquivalenceTemplateTest, ExactlySameTemplates) {
  auto t = makeNamedDecls("template <class T> struct foo;",
                          "template <class T> struct foo;", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceTemplateTest, DifferentTemplateArgName) {
  auto t = makeNamedDecls("template <class T> struct foo;",
                          "template <class U> struct foo;", Lang_CXX03);
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceTemplateTest, DifferentTemplateArgKind) {
  auto t = makeNamedDecls("template <class T> struct foo;",
                          "template <int T> struct foo;", Lang_CXX03);
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceTemplateTest, ExplicitBoolSame) {
  auto Decls = makeNamedDecls(
      "template <bool b> struct foo {explicit(b) foo(int);};",
      "template <bool b> struct foo {explicit(b) foo(int);};", Lang_CXX20);
  CXXConstructorDecl *First = FirstDeclMatcher<CXXConstructorDecl>().match(
      get<0>(Decls), cxxConstructorDecl(hasName("foo<b>")));
  CXXConstructorDecl *Second = FirstDeclMatcher<CXXConstructorDecl>().match(
      get<1>(Decls), cxxConstructorDecl(hasName("foo<b>")));
  EXPECT_TRUE(testStructuralMatch(First, Second));
}

TEST_F(StructuralEquivalenceTemplateTest, ExplicitBoolDifference) {
  auto Decls = makeNamedDecls(
      "template <bool b> struct foo {explicit(b) foo(int);};",
      "template <bool b> struct foo {explicit(!b) foo(int);};", Lang_CXX20);
  CXXConstructorDecl *First = FirstDeclMatcher<CXXConstructorDecl>().match(
      get<0>(Decls), cxxConstructorDecl(hasName("foo<b>")));
  CXXConstructorDecl *Second = FirstDeclMatcher<CXXConstructorDecl>().match(
      get<1>(Decls), cxxConstructorDecl(hasName("foo<b>")));
  EXPECT_FALSE(testStructuralMatch(First, Second));
}

TEST_F(StructuralEquivalenceTemplateTest,
       TemplateVsSubstTemplateTemplateParmInArgEq) {
  auto t = makeDecls<ClassTemplateSpecializationDecl>(
      R"(
template <typename P1> class Arg { };
template <template <typename PP1> class P1> class Primary { };

void f() {
  // Make specialization with simple template.
  Primary <Arg> A;
}
      )",
      R"(
template <typename P1> class Arg { };
template <template <typename PP1> class P1> class Primary { };

template <template <typename PP1> class P1> class Templ {
  void f() {
    // Make specialization with substituted template template param.
    Primary <P1> A;
  };
};

// Instantiate with substitution Arg into P1.
template class Templ <Arg>;
      )",
      Lang_CXX03, classTemplateSpecializationDecl(hasName("Primary")));
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceTemplateTest,
       TemplateVsSubstTemplateTemplateParmInArgNotEq) {
  auto t = makeDecls<ClassTemplateSpecializationDecl>(
      R"(
template <typename P1> class Arg { };
template <template <typename PP1> class P1> class Primary { };

void f() {
  // Make specialization with simple template.
  Primary <Arg> A;
}
      )",
      R"(
// Arg is different from the other, this should cause non-equivalence.
template <typename P1> class Arg { int X; };
template <template <typename PP1> class P1> class Primary { };

template <template <typename PP1> class P1> class Templ {
  void f() {
    // Make specialization with substituted template template param.
    Primary <P1> A;
  };
};

// Instantiate with substitution Arg into P1.
template class Templ <Arg>;
      )",
      Lang_CXX03, classTemplateSpecializationDecl(hasName("Primary")));
  EXPECT_FALSE(testStructuralMatch(t));
}

struct StructuralEquivalenceDependentTemplateArgsTest
    : StructuralEquivalenceTemplateTest {};

TEST_F(StructuralEquivalenceDependentTemplateArgsTest,
       SameStructsInDependentArgs) {
  std::string Code =
      R"(
      template <typename>
      struct S1;

      template <typename>
      struct enable_if;

      struct S
      {
        template <typename T, typename enable_if<S1<T>>::type>
        void f();
      };
      )";
  auto t = makeDecls<FunctionTemplateDecl>(Code, Code, Lang_CXX11,
                                           functionTemplateDecl(hasName("f")));
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceDependentTemplateArgsTest,
       DifferentStructsInDependentArgs) {
  std::string Code =
      R"(
      template <typename>
      struct S1;

      template <typename>
      struct S2;

      template <typename>
      struct enable_if;
      )";
  auto t = makeDecls<FunctionTemplateDecl>(Code + R"(
      struct S
      {
        template <typename T, typename enable_if<S1<T>>::type>
        void f();
      };
      )",
                                           Code + R"(
      struct S
      {
        template <typename T, typename enable_if<S2<T>>::type>
        void f();
      };
      )",
                                           Lang_CXX11,
                                           functionTemplateDecl(hasName("f")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceDependentTemplateArgsTest,
       SameStructsInDependentScopeDeclRefExpr) {
  std::string Code =
      R"(
      template <typename>
      struct S1;

      template <bool>
      struct enable_if;

      struct S
      {
        template <typename T, typename enable_if<S1<T>::value>::type>
        void f();   // DependentScopeDeclRefExpr:^^^^^^^^^^^^
      };
      )";
  auto t = makeDecls<FunctionTemplateDecl>(Code, Code, Lang_CXX11,
                                           functionTemplateDecl(hasName("f")));
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceDependentTemplateArgsTest,
       DifferentStructsInDependentScopeDeclRefExpr) {
  std::string Code =
      R"(
      template <typename>
      struct S1;

      template <typename>
      struct S2;

      template <bool>
      struct enable_if;
      )";
  auto t = makeDecls<FunctionTemplateDecl>(Code + R"(
      struct S
      {
        template <typename T, typename enable_if<S1<T>::value>::type>
        void f();   // DependentScopeDeclRefExpr:^^^^^^^^^^^^
      };
      )",
                                           Code + R"(
      struct S
      {
        template <typename T, typename enable_if<S2<T>::value>::type>
        void f();
      };
      )",
                                           Lang_CXX03,
                                           functionTemplateDecl(hasName("f")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(StructuralEquivalenceDependentTemplateArgsTest,
       DifferentValueInDependentScopeDeclRefExpr) {
  std::string Code =
      R"(
      template <typename>
      struct S1;

      template <bool>
      struct enable_if;
      )";
  auto t = makeDecls<FunctionTemplateDecl>(Code + R"(
      struct S
      {
        template <typename T, typename enable_if<S1<T>::value1>::type>
        void f();   // DependentScopeDeclRefExpr:^^^^^^^^^^^^
      };
      )",
                                           Code + R"(
      struct S
      {
        template <typename T, typename enable_if<S1<T>::value2>::type>
        void f();
      };
      )",
                                           Lang_CXX03,
                                           functionTemplateDecl(hasName("f")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(
    StructuralEquivalenceTemplateTest,
    ClassTemplSpecWithQualifiedAndNonQualifiedTypeArgsShouldBeEqual) {
  auto t = makeDecls<ClassTemplateSpecializationDecl>(
      R"(
      template <class T> struct Primary {};
      namespace N {
        struct Arg;
      }
      // Explicit instantiation with qualified name.
      template struct Primary<N::Arg>;
      )",
      R"(
      template <class T> struct Primary {};
      namespace N {
        struct Arg;
      }
      using namespace N;
      // Explicit instantiation with UNqualified name.
      template struct Primary<Arg>;
      )",
      Lang_CXX03, classTemplateSpecializationDecl(hasName("Primary")));
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(
    StructuralEquivalenceTemplateTest,
    ClassTemplSpecWithInequivalentQualifiedAndNonQualifiedTypeArgs) {
  auto t = makeDecls<ClassTemplateSpecializationDecl>(
      R"(
      template <class T> struct Primary {};
      namespace N {
        struct Arg { int a; };
      }
      // Explicit instantiation with qualified name.
      template struct Primary<N::Arg>;
      )",
      R"(
      template <class T> struct Primary {};
      namespace N {
        // This struct is not equivalent with the other in the prev TU.
        struct Arg { double b; }; // -- Field mismatch.
      }
      using namespace N;
      // Explicit instantiation with UNqualified name.
      template struct Primary<Arg>;
      )",
      Lang_CXX03, classTemplateSpecializationDecl(hasName("Primary")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(
    StructuralEquivalenceTemplateTest,
    ClassTemplSpecWithQualifiedAndNonQualifiedTemplArgsShouldBeEqual) {
  auto t = makeDecls<ClassTemplateSpecializationDecl>(
      R"(
      template <template <class> class T> struct Primary {};
      namespace N {
        template <class T> struct Arg;
      }
      // Explicit instantiation with qualified name.
      template struct Primary<N::Arg>;
      )",
      R"(
      template <template <class> class T> struct Primary {};
      namespace N {
        template <class T> struct Arg;
      }
      using namespace N;
      // Explicit instantiation with UNqualified name.
      template struct Primary<Arg>;
      )",
      Lang_CXX03, classTemplateSpecializationDecl(hasName("Primary")));
  EXPECT_TRUE(testStructuralMatch(t));
}

TEST_F(
    StructuralEquivalenceTemplateTest,
    ClassTemplSpecWithInequivalentQualifiedAndNonQualifiedTemplArgs) {
  auto t = makeDecls<ClassTemplateSpecializationDecl>(
      R"(
      template <template <class> class T> struct Primary {};
      namespace N {
        template <class T> struct Arg { int a; };
      }
      // Explicit instantiation with qualified name.
      template struct Primary<N::Arg>;
      )",
      R"(
      template <template <class> class T> struct Primary {};
      namespace N {
        // This template is not equivalent with the other in the prev TU.
        template <class T> struct Arg { double b; }; // -- Field mismatch.
      }
      using namespace N;
      // Explicit instantiation with UNqualified name.
      template struct Primary<Arg>;
      )",
      Lang_CXX03, classTemplateSpecializationDecl(hasName("Primary")));
  EXPECT_FALSE(testStructuralMatch(t));
}

TEST_F(
    StructuralEquivalenceTemplateTest,
    ClassTemplSpecWithInequivalentShadowedTemplArg) {
  auto t = makeDecls<ClassTemplateSpecializationDecl>(
      R"(
      template <template <class> class T> struct Primary {};
      template <class T> struct Arg { int a; };
      // Explicit instantiation with ::Arg
      template struct Primary<Arg>;
      )",
      R"(
      template <template <class> class T> struct Primary {};
      template <class T> struct Arg { int a; };
      namespace N {
        // This template is not equivalent with the other in the global scope.
        template <class T> struct Arg { double b; }; // -- Field mismatch.
        // Explicit instantiation with N::Arg which shadows ::Arg
        template struct Primary<Arg>;
      }
      )",
      Lang_CXX03, classTemplateSpecializationDecl(hasName("Primary")));
  EXPECT_FALSE(testStructuralMatch(t));
}
struct StructuralEquivalenceCacheTest : public StructuralEquivalenceTest {
  llvm::DenseSet<std::pair<Decl *, Decl *>> NonEquivalentDecls;

  template <typename NodeType, typename MatcherType>
  std::pair<NodeType *, NodeType *>
  findDeclPair(std::tuple<TranslationUnitDecl *, TranslationUnitDecl *> TU,
               MatcherType M) {
    NodeType *D0 = FirstDeclMatcher<NodeType>().match(get<0>(TU), M);
    NodeType *D1 = FirstDeclMatcher<NodeType>().match(get<1>(TU), M);
    return {D0, D1};
  }

  template <typename NodeType>
  bool isInNonEqCache(std::pair<NodeType *, NodeType *> D) {
    return NonEquivalentDecls.count(D) > 0;
  }
};

TEST_F(StructuralEquivalenceCacheTest, SimpleNonEq) {
  auto TU = makeTuDecls(
      R"(
      class A {};
      class B {};
      void x(A, A);
      )",
      R"(
      class A {};
      class B {};
      void x(A, B);
      )",
      Lang_CXX03);

  StructuralEquivalenceContext Ctx(
      get<0>(TU)->getASTContext(), get<1>(TU)->getASTContext(),
      NonEquivalentDecls, StructuralEquivalenceKind::Default, false, false);

  auto X = findDeclPair<FunctionDecl>(TU, functionDecl(hasName("x")));
  EXPECT_FALSE(Ctx.IsEquivalent(X.first, X.second));

  EXPECT_FALSE(isInNonEqCache(findDeclPair<CXXRecordDecl>(
      TU, cxxRecordDecl(hasName("A"), unless(isImplicit())))));
  EXPECT_FALSE(isInNonEqCache(findDeclPair<CXXRecordDecl>(
      TU, cxxRecordDecl(hasName("B"), unless(isImplicit())))));
}

TEST_F(StructuralEquivalenceCacheTest, SpecialNonEq) {
  auto TU = makeTuDecls(
      R"(
      class A {};
      class B { int i; };
      void x(A *);
      void y(A *);
      class C {
        friend void x(A *);
        friend void y(A *);
      };
      )",
      R"(
      class A {};
      class B { int i; };
      void x(A *);
      void y(B *);
      class C {
        friend void x(A *);
        friend void y(B *);
      };
      )",
      Lang_CXX03);

  StructuralEquivalenceContext Ctx(
      get<0>(TU)->getASTContext(), get<1>(TU)->getASTContext(),
      NonEquivalentDecls, StructuralEquivalenceKind::Default, false, false);

  auto C = findDeclPair<CXXRecordDecl>(
      TU, cxxRecordDecl(hasName("C"), unless(isImplicit())));
  EXPECT_FALSE(Ctx.IsEquivalent(C.first, C.second));

  EXPECT_FALSE(isInNonEqCache(C));
  EXPECT_FALSE(isInNonEqCache(findDeclPair<CXXRecordDecl>(
      TU, cxxRecordDecl(hasName("A"), unless(isImplicit())))));
  EXPECT_FALSE(isInNonEqCache(findDeclPair<CXXRecordDecl>(
      TU, cxxRecordDecl(hasName("B"), unless(isImplicit())))));
  EXPECT_FALSE(isInNonEqCache(
      findDeclPair<FunctionDecl>(TU, functionDecl(hasName("x")))));
  EXPECT_FALSE(isInNonEqCache(
      findDeclPair<FunctionDecl>(TU, functionDecl(hasName("y")))));
}

TEST_F(StructuralEquivalenceCacheTest, Cycle) {
  auto TU = makeTuDecls(
      R"(
      class C;
      class A { C *c; };
      void x(A *);
      class C {
        friend void x(A *);
      };
      )",
      R"(
      class C;
      class A { C *c; };
      void x(A *);
      class C {
        friend void x(A *);
      };
      )",
      Lang_CXX03);

  StructuralEquivalenceContext Ctx(
      get<0>(TU)->getASTContext(), get<1>(TU)->getASTContext(),
      NonEquivalentDecls, StructuralEquivalenceKind::Default, false, false);

  auto C = findDeclPair<CXXRecordDecl>(
      TU, cxxRecordDecl(hasName("C"), unless(isImplicit())));
  EXPECT_TRUE(Ctx.IsEquivalent(C.first, C.second));

  EXPECT_FALSE(isInNonEqCache(C));
  EXPECT_FALSE(isInNonEqCache(findDeclPair<CXXRecordDecl>(
      TU, cxxRecordDecl(hasName("A"), unless(isImplicit())))));
  EXPECT_FALSE(isInNonEqCache(
      findDeclPair<FunctionDecl>(TU, functionDecl(hasName("x")))));
}

} // end namespace ast_matchers
} // end namespace clang
