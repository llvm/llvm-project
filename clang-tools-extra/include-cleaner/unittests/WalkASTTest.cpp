//===--- WalkASTTest.cpp ------------------------------------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AnalysisInternal.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Testing/TestAST.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clang::include_cleaner {
namespace {
using testing::ElementsAre;

// Specifies a test of which symbols are referenced by a piece of code.
// Target should contain points annotated with the reference kind.
// Example:
//   Target:      int $explicit^foo();
//   Referencing: int x = ^foo();
// There must be exactly one referencing location marked.
// Returns target decls.
std::vector<Decl::Kind>
testWalk(llvm::StringRef TargetCode, llvm::StringRef ReferencingCode,
         std::vector<std::string> ExtraArgs = {"-std=c++20"}) {
  llvm::Annotations Target(TargetCode);
  llvm::Annotations Referencing(ReferencingCode);

  TestInputs Inputs(Referencing.code());
  Inputs.ExtraFiles["target.h"] = Target.code().str();
  Inputs.ExtraArgs.push_back("-include");
  Inputs.ExtraArgs.push_back("target.h");
  for (const auto &Arg : ExtraArgs)
    Inputs.ExtraArgs.push_back(Arg);
  TestAST AST(Inputs);
  const auto &SM = AST.sourceManager();

  // We're only going to record references from the nominated point,
  // to the target file.
  FileID ReferencingFile = SM.getMainFileID();
  SourceLocation ReferencingLoc =
      SM.getComposedLoc(ReferencingFile, Referencing.point());
  FileID TargetFile = SM.translateFile(
      llvm::cantFail(AST.fileManager().getFileRef("target.h")));

  std::vector<Decl::Kind> TargetDecls;
  // Perform the walk, and capture the offsets of the referenced targets.
  std::unordered_map<RefType, std::vector<size_t>> ReferencedOffsets;
  llvm::SmallVector<Decl *> TopLevelDecls;
  for (Decl *D : AST.context().getTranslationUnitDecl()->decls()) {
    if (ReferencingFile == SM.getDecomposedExpansionLoc(D->getLocation()).first)
      TopLevelDecls.push_back(D);
  }
  walkAST(TopLevelDecls, [&](SourceLocation Loc, NamedDecl &ND, RefType RT) {
    if (SM.getFileLoc(Loc) != ReferencingLoc)
      return;
    auto NDLoc = SM.getDecomposedLoc(SM.getFileLoc(ND.getLocation()));
    if (NDLoc.first != TargetFile)
      return;
    ReferencedOffsets[RT].push_back(NDLoc.second);
    TargetDecls.push_back(ND.getKind());
  });
  for (auto &Entry : ReferencedOffsets)
    llvm::sort(Entry.second);

  // Compare results to the expected points.
  // For each difference, show the target point in context, like a diagnostic.
  std::string DiagBuf;
  llvm::raw_string_ostream DiagOS(DiagBuf);
  DiagnosticOptions DiagOpts;
  DiagOpts.ShowLevel = 0;
  DiagOpts.ShowNoteIncludeStack = 0;
  TextDiagnostic Diag(DiagOS, AST.context().getLangOpts(), DiagOpts);
  auto DiagnosePoint = [&](llvm::StringRef Message, unsigned Offset) {
    Diag.emitDiagnostic(
        FullSourceLoc(SM.getComposedLoc(TargetFile, Offset), SM),
        DiagnosticsEngine::Note, Message, {}, {});
  };
  for (auto RT : {RefType::Explicit, RefType::Implicit, RefType::Ambiguous}) {
    auto RTStr = llvm::to_string(RT);
    for (auto Expected : Target.points(RTStr))
      if (!llvm::is_contained(ReferencedOffsets[RT], Expected))
        DiagnosePoint("location not marked used with type " + RTStr, Expected);
    for (auto Actual : ReferencedOffsets[RT])
      if (!llvm::is_contained(Target.points(RTStr), Actual))
        DiagnosePoint("location unexpectedly used with type " + RTStr, Actual);
  }

  // If there were any differences, we print the entire referencing code once.
  if (!DiagBuf.empty())
    ADD_FAILURE() << DiagBuf << "\nfrom code:\n" << ReferencingCode;
  return TargetDecls;
}

TEST(WalkAST, DeclRef) {
  testWalk("int $explicit^x;", "int y = ^x;");
  testWalk("int $explicit^foo();", "int y = ^foo();");
  testWalk("namespace ns { int $explicit^x; }", "int y = ns::^x;");
  testWalk("struct S { static int x; };", "int y = S::^x;");
  // Canonical declaration only.
  testWalk("extern int $explicit^x; int x;", "int y = ^x;");
  // Return type of `foo` isn't used.
  testWalk("struct S{}; S $explicit^foo();", "auto bar() { return ^foo(); }");
}

TEST(WalkAST, TagType) {
  testWalk("struct $explicit^S {};", "^S *y;");
  testWalk("enum $explicit^E {};", "^E *y;");
  testWalk("struct $explicit^S { static int x; };", "int y = ^S::x;");
  // One explicit call from the TypeLoc in constructor spelling, another
  // implicit reference through the constructor call.
  testWalk("struct $explicit^$implicit^S { static int x; };", "auto y = ^S();");
}

TEST(WalkAST, ClassTemplates) {
  // Explicit instantiation and (partial) specialization references primary
  // template.
  EXPECT_THAT(testWalk("template<typename> struct $explicit^Foo{};",
                       "template struct ^Foo<int>;"),
              ElementsAre(Decl::CXXRecord));
  EXPECT_THAT(testWalk("template<typename> struct $explicit^Foo{};",
                       "template<> struct ^Foo<int> {};"),
              ElementsAre(Decl::CXXRecord));
  EXPECT_THAT(testWalk("template<typename> struct $explicit^Foo{};",
                       "template<typename T> struct ^Foo<T*> {};"),
              ElementsAre(Decl::CXXRecord));

  // Implicit instantiations references most relevant template.
  EXPECT_THAT(
      testWalk("template<typename> struct $explicit^Foo;", "^Foo<int> x();"),
      ElementsAre(Decl::Kind::ClassTemplate));
  EXPECT_THAT(
      testWalk("template<typename> struct $explicit^Foo {};", "^Foo<int> x;"),
      ElementsAre(Decl::CXXRecord));
  EXPECT_THAT(testWalk(R"cpp(
    template<typename> struct Foo {};
    template<> struct $explicit^Foo<int> {};)cpp",
                       "^Foo<int> x;"),
              ElementsAre(Decl::ClassTemplateSpecialization));
  EXPECT_THAT(testWalk(R"cpp(
    template<typename> struct Foo {};
    template<typename T> struct $explicit^Foo<T*> {};)cpp",
                       "^Foo<int *> x;"),
              ElementsAre(Decl::ClassTemplatePartialSpecialization));
  // Incomplete instantiations don't have a specific specialization associated.
  EXPECT_THAT(testWalk(R"cpp(
    template<typename> struct $explicit^Foo;
    template<typename T> struct Foo<T*>;)cpp",
                       "^Foo<int *> x();"),
              ElementsAre(Decl::Kind::ClassTemplate));
  EXPECT_THAT(testWalk(R"cpp(
    template<typename> struct $explicit^Foo {};
    template struct Foo<int>;)cpp",
                       "^Foo<int> x;"),
              ElementsAre(Decl::CXXRecord));
  // FIXME: This is broken due to
  // https://github.com/llvm/llvm-project/issues/42259.
  EXPECT_THAT(testWalk(R"cpp(
    template<typename T> struct $explicit^Foo { Foo(T); };
    template<> struct Foo<int> { Foo(int); };)cpp",
                       "^Foo x(3);"),
              ElementsAre(Decl::ClassTemplate));
}
TEST(WalkAST, VarTemplates) {
  // Explicit instantiation and (partial) specialization references primary
  // template.
  // FIXME: Explicit instantiations has wrong source location, they point at the
  // primary template location (hence we drop the reference).
  EXPECT_THAT(
      testWalk("template<typename T> T Foo = 0;", "template int ^Foo<int>;"),
      ElementsAre());
  EXPECT_THAT(testWalk("template<typename T> T $explicit^Foo = 0;",
                       "template<> int ^Foo<int> = 2;"),
              ElementsAre(Decl::Var));
  EXPECT_THAT(testWalk("template<typename T> T $explicit^Foo = 0;",
                       "template<typename T> T* ^Foo<T*> = 1;"),
              ElementsAre(Decl::Var));

  // Implicit instantiations references most relevant template.
  // FIXME: This points at implicit specialization, instead we should point to
  // pattern.
  EXPECT_THAT(testWalk(R"cpp(
    template <typename T> T $explicit^Foo = 0;)cpp",
                       "int z = ^Foo<int>;"),
              ElementsAre(Decl::VarTemplateSpecialization));
  EXPECT_THAT(testWalk(R"cpp(
    template<typename T> T Foo = 0;
    template<> int $explicit^Foo<int> = 1;)cpp",
                       "int x = ^Foo<int>;"),
              ElementsAre(Decl::VarTemplateSpecialization));
  // FIXME: This points at implicit specialization, instead we should point to
  // explicit partial specializaiton pattern.
  EXPECT_THAT(testWalk(R"cpp(
    template<typename T> T Foo = 0;
    template<typename T> T* $explicit^Foo<T*> = nullptr;)cpp",
                       "int *x = ^Foo<int *>;"),
              ElementsAre(Decl::VarTemplateSpecialization));
  // Implicit specializations through explicit instantiations has source
  // locations pointing at the primary template.
  EXPECT_THAT(testWalk(R"cpp(
    template<typename T> T $explicit^Foo = 0;
    template int Foo<int>;)cpp",
                       "int x = ^Foo<int>;"),
              ElementsAre(Decl::VarTemplateSpecialization));
}
TEST(WalkAST, FunctionTemplates) {
  // Explicit instantiation and (partial) specialization references primary
  // template.
  // FIXME: Explicit instantiations has wrong source location, they point at the
  // primary template location (hence we drop the reference).
  EXPECT_THAT(testWalk("template<typename T> void foo(T) {}",
                       "template void ^foo<int>(int);"),
              ElementsAre());
  EXPECT_THAT(testWalk("template<typename T> void $explicit^foo(T);",
                       "template<> void ^foo<int>(int);"),
              ElementsAre(Decl::FunctionTemplate));

  // Implicit instantiations references most relevant template.
  EXPECT_THAT(testWalk(R"cpp(
    template <typename T> void $explicit^foo() {})cpp",
                       "auto x = []{ ^foo<int>(); };"),
              ElementsAre(Decl::Function));
  EXPECT_THAT(testWalk(R"cpp(
    template<typename T> void foo() {}
    template<> void $explicit^foo<int>(){})cpp",
                       "auto x = []{ ^foo<int>(); };"),
              ElementsAre(Decl::Function));
  // The decl is actually the specialization, but explicit instantations point
  // at the primary template.
  EXPECT_THAT(testWalk(R"cpp(
    template<typename T> void $explicit^foo() {};
    template void foo<int>();)cpp",
                       "auto x = [] { ^foo<int>(); };"),
              ElementsAre(Decl::Function));
}
TEST(WalkAST, TemplateSpecializationsFromUsingDecl) {
  // Class templates
  testWalk(R"cpp(
namespace ns {
template<class T> class $explicit^Z {};      // primary template
template<class T> class $ambiguous^Z<T*> {};  // partial specialization
template<> class $ambiguous^Z<int> {};        // full specialization
}
  )cpp",
           "using ns::^Z;");

  // Var templates
  testWalk(R"cpp(
namespace ns {
template<class T> T $explicit^foo;      // primary template
template<class T> T $ambiguous^foo<T*>;  // partial specialization
template<> int* $ambiguous^foo<int>;     // full specialization
}
  )cpp",
           "using ns::^foo;");
  // Function templates, no partial template specializations.
  testWalk(R"cpp(
namespace ns {
template<class T> void $ambiguous^function(T);  // primary template
template<> void $ambiguous^function(int);       // full specialization
}
  )cpp",
           "using ns::^function;");
}

TEST(WalkAST, Alias) {
  testWalk(R"cpp(
    namespace ns { int x; }
    using ns::$explicit^x;
  )cpp",
           "int y = ^x;");
  testWalk("using $explicit^foo = int;", "^foo x;");
  testWalk("struct S {}; using $explicit^foo = S;", "^foo x;");
  testWalk(R"cpp(
    template<typename> struct Foo {};
    template<> struct Foo<int> {};
    namespace ns { using ::$explicit^Foo; })cpp",
           "ns::^Foo<int> x;");
  testWalk(R"cpp(
    template<typename> struct Foo {};
    namespace ns { using ::Foo; }
    template<> struct ns::$explicit^Foo<int> {};)cpp",
           "^Foo<int> x;");
  // AST doesn't have enough information to figure out whether specialization
  // happened through an exported type or not. So err towards attributing use to
  // the using-decl, specializations on the exported type should be rare and
  // they're not permitted on type-aliases.
  testWalk(R"cpp(
    template<typename> struct Foo {};
    namespace ns { using ::$explicit^Foo; }
    template<> struct ns::Foo<int> {};)cpp",
           "ns::^Foo<int> x;");
  testWalk(R"cpp(
    namespace ns { enum class foo { bar }; }
    using ns::foo;)cpp",
           "auto x = foo::^bar;");
  testWalk(R"cpp(
    namespace ns { enum foo { bar }; }
    using ns::foo::$explicit^bar;)cpp",
           "auto x = ^bar;");
}

TEST(WalkAST, Using) {
  // We should report unused overloads as ambiguous.
  testWalk(R"cpp(
    namespace ns {
      void $explicit^x(); void $ambiguous^x(int); void $ambiguous^x(char);
    })cpp",
           "using ns::^x; void foo() { x(); }");
  testWalk(R"cpp(
    namespace ns {
      void $ambiguous^x(); void $ambiguous^x(int); void $ambiguous^x(char);
    })cpp",
           "using ns::^x;");
  testWalk("namespace ns { struct S; } using ns::$explicit^S;", "^S *s;");

  testWalk(R"cpp(
    namespace ns {
      template<class T>
      class $explicit^Y {};
    })cpp",
           "using ns::^Y;");
  testWalk(R"cpp(
    namespace ns {
      class $explicit^Y {};
    })cpp",
           "using ns::^Y;");
  testWalk(R"cpp(
    namespace ns {
      template<class T>
      class Y {};
    }
    using ns::$explicit^Y;)cpp",
           "^Y<int> x;");
  testWalk("namespace ns { enum E {A}; } using enum ns::$explicit^E;",
           "auto x = ^A;");
}

TEST(WalkAST, Namespaces) {
  testWalk("namespace ns { void x(); }", "using namespace ^ns;");
}

TEST(WalkAST, TemplateNames) {
  testWalk("template<typename> struct $explicit^S {};", "^S<int> s;");
  // FIXME: Template decl has the wrong primary location for type-alias template
  // decls.
  testWalk(R"cpp(
      template <typename> struct S {};
      template <typename T> $explicit^using foo = S<T>;)cpp",
           "^foo<int> x;");
  testWalk(R"cpp(
      namespace ns {template <typename> struct S {}; }
      using ns::$explicit^S;)cpp",
           "^S<int> x;");
  testWalk(R"cpp(
      namespace ns {
        template <typename T> struct S { S(T);};
        template <typename T> S(T t) -> S<T>;
      }
      using ns::$explicit^S;)cpp",
           "^S x(123);");
  testWalk("template<typename> struct $explicit^S {};",
           R"cpp(
      template <template <typename> typename> struct X {};
      X<^S> x;)cpp");
  testWalk("template<typename T> struct $explicit^S { S(T); };", "^S s(42);");
}

TEST(WalkAST, NestedTypes) {
  testWalk(R"cpp(
      struct Base { typedef int $implicit^a; };
      struct Derived : public Base {};)cpp",
           "void fun() { Derived::^a x; }");
  testWalk(R"cpp(
      struct Base { using $implicit^a = int; };
      struct Derived : public Base {};)cpp",
           "void fun() { Derived::^a x; }");
  testWalk(R"cpp(
      struct ns { struct a {}; };
      struct Base : public ns { using ns::$implicit^a; };
      struct Derived : public Base {};)cpp",
           "void fun() { Derived::^a x; }");
  testWalk(R"cpp(
      struct Base { struct $implicit^a {}; };
      struct Derived : public Base {};)cpp",
           "void fun() { Derived::^a x; }");
  testWalk("struct Base { struct $implicit^a {}; };",
           "struct Derived : public Base { ^a x; };");
  testWalk(R"cpp(
      struct Base { struct $implicit^a {}; };
      struct Derived : public Base {};
      struct SoDerived : public Derived {};
      )cpp",
           "void fun() { SoDerived::Derived::^a x; }");
}

TEST(WalkAST, MemberExprs) {
  testWalk("struct S { static int f; };", "void foo() { S::^f; }");
  testWalk("struct B { static int f; }; struct S : B {};",
           "void foo() { S::^f; }");
  testWalk("struct B { static void f(); }; struct S : B {};",
           "void foo() { S::^f; }");
  testWalk("struct B { static void f(); }; ",
           "struct S : B { void foo() { ^f(); } };");
  testWalk("struct $implicit^S { void foo(); };", "void foo() { S{}.^foo(); }");
  testWalk(
      "struct S { void foo(); }; struct $implicit^X : S { using S::foo; };",
      "void foo() { X{}.^foo(); }");
  testWalk("struct Base { int a; }; struct $implicit^Derived : public Base {};",
           "void fun(Derived d) { d.^a; }");
  testWalk("struct Base { int a; }; struct $implicit^Derived : public Base {};",
           "void fun(Derived* d) { d->^a; }");
  testWalk("struct Base { int a; }; struct $implicit^Derived : public Base {};",
           "void fun(Derived& d) { d.^a; }");
  testWalk("struct Base { int a; }; struct $implicit^Derived : public Base {};",
           "void fun() { Derived().^a; }");
  testWalk("struct Base { int a; }; struct $implicit^Derived : public Base {};",
           "Derived foo(); void fun() { foo().^a; }");
  testWalk("struct Base { int a; }; struct $implicit^Derived : public Base {};",
           "Derived& foo(); void fun() { foo().^a; }");
  testWalk(R"cpp(
      template <typename T>
      struct unique_ptr {
        T *operator->();
      };
      struct $implicit^Foo { int a; };)cpp",
           "void test(unique_ptr<Foo> &V) { V->^a; }");
  testWalk(R"cpp(
      template <typename T>
      struct $implicit^unique_ptr {
        void release();
      };
      struct Foo {};)cpp",
           "void test(unique_ptr<Foo> &V) { V.^release(); }");
  // Respect the sugar type (typedef, using-type).
  testWalk(R"cpp(
      namespace ns { struct Foo { int a; }; }
      using $implicit^Bar = ns::Foo;)cpp",
           "void test(Bar b) { b.^a; }");
  testWalk(R"cpp(
      namespace ns { struct Foo { int a; }; }
      using ns::$implicit^Foo;)cpp",
           "void test(Foo b) { b.^a; }");
  testWalk(R"cpp(
      namespace ns { struct Foo { int a; }; }
      namespace ns2 { using Bar = ns::Foo; }
      using ns2::$implicit^Bar;
      )cpp",
           "void test(Bar b) { b.^a; }");
  testWalk(R"cpp(
      namespace ns { template<typename> struct Foo { int a; }; }
      using ns::$implicit^Foo;)cpp",
           "void k(Foo<int> b) { b.^a; }");
  // Test the dependent-type case (CXXDependentScopeMemberExpr)
  testWalk("template<typename T> struct $implicit^Base { void method(); };",
           "template<typename T> void k(Base<T> t) { t.^method(); }");
  testWalk("template<typename T> struct $implicit^Base { void method(); };",
           "template<typename T> void k(Base<T>& t) { t.^method(); }");
  testWalk("template<typename T> struct $implicit^Base { void method(); };",
           "template<typename T> void k(Base<T>* t) { t->^method(); }");
}

TEST(WalkAST, ConstructExprs) {
  testWalk("struct $implicit^S {};", "S ^t;");
  testWalk("struct $implicit^S { S(); };", "S ^t;");
  testWalk("struct $implicit^S { S(int); };", "S ^t(42);");
  testWalk("struct $implicit^S { S(int); };", "S t = ^42;");
  testWalk("namespace ns { struct S{}; } using ns::$implicit^S;", "S ^t;");
}

TEST(WalkAST, Operator) {
  // Operator calls are marked as implicit references as they're ADL-used and
  // type should be providing them.
  testWalk(
      "struct string { friend int $implicit^operator+(string, string); }; ",
      "int k = string() ^+ string();");
  // Treat member operators as regular member expr calls.
  testWalk("struct $implicit^string {int operator+(string); }; ",
           "int k = string() ^+ string();");
  // Make sure usage is attributed to the alias.
  testWalk(
      "struct string {int operator+(string); }; using $implicit^foo = string;",
      "int k = foo() ^+ string();");
}

TEST(WalkAST, VarDecls) {
  // Definition uses declaration, not the other way around.
  testWalk("extern int $explicit^x;", "int ^x = 1;");
  testWalk("int x = 1;", "extern int ^x;");
}

TEST(WalkAST, Functions) {
  // Definition uses declaration, not the other way around.
  testWalk("void $explicit^foo();", "void ^foo() {}");
  testWalk("void foo() {}", "void ^foo();");
  testWalk("template <typename> void $explicit^foo();",
           "template <typename> void ^foo() {}");

  // Unresolved calls marks all the overloads.
  testWalk("void $ambiguous^foo(int); void $ambiguous^foo(char);",
           "template <typename T> void bar() { ^foo(T{}); }");
}

TEST(WalkAST, Enums) {
  testWalk("enum E { $explicit^A = 42 };", "int e = ^A;");
  testWalk("enum class $explicit^E : int;", "enum class ^E : int {};");
  testWalk("enum class E : int {};", "enum class ^E : int ;");
  testWalk("namespace ns { enum E { $explicit^A = 42 }; }", "int e = ns::^A;");
  testWalk("namespace ns { enum E { A = 42 }; } using ns::E::$explicit^A;",
           "int e = ^A;");
  testWalk("namespace ns { enum E { A = 42 }; } using enum ns::$explicit^E;",
           "int e = ^A;");
  testWalk(R"(namespace ns { enum E { A = 42 }; }
              struct S { using enum ns::E; };)",
           "int e = S::^A;");
  testWalk(R"(namespace ns { enum E { A = 42 }; }
              struct S { using ns::E::A; };)",
           "int e = S::^A;");
  testWalk(R"(namespace ns { enum E { $explicit^A = 42 }; })",
           "namespace z = ns; int e = z::^A;");
  testWalk(R"(enum E { $explicit^A = 42 };)", "int e = ::^A;");
}

TEST(WalkAST, InitializerList) {
  testWalk(R"cpp(
       namespace std {
        template <typename T> struct $implicit^initializer_list { const T *a, *b; };
       })cpp",
           R"cpp(
       const char* s = "";
       auto sx = ^{s};)cpp");
}

TEST(WalkAST, Concepts) {
  std::string Concept = "template<typename T> concept $explicit^Foo = true;";
  testWalk(Concept, "template<typename T>concept Bar = ^Foo<T> && true;");
  testWalk(Concept, "template<^Foo T>void func() {}");
  testWalk(Concept, "template<typename T> requires ^Foo<T> void func() {}");
  testWalk(Concept, "template<typename T> void func() requires ^Foo<T> {}");
  testWalk(Concept, "void func(^Foo auto x) {}");
  testWalk(Concept, "void func() { ^Foo auto x = 1; }");
}

TEST(WalkAST, FriendDecl) {
  testWalk("void $explicit^foo();", "struct Bar { friend void ^foo(); };");
  testWalk("struct $explicit^Foo {};", "struct Bar { friend struct ^Foo; };");
}

TEST(WalkAST, OperatorNewDelete) {
  testWalk("void* $ambiguous^operator new(decltype(sizeof(int)), void*);",
           "struct Bar { void foo() { Bar b; ^new (&b) Bar; } };");
  testWalk("struct A { static void $ambiguous^operator delete(void*); };",
           "void foo() { A a; ^delete &a; }");
}

TEST(WalkAST, CleanupAttr) {
  testWalk("void* $explicit^freep(void *p);",
           "void foo() { __attribute__((__cleanup__(^freep))) char* x = 0; }");
}

TEST(WalkAST, ObjCInterfaceTypeLoc) {
  testWalk(R"objc(
    @interface $explicit^MyClass
    @end
  )objc",
           R"objc(
    void test() {
      ^MyClass *obj;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCImplementationDeclDependsOnInterface) {
  testWalk(R"objc(
    @interface $explicit^MyClass
    @end
  )objc",
           R"objc(
    @implementation ^MyClass
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCMessageExprSelectorLoc) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    $explicit^- (void)doSomething;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      [obj ^doSomething];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCMessageExprSelectorLocProtocol) {
  testWalk(R"objc(
    @protocol $implicit^MyProtocol
    $explicit^- (void)doSomething;
    @end
  )objc",
           R"objc(
    void test(id<MyProtocol> obj) {
      [obj ^doSomething];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCMessageExprSelectorLocNestedProtocol) {
  testWalk(R"objc(
    @protocol FirstProtocol
        $explicit^- (void)doSomething;
    @end
    @protocol $implicit^SecondProtocol <FirstProtocol>
    @end
  )objc",
           R"objc(
    void test(id<SecondProtocol> obj) {
      [obj ^doSomething];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCMessageExprSelectorLocMultipleProtocol) {
  testWalk(R"objc(
    @protocol $implicit^FirstProtocol
    @end
    @protocol $implicit^SecondProtocol
      $explicit^- (void)doSomething;
    @end
  )objc",
           R"objc(
    void test(id<FirstProtocol, SecondProtocol> obj) {
      [obj ^doSomething];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCMessageExprSelectorMessageChaining) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    $explicit^- (void)doSomething;
    @end
    @interface WrapperClass
    - (MyClass *)myClass;
    @end
  )objc",
           R"objc(
    void test(WrapperClass *obj) {
      // Weird space avoids Annotations thinking this is a range.
      [ [obj myClass] ^doSomething];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCMessageExprClassReceiver) {
  testWalk(R"objc(
    @interface $explicit^MyClass
    + (void)classMethod;
    @end
  )objc",
           R"objc(
    void test() {
      [^MyClass classMethod];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprExplicit) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    @property(nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      int x = obj.^foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprImplicitGetter) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    $explicit^- (int)foo;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      int x = obj.^foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprImplicitSetter) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    $explicit^- (void)setFoo:(int)val;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      obj.^foo = 42;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprExplicitSetter) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    @property(nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      obj.^foo = 42;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprDesugaredSetter) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    @property(nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      [obj ^setFoo:42];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprDesugaredGetter) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    @property(nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      [obj ^foo];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprDesugaredClassSetter) {
  testWalk(R"objc(
    @interface MyClass
    @property(class) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test() {
      [MyClass ^setFoo:42];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprDesugaredClassGetter) {
  testWalk(R"objc(
    @interface MyClass
    @property(class) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test() {
      [MyClass ^foo];
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprProtocol) {
  testWalk(R"objc(
    @protocol $implicit^MyProtocol
    @property(nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test(id<MyProtocol> obj) {
      int x = obj.^foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprNestedProtocol) {
  testWalk(R"objc(
    @protocol FirstProtocol
    @property(nonatomic) int $explicit^foo;
    @end
    @protocol $implicit^SecondProtocol <FirstProtocol>
    @end
  )objc",
           R"objc(
    void test(id<SecondProtocol> obj) {
      int x = obj.^foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprMultipleProtocol) {
  testWalk(R"objc(
    @protocol $implicit^FirstProtocol
    @end
    @protocol $implicit^SecondProtocol
    @property(nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test(id<FirstProtocol, SecondProtocol> obj) {
      int x = obj.^foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprClassReceiver) {
  testWalk(R"objc(
    @interface MyClass
    @property(class, nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test() {
      int x = MyClass.^foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprClassReceiverInterface) {
  testWalk(R"objc(
    @interface $explicit^MyClass
    @property(class, nonatomic) int foo;
    @end
  )objc",
           R"objc(
    void test() {
      int x = ^MyClass.foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprSuperReceiver) {
  testWalk(R"objc(
    @interface $implicit^ParentClass
    @property(nonatomic) int $explicit^foo;
    @end
    @interface MyClass : ParentClass
    @end
  )objc",
           R"objc(
    @implementation MyClass
    - (void)testSummary {
      int x = super.^foo;
    }
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprClassSuperReceiver) {
  testWalk(R"objc(
    @interface $implicit^ParentClass
    @property(class, nonatomic) int $explicit^foo;
    @end
    @interface MyClass : ParentClass
    @end
  )objc",
           R"objc(
    @implementation MyClass
    + (void)testSummary {
      int x = super.^foo;
    }
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprClassSuperSetter) {
  testWalk(R"objc(
    @interface $implicit^ParentClass
    @property(class, nonatomic) int $explicit^foo;
    @end
    @interface MyClass : ParentClass
    @end
  )objc",
           R"objc(
    @implementation MyClass
    + (void)testSummary {
      super.^foo = 1;
    }
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprClassSuperProtocolReceiver) {
  testWalk(R"objc(
    @protocol MyProtocol
    @property(class) int $explicit^foo;
    @end
    @interface $implicit^ParentClass <MyProtocol>
    @end
    @interface MyClass : ParentClass
    @end
  )objc",
           R"objc(
    @implementation MyClass
    + (void)testSummary {
      int x = super.^foo;
    }
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprSuperMultipleProtocolReceiver) {
  testWalk(R"objc(
    @protocol FirstProtocol
    @end
    @protocol SecondProtocol
    @property(nonatomic) int $explicit^foo;
    @end
    @interface $implicit^ParentClass <FirstProtocol, SecondProtocol>
    @end
    @interface MyClass : ParentClass
    @end
  )objc",
           R"objc(
    @implementation MyClass
    - (void)testSummary {
      int x = super.^foo;
    }
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprSuperNestedProtocolReceiver) {
  testWalk(R"objc(
    @protocol FirstProtocol
    @property(nonatomic) int $explicit^foo;
    @end
    @protocol SecondProtocol <FirstProtocol>
    @end
    @interface $implicit^ParentClass <SecondProtocol>
    @end
    @interface MyClass : ParentClass
    @end
  )objc",
           R"objc(
    @implementation MyClass
    - (void)testSummary {
      int x = super.^foo;
    }
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCProtocolInType) {
  testWalk(R"objc(
    @protocol $explicit^MyProtocol
    @end
  )objc",
           R"objc(
    void test() {
      id<^MyProtocol> obj;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCProtocolInClassInterface) {
  testWalk(R"objc(
    @protocol $explicit^MyProtocol
    @end
  )objc",
           R"objc(
    @interface MyClass <^MyProtocol>
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCProtocolInProtocolInheritance) {
  testWalk(R"objc(
    @protocol $explicit^ParentProtocol
    @end
  )objc",
           R"objc(
    @protocol MyProtocol <^ParentProtocol>
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCProtocolExpr) {
  testWalk(R"objc(
    @protocol $explicit^MyProtocol
    @end
  )objc",
           R"objc(
    void test() {
      Protocol* p = @protocol(^MyProtocol);
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCCategoryDeclDependsOnInterface) {
  testWalk(R"objc(
    @interface $explicit^MyClass
    @end
  )objc",
           R"objc(
    @interface ^MyClass (Category)
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCCategoryImplDependsOnInterface) {
  testWalk(R"objc(
    @interface $explicit^MyClass
    @end
  )objc",
           R"objc(
    @interface MyClass (Category)
    @end
    @implementation ^MyClass (Category)
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCCategoryImplDependsOnCategoryDecl) {
  testWalk(R"objc(
    @interface MyClass
    @end
    @interface $explicit^MyClass (Category)
    @end
  )objc",
           R"objc(
    @implementation MyClass (^Category)
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCImplicitCastToProtocolConformingCategory) {
  testWalk(R"objc(
    @protocol MyProtocol
    @end
    @interface MyClass
    @end
    @interface $implicit^MyClass (MyCategory) <MyProtocol>
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      id<MyProtocol> p = ^obj;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCCompatibleAliasDecl) {
  testWalk(R"objc(
    @interface $explicit^MyClass
    @end
  )objc",
           R"objc(
    ^@compatibility_alias AliasName MyClass;
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCCompatibleAliasUsage) {
  testWalk(R"objc(
    @interface $explicit^MyClass
    @end
    @compatibility_alias AliasName MyClass;
  )objc",
           R"objc(
    void test() {
      ^AliasName *obj;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCIvarRefExprExplicit) {
  testWalk(R"objc(
    @interface MyClass {
      @public
      int $explicit^foo;
    }
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      int x = obj->^foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCIvarRefExprFree) {
  testWalk(R"objc(
    @interface MyClass {
      int $explicit^foo;
    }
    @end
  )objc",
           R"objc(
    @implementation MyClass
    - (void)test {
      int x = ^foo;
    }
    @end
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCSelectorExpr) {
  testWalk(R"objc(
    @interface MyClass
    $ambiguous^- (void)doSomething;
    @end
  )objc",
           R"objc(
    void test() {
      SEL s = @selector(^doSomething);
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCSelectorExprPropertyGetter) {
  auto Decls = testWalk(R"objc(
    @interface MyClass
    @property(nonatomic) int $ambiguous^foo;
    @end
  )objc",
                        R"objc(
    void test() {
      SEL s = @selector(^foo);
    }
  )objc",
                        {"-x", "objective-c"});
  EXPECT_THAT(Decls, ElementsAre(Decl::ObjCProperty));
}

TEST(WalkAST, ObjCSelectorExprPropertySetter) {
  auto Decls = testWalk(R"objc(
    @interface MyClass
    @property(nonatomic) int $ambiguous^foo;
    @end
  )objc",
                        R"objc(
    void test() {
      SEL s = @selector(^setFoo:);
    }
  )objc",
                        {"-x", "objective-c"});
  EXPECT_THAT(Decls, ElementsAre(Decl::ObjCProperty));
}

TEST(WalkAST, ObjCSelectorExprReadOnlyPropertySetter) {
  // Read-only properties do not generate setter selectors.
  testWalk(R"objc(
    @interface MyClass
    @property(readonly, nonatomic) int foo;
    @end
  )objc",
           R"objc(
    void test() {
      SEL s = @selector(^setFoo:);
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCSelectorExprMultipleMatches) {
  testWalk(R"objc(
    @interface MyClass1
    $ambiguous^- (void)doSomething;
    @end

    @interface MyClass2
    $ambiguous^- (void)doSomething;
    @end
  )objc",
           R"objc(
    void test() {
      SEL s = @selector(^doSomething);
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCSelectorExprInProtocol) {
  testWalk(R"objc(
    @protocol MyProtocol
    $ambiguous^- (void)protocolMethod;
    @end
  )objc",
           R"objc(
    void test() {
      SEL s = @selector(^protocolMethod);
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCSelectorExprMultiColon) {
  testWalk(R"objc(
    @interface MyClass
    $ambiguous^- (void)doA:(int)a withB:(int)b;
    @end
  )objc",
           R"objc(
    void test() {
      SEL s = @selector(^doA:withB:);
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprCustomGetter) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    @property(getter=isFoo, setter=setTheFoo:, nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      int x = obj.^foo;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCPropertyRefExprCustomSetter) {
  testWalk(R"objc(
    @interface $implicit^MyClass
    @property(getter=isFoo, setter=setTheFoo:, nonatomic) int $explicit^foo;
    @end
  )objc",
           R"objc(
    void test(MyClass *obj) {
      obj.^foo = 42;
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCSelectorExprCustomPropertyGetter) {
  auto Decls = testWalk(R"objc(
    @interface MyClass
    @property(getter=isFoo, setter=setTheFoo:, nonatomic) int $ambiguous^foo;
    @end
  )objc",
                        R"objc(
    void test() {
      SEL s = @selector(^isFoo);
    }
  )objc",
                        {"-x", "objective-c"});
  EXPECT_THAT(Decls, ElementsAre(Decl::ObjCProperty));
}

TEST(WalkAST, ObjCSelectorExprCustomPropertySetter) {
  auto Decls = testWalk(R"objc(
    @interface MyClass
    @property(getter=isFoo, setter=setTheFoo:, nonatomic) int $ambiguous^foo;
    @end
  )objc",
                        R"objc(
    void test() {
      SEL s = @selector(^setTheFoo:);
    }
  )objc",
                        {"-x", "objective-c"});
  EXPECT_THAT(Decls, ElementsAre(Decl::ObjCProperty));
}

TEST(WalkAST, ObjcSelectorDeclarationAndDefinition) {
  auto Decls = testWalk(R"objc(
    @interface MyClass
    $ambiguous^- (void)doSomething;
    @end
  )objc",
                        R"objc(
    @implementation MyClass
    - (void)doSomething {}
    @end
    void test() { SEL s = @selector(^doSomething); }
  )objc",
                        {"-x", "objective-c"});

  // This here: v
  EXPECT_THAT(Decls, ElementsAre(Decl::ObjCMethod));
}

TEST(WalkAST, ObjcSelectorMultipleIdenticalCalls) {
  testWalk(R"objc(
    @interface MyClass
    $ambiguous^- (void)doSomething;
    @end
  )objc",
           R"objc(
    void testA() { SEL s1 = @selector(^doSomething); }
    void testB() { SEL s2 = @selector(doSomething); }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjcSelectorInstanceAndClassMethodDisambiguation) {
  testWalk(R"objc(
    @interface MyClass
    $ambiguous^- (void)doSomething;
    $ambiguous^+ (void)doSomething;
    @end
  )objc",
           R"objc(
    void test() { SEL s = @selector(^doSomething); }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjcSelectorCustomGetterClashWithMethod) {
  testWalk(R"objc(
    @protocol MyProtocol
    $ambiguous^- (void)isFoo;
    @end

    @interface MyClass
    @property(getter=isFoo, nonatomic) int $ambiguous^foo;
    @end
  )objc",
           R"objc(
    void test() { SEL s = @selector(^isFoo); }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCAtCatchStmt) {
  testWalk(R"objc(
    @interface $explicit^CustomException
    @end
  )objc",
           R"objc(
    void test() {
      @try {}
      @catch (^CustomException *e) {}
    }
  )objc",
           {"-x", "objective-c", "-fobjc-exceptions"});
}

TEST(WalkAST, ObjCAtSynchronizedStmt) {
  testWalk(R"objc(
    @interface $explicit^LockObject
    + (id)sharedLock;
    @end
  )objc",
           R"objc(
    void test() {
      @synchronized([^LockObject sharedLock]) {}
    }
  )objc",
           {"-x", "objective-c"});
}

TEST(WalkAST, ObjCEncodeExpr) {
  testWalk(R"objc(
    struct $explicit^MyStruct { int x; };
  )objc",
           R"objc(
    void test() {
      const char *enc = @encode(struct ^MyStruct);
    }
  )objc",
           {"-x", "objective-c"});
}

} // namespace
} // namespace clang::include_cleaner
