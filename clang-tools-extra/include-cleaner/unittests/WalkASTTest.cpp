//===--- WalkASTTest.cpp ------------------------------------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AnalysisInternal.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Testing/TestAST.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clang::include_cleaner {
namespace {

// Specifies a test of which symbols are referenced by a piece of code.
// Target should contain points annotated with the reference kind.
// Example:
//   Target:      int $explicit^foo();
//   Referencing: int x = ^foo();
// There must be exactly one referencing location marked.
void testWalk(llvm::StringRef TargetCode, llvm::StringRef ReferencingCode) {
  llvm::Annotations Target(TargetCode);
  llvm::Annotations Referencing(ReferencingCode);

  TestInputs Inputs(Referencing.code());
  Inputs.ExtraFiles["target.h"] = Target.code().str();
  Inputs.ExtraArgs.push_back("-include");
  Inputs.ExtraArgs.push_back("target.h");
  Inputs.ExtraArgs.push_back("-std=c++17");
  TestAST AST(Inputs);
  const auto &SM = AST.sourceManager();

  // We're only going to record references from the nominated point,
  // to the target file.
  FileID ReferencingFile = SM.getMainFileID();
  SourceLocation ReferencingLoc =
      SM.getComposedLoc(ReferencingFile, Referencing.point());
  FileID TargetFile = SM.translateFile(
      llvm::cantFail(AST.fileManager().getFileRef("target.h")));

  // Perform the walk, and capture the offsets of the referenced targets.
  std::unordered_map<RefType, std::vector<size_t>> ReferencedOffsets;
  for (Decl *D : AST.context().getTranslationUnitDecl()->decls()) {
    if (ReferencingFile != SM.getDecomposedExpansionLoc(D->getLocation()).first)
      continue;
    walkAST(*D, [&](SourceLocation Loc, NamedDecl &ND, RefType RT) {
      if (SM.getFileLoc(Loc) != ReferencingLoc)
        return;
      auto NDLoc = SM.getDecomposedLoc(SM.getFileLoc(ND.getLocation()));
      if (NDLoc.first != TargetFile)
        return;
      ReferencedOffsets[RT].push_back(NDLoc.second);
    });
  }
  for (auto &Entry : ReferencedOffsets)
    llvm::sort(Entry.second);

  // Compare results to the expected points.
  // For each difference, show the target point in context, like a diagnostic.
  std::string DiagBuf;
  llvm::raw_string_ostream DiagOS(DiagBuf);
  auto *DiagOpts = new DiagnosticOptions();
  DiagOpts->ShowLevel = 0;
  DiagOpts->ShowNoteIncludeStack = 0;
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
}

TEST(WalkAST, DeclRef) {
  testWalk("int $explicit^x;", "int y = ^x;");
  testWalk("int $explicit^foo();", "int y = ^foo();");
  testWalk("namespace ns { int $explicit^x; }", "int y = ns::^x;");
  testWalk("struct S { static int $explicit^x; };", "int y = S::^x;");
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
  testWalk("template<typename> struct $explicit^Foo {};", "^Foo<int> x;");
  testWalk(R"cpp(
    template<typename> struct Foo {};
    template<> struct $explicit^Foo<int> {};)cpp",
           "^Foo<int> x;");
  testWalk(R"cpp(
    template<typename> struct Foo {};
    template<typename T> struct $explicit^Foo<T*> { void x(); };)cpp",
           "^Foo<int *> x;");
  testWalk(R"cpp(
    template<typename> struct Foo {};
    template struct $explicit^Foo<int>;)cpp",
           "^Foo<int> x;");
  // FIXME: This is broken due to
  // https://github.com/llvm/llvm-project/issues/42259.
  testWalk(R"cpp(
    template<typename T> struct $explicit^Foo { Foo(T); };
    template<> struct Foo<int> { void get(); Foo(int); };)cpp",
           "^Foo x(3);");
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
      class $ambiguous^Y {};
    })cpp",
           "using ns::^Y;");
  testWalk(R"cpp(
    namespace ns {
      template<class T>
      class Y {};
    }
    using ns::$explicit^Y;)cpp",
           "^Y<int> x;");
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

TEST(WalkAST, MemberExprs) {
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
  // References to operators are not counted as uses.
  testWalk("struct string {}; int operator+(string, string);",
           "int k = string() ^+ string();");
  testWalk("struct string {int operator+(string); }; ",
           "int k = string() ^+ string();");
  testWalk("struct string { friend int operator+(string, string); }; ",
           "int k = string() ^+ string();");
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

  // Unresolved calls marks all the overloads.
  testWalk("void $ambiguous^foo(int); void $ambiguous^foo(char);",
           "template <typename T> void bar() { ^foo(T{}); }");
}

TEST(WalkAST, Enums) {
  testWalk("enum E { $explicit^A = 42, B = 43 };", "int e = ^A;");
  testWalk("enum class $explicit^E : int;", "enum class ^E : int {};");
  testWalk("enum class E : int {};", "enum class ^E : int ;");
}

} // namespace
} // namespace clang::include_cleaner
