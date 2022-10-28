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
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <vector>

namespace clang::include_cleaner {
namespace {
using testing::Pair;
using testing::UnorderedElementsAre;

// Specifies a test of which symbols are referenced by a piece of code.
//
// Example:
//   Target:      int ^foo();
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
  std::vector<size_t> ReferencedOffsets;
  for (Decl *D : AST.context().getTranslationUnitDecl()->decls()) {
    if (ReferencingFile != SM.getDecomposedExpansionLoc(D->getLocation()).first)
      continue;
    walkAST(*D, [&](SourceLocation Loc, NamedDecl &ND) {
      if (SM.getFileLoc(Loc) != ReferencingLoc)
        return;
      auto NDLoc = SM.getDecomposedLoc(SM.getFileLoc(ND.getLocation()));
      if (NDLoc.first != TargetFile)
        return;
      ReferencedOffsets.push_back(NDLoc.second);
    });
  }
  llvm::sort(ReferencedOffsets);

  // Compare results to the expected points.
  // For each difference, show the target point in context, like a diagnostic.
  std::string DiagBuf;
  llvm::raw_string_ostream DiagOS(DiagBuf);
  auto *DiagOpts = new DiagnosticOptions();
  DiagOpts->ShowLevel = 0;
  DiagOpts->ShowNoteIncludeStack = 0;
  TextDiagnostic Diag(DiagOS, AST.context().getLangOpts(), DiagOpts);
  auto DiagnosePoint = [&](const char *Message, unsigned Offset) {
    Diag.emitDiagnostic(
        FullSourceLoc(SM.getComposedLoc(TargetFile, Offset), SM),
        DiagnosticsEngine::Note, Message, {}, {});
  };
  for (auto Expected : Target.points())
    if (!llvm::is_contained(ReferencedOffsets, Expected))
      DiagnosePoint("location not marked used", Expected);
  for (auto Actual : ReferencedOffsets)
    if (!llvm::is_contained(Target.points(), Actual))
      DiagnosePoint("location unexpectedly used", Actual);

  // If there were any differences, we print the entire referencing code once.
  if (!DiagBuf.empty())
    ADD_FAILURE() << DiagBuf << "\nfrom code:\n" << ReferencingCode;
}

TEST(WalkAST, DeclRef) {
  testWalk("int ^x;", "int y = ^x;");
  testWalk("int ^foo();", "int y = ^foo();");
  testWalk("namespace ns { int ^x; }", "int y = ns::^x;");
  testWalk("struct S { static int ^x; };", "int y = S::^x;");
  // Canonical declaration only.
  testWalk("extern int ^x; int x;", "int y = ^x;");
  // Return type of `foo` isn't used.
  testWalk("struct S{}; S ^foo();", "auto bar() { return ^foo(); }");
}

TEST(WalkAST, TagType) {
  testWalk("struct ^S {};", "^S *y;");
  testWalk("enum ^E {};", "^E *y;");
  testWalk("struct ^S { static int x; };", "int y = ^S::x;");
}

TEST(WalkAST, Alias) {
  testWalk(R"cpp(
    namespace ns { int x; }
    using ns::^x;
  )cpp",
           "int y = ^x;");
  testWalk("using ^foo = int;", "^foo x;");
  testWalk("struct S {}; using ^foo = S;", "^foo x;");
}

TEST(WalkAST, Using) {
  testWalk("namespace ns { void ^x(); void ^x(int); }", "using ns::^x;");
  testWalk("namespace ns { struct S; } using ns::^S;", "^S *s;");
}

TEST(WalkAST, Namespaces) {
  testWalk("namespace ns { void x(); }", "using namespace ^ns;");
}

TEST(WalkAST, TemplateNames) {
  testWalk("template<typename> struct ^S {};", "^S<int> s;");
  // FIXME: Template decl has the wrong primary location for type-alias template
  // decls.
  testWalk(R"cpp(
      template <typename> struct S {};
      template <typename T> ^using foo = S<T>;)cpp",
           "^foo<int> x;");
  testWalk(R"cpp(
      namespace ns {template <typename> struct S {}; }
      using ns::^S;)cpp",
           "^S<int> x;");
  testWalk("template<typename> struct ^S {};",
           R"cpp(
      template <template <typename> typename> struct X {};
      X<^S> x;)cpp");
  testWalk("template<typename T> struct ^S { S(T); };", "^S s(42);");
  // Should we mark the specialization instead?
  testWalk("template<typename> struct ^S {}; template <> struct S<int> {};",
           "^S<int> s;");
}

TEST(WalkAST, MemberExprs) {
  testWalk("struct S { void ^foo(); };", "void foo() { S{}.^foo(); }");
  testWalk("struct S { void foo(); }; struct X : S { using S::^foo; };",
           "void foo() { X{}.^foo(); }");
}

TEST(WalkAST, ConstructExprs) {
  testWalk("struct ^S {};", "S ^t;");
  testWalk("struct S { ^S(int); };", "S ^t(42);");
}

TEST(WalkAST, Functions) {
  // Definition uses declaration, not the other way around.
  testWalk("void ^foo();", "void ^foo() {}");
  testWalk("void foo() {}", "void ^foo();");

  // Unresolved calls marks all the overloads.
  testWalk("void ^foo(int); void ^foo(char);",
           "template <typename T> void bar() { ^foo(T{}); }");
}

TEST(WalkAST, Enums) {
  testWalk("enum E { ^A = 42, B = 43 };", "int e = ^A;");
  testWalk("enum class ^E : int;", "enum class ^E : int {};");
  testWalk("enum class E : int {};", "enum class ^E : int ;");
}

} // namespace
} // namespace clang::include_cleaner
