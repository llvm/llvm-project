//===-- ClangdUnitTests.cpp - ClangdUnit tests ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdUnit.h"
#include "Annotations.h"
#include "TestFS.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Frontend/Utils.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
using namespace llvm;
void PrintTo(const DiagWithFixIts &D, std::ostream *O) {
  llvm::raw_os_ostream OS(*O);
  OS << D.Diag;
  if (!D.FixIts.empty()) {
    OS << " {";
    const char *Sep = "";
    for (const auto &F : D.FixIts) {
      OS << Sep << F;
      Sep = ", ";
    }
    OS << "}";
  }
}

namespace {
using testing::ElementsAre;

// FIXME: this is duplicated with FileIndexTests. Share it.
ParsedAST build(StringRef Code, std::vector<const char*> Flags = {}) {
  std::vector<const char*> Cmd = {"clang", "main.cpp"};
  Cmd.insert(Cmd.begin() + 1, Flags.begin(), Flags.end());
  auto CI = createInvocationFromCommandLine(Cmd);
  auto Buf = MemoryBuffer::getMemBuffer(Code);
  auto AST = ParsedAST::Build(std::move(CI), nullptr, std::move(Buf),
                              std::make_shared<PCHContainerOperations>(),
                              vfs::getRealFileSystem());
  assert(AST.hasValue());
  return std::move(*AST);
}

MATCHER_P2(Diag, Range, Message,
           "Diagnostic at " + llvm::to_string(Range) + " = [" + Message + "]") {
  return arg.Diag.range == Range && arg.Diag.message == Message &&
         arg.FixIts.empty();
}

MATCHER_P3(Fix, Range, Replacement, Message,
           "Fix " + llvm::to_string(Range) + " => " +
               testing::PrintToString(Replacement) + " = [" + Message + "]") {
  return arg.Diag.range == Range && arg.Diag.message == Message &&
         arg.FixIts.size() == 1 && arg.FixIts[0].range == Range &&
         arg.FixIts[0].newText == Replacement;
}

TEST(DiagnosticsTest, DiagnosticRanges) {
  // Check we report correct ranges, including various edge-cases.
  Annotations Test(R"cpp(
    void $decl[[foo]]();
    int main() {
      $typo[[go\
o]]();
      foo()$semicolon[[]]
      $unk[[unknown]]();
    }
  )cpp");
      llvm::errs() << Test.code();
      EXPECT_THAT(
          build(Test.code()).getDiagnostics(),
          ElementsAre(
              // This range spans lines.
              Fix(Test.range("typo"), "foo",
                  "use of undeclared identifier 'goo'; did you mean 'foo'?"),
              // This is a pretty normal range.
              Diag(Test.range("decl"), "'foo' declared here"),
              // This range is zero-width, and at the end of a line.
              Fix(Test.range("semicolon"), ";",
                  "expected ';' after expression"),
              // This range isn't provided by clang, we expand to the token.
              Diag(Test.range("unk"),
                   "use of undeclared identifier 'unknown'")));
}

TEST(DiagnosticsTest, FlagsMatter) {
  Annotations Test("[[void]] main() {}");
  EXPECT_THAT(
      build(Test.code()).getDiagnostics(),
      ElementsAre(Fix(Test.range(), "int", "'main' must return 'int'")));
  // Same code built as C gets different diagnostics.
  EXPECT_THAT(
      build(Test.code(), {"-x", "c"}).getDiagnostics(),
      ElementsAre(
          // FIXME: ideally this would be one diagnostic with a named FixIt.
          Diag(Test.range(), "return type of 'main' is not 'int'"),
          Fix(Test.range(), "int", "change return type to 'int'")));
}

TEST(DiagnosticsTest, Preprocessor) {
  // This looks like a preamble, but there's an #else in the middle!
  // Check that:
  //  - the #else doesn't generate diagnostics (we had this bug)
  //  - we get diagnostics from the taken branch
  //  - we get no diagnostics from the not taken branch
  Annotations Test(R"cpp(
    #ifndef FOO
    #define FOO
      int a = [[b]];
    #else
      int x = y;
    #endif
    )cpp");
  EXPECT_THAT(
      build(Test.code()).getDiagnostics(),
      ElementsAre(Diag(Test.range(), "use of undeclared identifier 'b'")));
}

} // namespace
} // namespace clangd
} // namespace clang
