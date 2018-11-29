//===-- ClangdUnitTests.cpp - ClangdUnit tests ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdUnit.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
using namespace llvm;

namespace {
using testing::ElementsAre;
using testing::Field;
using testing::IsEmpty;
using testing::Pair;

testing::Matcher<const Diag &> WithFix(testing::Matcher<Fix> FixMatcher) {
  return Field(&Diag::Fixes, ElementsAre(FixMatcher));
}

testing::Matcher<const Diag &> WithNote(testing::Matcher<Note> NoteMatcher) {
  return Field(&Diag::Notes, ElementsAre(NoteMatcher));
}

MATCHER_P2(Diag, Range, Message,
           "Diag at " + llvm::to_string(Range) + " = [" + Message + "]") {
  return arg.Range == Range && arg.Message == Message;
}

MATCHER_P3(Fix, Range, Replacement, Message,
           "Fix " + llvm::to_string(Range) + " => " +
               testing::PrintToString(Replacement) + " = [" + Message + "]") {
  return arg.Message == Message && arg.Edits.size() == 1 &&
         arg.Edits[0].range == Range && arg.Edits[0].newText == Replacement;
}

MATCHER_P(EqualToLSPDiag, LSPDiag,
          "LSP diagnostic " + llvm::to_string(LSPDiag)) {
  return std::tie(arg.range, arg.severity, arg.message) ==
         std::tie(LSPDiag.range, LSPDiag.severity, LSPDiag.message);
}

MATCHER_P(EqualToFix, Fix, "LSP fix " + llvm::to_string(Fix)) {
  if (arg.Message != Fix.Message)
    return false;
  if (arg.Edits.size() != Fix.Edits.size())
    return false;
  for (std::size_t I = 0; I < arg.Edits.size(); ++I) {
    if (arg.Edits[I].range != Fix.Edits[I].range ||
        arg.Edits[I].newText != Fix.Edits[I].newText)
      return false;
  }
  return true;
}

// Helper function to make tests shorter.
Position pos(int line, int character) {
  Position Res;
  Res.line = line;
  Res.character = character;
  return Res;
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
  EXPECT_THAT(
      TestTU::withCode(Test.code()).build().getDiagnostics(),
      ElementsAre(
          // This range spans lines.
          AllOf(Diag(Test.range("typo"),
                     "use of undeclared identifier 'goo'; did you mean 'foo'?"),
                WithFix(
                    Fix(Test.range("typo"), "foo", "change 'go\\ o' to 'foo'")),
                // This is a pretty normal range.
                WithNote(Diag(Test.range("decl"), "'foo' declared here"))),
          // This range is zero-width, and at the end of a line.
          AllOf(Diag(Test.range("semicolon"), "expected ';' after expression"),
                WithFix(Fix(Test.range("semicolon"), ";", "insert ';'"))),
          // This range isn't provided by clang, we expand to the token.
          Diag(Test.range("unk"), "use of undeclared identifier 'unknown'")));
}

TEST(DiagnosticsTest, FlagsMatter) {
  Annotations Test("[[void]] main() {}");
  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(TU.build().getDiagnostics(),
              ElementsAre(AllOf(Diag(Test.range(), "'main' must return 'int'"),
                                WithFix(Fix(Test.range(), "int",
                                            "change 'void' to 'int'")))));
  // Same code built as C gets different diagnostics.
  TU.Filename = "Plain.c";
  EXPECT_THAT(
      TU.build().getDiagnostics(),
      ElementsAre(AllOf(
          Diag(Test.range(), "return type of 'main' is not 'int'"),
          WithFix(Fix(Test.range(), "int", "change return type to 'int'")))));
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
      TestTU::withCode(Test.code()).build().getDiagnostics(),
      ElementsAre(Diag(Test.range(), "use of undeclared identifier 'b'")));
}

TEST(DiagnosticsTest, ToLSP) {
  clangd::Diag D;
  D.Message = "something terrible happened";
  D.Range = {pos(1, 2), pos(3, 4)};
  D.InsideMainFile = true;
  D.Severity = DiagnosticsEngine::Error;
  D.File = "foo/bar/main.cpp";

  clangd::Note NoteInMain;
  NoteInMain.Message = "declared somewhere in the main file";
  NoteInMain.Range = {pos(5, 6), pos(7, 8)};
  NoteInMain.Severity = DiagnosticsEngine::Remark;
  NoteInMain.File = "../foo/bar/main.cpp";
  NoteInMain.InsideMainFile = true;
  D.Notes.push_back(NoteInMain);

  clangd::Note NoteInHeader;
  NoteInHeader.Message = "declared somewhere in the header file";
  NoteInHeader.Range = {pos(9, 10), pos(11, 12)};
  NoteInHeader.Severity = DiagnosticsEngine::Note;
  NoteInHeader.File = "../foo/baz/header.h";
  NoteInHeader.InsideMainFile = false;
  D.Notes.push_back(NoteInHeader);

  clangd::Fix F;
  F.Message = "do something";
  D.Fixes.push_back(F);

  auto MatchingLSP = [](const DiagBase &D, llvm::StringRef Message) {
    clangd::Diagnostic Res;
    Res.range = D.Range;
    Res.severity = getSeverity(D.Severity);
    Res.message = Message;
    return Res;
  };

  // Diagnostics should turn into these:
  clangd::Diagnostic MainLSP = MatchingLSP(D, R"(Something terrible happened

main.cpp:6:7: remark: declared somewhere in the main file

../foo/baz/header.h:10:11:
note: declared somewhere in the header file)");

  clangd::Diagnostic NoteInMainLSP =
      MatchingLSP(NoteInMain, R"(Declared somewhere in the main file

main.cpp:2:3: error: something terrible happened)");

  // Transform dianostics and check the results.
  std::vector<std::pair<clangd::Diagnostic, std::vector<clangd::Fix>>> LSPDiags;
  toLSPDiags(D, [&](clangd::Diagnostic LSPDiag,
                    llvm::ArrayRef<clangd::Fix> Fixes) {
    LSPDiags.push_back({std::move(LSPDiag),
                        std::vector<clangd::Fix>(Fixes.begin(), Fixes.end())});
  });

  EXPECT_THAT(
      LSPDiags,
      ElementsAre(Pair(EqualToLSPDiag(MainLSP), ElementsAre(EqualToFix(F))),
                  Pair(EqualToLSPDiag(NoteInMainLSP), IsEmpty())));
}

TEST(ClangdUnitTest, GetBeginningOfIdentifier) {
  // First ^ is the expected beginning, last is the search position.
  for (const char *Text : {
           "int ^f^oo();", // inside identifier
           "int ^foo();",  // beginning of identifier
           "int ^foo^();", // end of identifier
           "int foo(^);",  // non-identifier
           "^int foo();",  // beginning of file (can't back up)
           "int ^f0^0();", // after a digit (lexing at N-1 is wrong)
           "int ^λλ^λ();", // UTF-8 handled properly when backing up
       }) {
    Annotations TestCase(Text);
    auto AST = TestTU::withCode(TestCase.code()).build();
    const auto &SourceMgr = AST.getASTContext().getSourceManager();
    SourceLocation Actual = getBeginningOfIdentifier(
        AST, TestCase.points().back(), SourceMgr.getMainFileID());
    Position ActualPos =
        offsetToPosition(TestCase.code(), SourceMgr.getFileOffset(Actual));
    EXPECT_EQ(TestCase.points().front(), ActualPos) << Text;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
