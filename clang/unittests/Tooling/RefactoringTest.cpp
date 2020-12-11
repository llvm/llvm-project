//===- unittest/Tooling/RefactoringTest.cpp - Refactoring unit tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring.h"
#include "ReplacementTest.h"
#include "RewriterTestContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

TEST_F(ReplacementTest, CanDeleteAllText) {
  FileID ID = Context.createInMemoryFile("input.cpp", "text");
  SourceLocation Location = Context.getLocation(ID, 1, 1);
  Replacement Replace(createReplacement(Location, 4, ""));
  EXPECT_TRUE(Replace.apply(Context.Rewrite));
  EXPECT_EQ("", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, CanDeleteAllTextInTextWithNewlines) {
  FileID ID = Context.createInMemoryFile("input.cpp", "line1\nline2\nline3");
  SourceLocation Location = Context.getLocation(ID, 1, 1);
  Replacement Replace(createReplacement(Location, 17, ""));
  EXPECT_TRUE(Replace.apply(Context.Rewrite));
  EXPECT_EQ("", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, CanAddText) {
  FileID ID = Context.createInMemoryFile("input.cpp", "");
  SourceLocation Location = Context.getLocation(ID, 1, 1);
  Replacement Replace(createReplacement(Location, 0, "result"));
  EXPECT_TRUE(Replace.apply(Context.Rewrite));
  EXPECT_EQ("result", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, CanReplaceTextAtPosition) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  SourceLocation Location = Context.getLocation(ID, 2, 3);
  Replacement Replace(createReplacement(Location, 12, "x"));
  EXPECT_TRUE(Replace.apply(Context.Rewrite));
  EXPECT_EQ("line1\nlixne4", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, CanReplaceTextAtPositionMultipleTimes) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  SourceLocation Location1 = Context.getLocation(ID, 2, 3);
  Replacement Replace1(createReplacement(Location1, 12, "x\ny\n"));
  EXPECT_TRUE(Replace1.apply(Context.Rewrite));
  EXPECT_EQ("line1\nlix\ny\nne4", Context.getRewrittenText(ID));

  // Since the original source has not been modified, the (4, 4) points to the
  // 'e' in the original content.
  SourceLocation Location2 = Context.getLocation(ID, 4, 4);
  Replacement Replace2(createReplacement(Location2, 1, "f"));
  EXPECT_TRUE(Replace2.apply(Context.Rewrite));
  EXPECT_EQ("line1\nlix\ny\nnf4", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, ApplyFailsForNonExistentLocation) {
  Replacement Replace("nonexistent-file.cpp", 0, 1, "");
  EXPECT_FALSE(Replace.apply(Context.Rewrite));
}

TEST_F(ReplacementTest, CanRetrivePath) {
  Replacement Replace("/path/to/file.cpp", 0, 1, "");
  EXPECT_EQ("/path/to/file.cpp", Replace.getFilePath());
}

TEST_F(ReplacementTest, ReturnsInvalidPath) {
  Replacement Replace1(Context.Sources, SourceLocation(), 0, "");
  EXPECT_TRUE(Replace1.getFilePath().empty());

  Replacement Replace2;
  EXPECT_TRUE(Replace2.getFilePath().empty());
}

// Checks that an llvm::Error instance contains a ReplacementError with expected
// error code, expected new replacement, and expected existing replacement.
static bool checkReplacementError(llvm::Error &&Error,
                                  replacement_error ExpectedErr,
                                  llvm::Optional<Replacement> ExpectedExisting,
                                  llvm::Optional<Replacement> ExpectedNew) {
  if (!Error) {
    llvm::errs() << "Error is a success.";
    return false;
  }
  std::string ErrorMessage;
  llvm::raw_string_ostream OS(ErrorMessage);
  llvm::handleAllErrors(std::move(Error), [&](const ReplacementError &RE) {
    llvm::errs() << "Handling error...\n";
    if (ExpectedErr != RE.get())
      OS << "Unexpected error code: " << int(RE.get()) << "\n";
    if (ExpectedExisting != RE.getExistingReplacement()) {
      OS << "Expected Existing != Actual Existing.\n";
      if (ExpectedExisting.hasValue())
        OS << "Expected existing replacement: " << ExpectedExisting->toString()
           << "\n";
      if (RE.getExistingReplacement().hasValue())
        OS << "Actual existing replacement: "
           << RE.getExistingReplacement()->toString() << "\n";
    }
    if (ExpectedNew != RE.getNewReplacement()) {
      OS << "Expected New != Actual New.\n";
      if (ExpectedNew.hasValue())
        OS << "Expected new replacement: " << ExpectedNew->toString() << "\n";
      if (RE.getNewReplacement().hasValue())
        OS << "Actual new replacement: " << RE.getNewReplacement()->toString()
           << "\n";
    }
  });
  OS.flush();
  if (ErrorMessage.empty()) return true;
  llvm::errs() << ErrorMessage;
  return false;
}

TEST_F(ReplacementTest, FailAddReplacements) {
  Replacements Replaces;
  Replacement Deletion("x.cc", 0, 10, "3");
  auto Err = Replaces.add(Deletion);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Replacement OverlappingReplacement("x.cc", 0, 2, "a");
  Err = Replaces.add(OverlappingReplacement);
  EXPECT_TRUE(checkReplacementError(std::move(Err),
                                    replacement_error::overlap_conflict,
                                    Deletion, OverlappingReplacement));

  Replacement ContainedReplacement("x.cc", 2, 2, "a");
  Err = Replaces.add(Replacement(ContainedReplacement));
  EXPECT_TRUE(checkReplacementError(std::move(Err),
                                    replacement_error::overlap_conflict,
                                    Deletion, ContainedReplacement));

  Replacement WrongPathReplacement("y.cc", 20, 2, "");
  Err = Replaces.add(WrongPathReplacement);
  EXPECT_TRUE(checkReplacementError(std::move(Err),
                                    replacement_error::wrong_file_path,
                                    Deletion, WrongPathReplacement));

  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(Deletion, *Replaces.begin());
}

TEST_F(ReplacementTest, DeletionInReplacements) {
  Replacements Replaces;
  Replacement R("x.cc", 0, 10, "3");
  auto Err = Replaces.add(R);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 0, 2, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 2, 2, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(R, *Replaces.begin());
}

TEST_F(ReplacementTest, OverlappingReplacements) {
  Replacements Replaces;
  auto Err = Replaces.add(Replacement("x.cc", 0, 3, "345"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 2, 3, "543"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(Replacement("x.cc", 0, 5, "34543"), *Replaces.begin());

  Err = Replaces.add(Replacement("x.cc", 2, 1, "5"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(Replacement("x.cc", 0, 5, "34543"), *Replaces.begin());
}

TEST_F(ReplacementTest, AddAdjacentInsertionAndReplacement) {
  Replacements Replaces;
  // Test adding an insertion at the offset of an existing replacement.
  auto Err = Replaces.add(Replacement("x.cc", 10, 3, "replace"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, "insert"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(Replaces.size(), 2u);

  Replaces.clear();
  // Test overlap with an existing insertion.
  Err = Replaces.add(Replacement("x.cc", 10, 0, "insert"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 3, "replace"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(Replaces.size(), 2u);
}

TEST_F(ReplacementTest, MergeNewDeletions) {
  Replacements Replaces;
  Replacement ContainingReplacement("x.cc", 0, 10, "");
  auto Err = Replaces.add(ContainingReplacement);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement("x.cc", 5, 3, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement("x.cc", 0, 10, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement("x.cc", 5, 5, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(*Replaces.begin(), ContainingReplacement);
}

TEST_F(ReplacementTest, MergeOverlappingButNotAdjacentReplacement) {
  Replacements Replaces;
  auto Err = Replaces.add(Replacement("x.cc", 0, 2, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement("x.cc", 5, 5, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Replacement After = Replacement("x.cc", 10, 5, "");
  Err = Replaces.add(After);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Replacement ContainingReplacement("x.cc", 0, 10, "");
  Err = Replaces.add(ContainingReplacement);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(2u, Replaces.size());
  EXPECT_EQ(*Replaces.begin(), ContainingReplacement);
  EXPECT_EQ(*(++Replaces.begin()), After);
}

TEST_F(ReplacementTest, InsertionBeforeMergedDeletions) {
  Replacements Replaces;

  Replacement Insertion("x.cc", 0, 0, "123");
  auto Err = Replaces.add(Insertion);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement("x.cc", 5, 5, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Replacement Deletion("x.cc", 0, 10, "");
  Err = Replaces.add(Deletion);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(2u, Replaces.size());
  EXPECT_EQ(*Replaces.begin(), Insertion);
  EXPECT_EQ(*(++Replaces.begin()), Deletion);
}

TEST_F(ReplacementTest, MergeOverlappingDeletions) {
  Replacements Replaces;
  auto Err = Replaces.add(Replacement("x.cc", 0, 2, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement("x.cc", 0, 5, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(Replacement("x.cc", 0, 5, ""), *Replaces.begin());

  Err = Replaces.add(Replacement("x.cc", 1, 5, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(Replacement("x.cc", 0, 6, ""), *Replaces.begin());
}

TEST_F(ReplacementTest, FailedMergeExistingDeletions) {
  Replacements Replaces;
  Replacement First("x.cc", 0, 2, "");
  auto Err = Replaces.add(First);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Replacement Second("x.cc", 5, 5, "");
  Err = Replaces.add(Second);
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement("x.cc", 1, 10, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(Replacement("x.cc", 0, 11, ""), *Replaces.begin());
}

TEST_F(ReplacementTest, FailAddRegression) {
  Replacements Replaces;
  // Create two replacements, where the second one is an insertion of the empty
  // string exactly at the end of the first one.
  auto Err = Replaces.add(Replacement("x.cc", 0, 10, "1"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  // Make sure we find the overlap with the first entry when inserting a
  // replacement that ends exactly at the seam of the existing replacements.
  Replacement OverlappingReplacement("x.cc", 5, 5, "fail");
  Err = Replaces.add(OverlappingReplacement);
  EXPECT_TRUE(checkReplacementError(std::move(Err),
                                    replacement_error::overlap_conflict,
                                    *Replaces.begin(), OverlappingReplacement));

  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
}

TEST_F(ReplacementTest, InsertAtOffsetOfReplacement) {
  Replacements Replaces;
  auto Err = Replaces.add(Replacement("x.cc", 10, 2, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(Replaces.size(), 2u);

  Replaces.clear();
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 2, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(Replaces.size(), 2u);
}

TEST_F(ReplacementTest, AddInsertAtOtherInsertWhenOderIndependent) {
  Replacements Replaces;
  auto Err = Replaces.add(Replacement("x.cc", 10, 0, "a"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Replacement ConflictInsertion("x.cc", 10, 0, "b");
  Err = Replaces.add(ConflictInsertion);
  EXPECT_TRUE(checkReplacementError(std::move(Err),
                                    replacement_error::insert_conflict,
                                    *Replaces.begin(), ConflictInsertion));

  Replaces.clear();
  Err = Replaces.add(Replacement("x.cc", 10, 0, "a"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, "aa"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(1u, Replaces.size());
  EXPECT_EQ(Replacement("x.cc", 10, 0, "aaa"), *Replaces.begin());

  Replaces.clear();
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 3, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(2u, Replaces.size());
  EXPECT_EQ(Replacement("x.cc", 10, 0, ""), *Replaces.begin());
  EXPECT_EQ(Replacement("x.cc", 10, 3, ""), *std::next(Replaces.begin()));
}

TEST_F(ReplacementTest, InsertBetweenAdjacentReplacements) {
  Replacements Replaces;
  auto Err = Replaces.add(Replacement("x.cc", 10, 5, "a"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 8, 2, "a"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, "b"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
}

TEST_F(ReplacementTest, CanApplyReplacements) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  Replacements Replaces =
      toReplacements({Replacement(Context.Sources,
                                  Context.getLocation(ID, 2, 1), 5, "replaced"),
                      Replacement(Context.Sources,
                                  Context.getLocation(ID, 3, 1), 5, "other")});
  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("line1\nreplaced\nother\nline4", Context.getRewrittenText(ID));
}

// Verifies that replacement/deletion is applied before insertion at the same
// offset.
TEST_F(ReplacementTest, InsertAndDelete) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  Replacements Replaces = toReplacements(
      {Replacement(Context.Sources, Context.getLocation(ID, 2, 1), 6, ""),
       Replacement(Context.Sources, Context.getLocation(ID, 2, 1), 0,
                   "other\n")});
  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("line1\nother\nline3\nline4", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, AdjacentReplacements) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "ab");
  Replacements Replaces = toReplacements(
      {Replacement(Context.Sources, Context.getLocation(ID, 1, 1), 1, "x"),
       Replacement(Context.Sources, Context.getLocation(ID, 1, 2), 1, "y")});
  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("xy", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, AddDuplicateReplacements) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  auto Replaces = toReplacements({Replacement(
      Context.Sources, Context.getLocation(ID, 2, 1), 5, "replaced")});

  auto Err = Replaces.add(Replacement(
      Context.Sources, Context.getLocation(ID, 2, 1), 5, "replaced"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement(Context.Sources, Context.getLocation(ID, 2, 1),
                                 5, "replaced"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));

  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("line1\nreplaced\nline3\nline4", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, FailOrderDependentReplacements) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  auto Replaces = toReplacements({Replacement(
      Context.Sources, Context.getLocation(ID, 2, 1), 5, "other")});

  Replacement ConflictReplacement(Context.Sources,
                                  Context.getLocation(ID, 2, 1), 5, "rehto");
  auto Err = Replaces.add(ConflictReplacement);
  EXPECT_TRUE(checkReplacementError(std::move(Err),
                                    replacement_error::overlap_conflict,
                                    *Replaces.begin(), ConflictReplacement));

  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("line1\nother\nline3\nline4", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, InvalidSourceLocationFailsApplyAll) {
  Replacements Replaces =
      toReplacements({Replacement(Context.Sources, SourceLocation(), 5, "2")});

  EXPECT_FALSE(applyAllReplacements(Replaces, Context.Rewrite));
}

TEST_F(ReplacementTest, MultipleFilesReplaceAndFormat) {
  // Column limit is 20.
  std::string Code1 = "Long *a =\n"
                      "    new Long();\n"
                      "long x = 1;";
  std::string Expected1 = "auto a = new Long();\n"
                          "long x =\n"
                          "    12345678901;";
  std::string Code2 = "int x = 123;\n"
                      "int y = 0;";
  std::string Expected2 = "int x =\n"
                          "    1234567890123;\n"
                          "int y = 10;";
  StringRef File1 = "format_1.cpp";
  StringRef File2 = "format_2.cpp";
  FileID ID1 = Context.createInMemoryFile(File1, Code1);
  FileID ID2 = Context.createInMemoryFile(File2, Code2);

  // Scrambled the order of replacements.
  std::map<std::string, Replacements> FileToReplaces;
  FileToReplaces[std::string(File1)] = toReplacements(
      {tooling::Replacement(Context.Sources, Context.getLocation(ID1, 1, 1), 6,
                            "auto "),
       tooling::Replacement(Context.Sources, Context.getLocation(ID1, 3, 10), 1,
                            "12345678901")});
  FileToReplaces[std::string(File2)] = toReplacements(
      {tooling::Replacement(Context.Sources, Context.getLocation(ID2, 1, 12), 0,
                            "4567890123"),
       tooling::Replacement(Context.Sources, Context.getLocation(ID2, 2, 9), 1,
                            "10")});
  EXPECT_TRUE(
      formatAndApplyAllReplacements(FileToReplaces, Context.Rewrite,
                                    "{BasedOnStyle: LLVM, ColumnLimit: 20}"));
  EXPECT_EQ(Expected1, Context.getRewrittenText(ID1));
  EXPECT_EQ(Expected2, Context.getRewrittenText(ID2));
}

TEST(ShiftedCodePositionTest, FindsNewCodePosition) {
  Replacements Replaces =
      toReplacements({Replacement("", 0, 1, ""), Replacement("", 4, 3, " ")});
  // Assume ' int   i;' is turned into 'int i;' and cursor is located at '|'.
  EXPECT_EQ(0u, Replaces.getShiftedCodePosition(0)); // |int   i;
  EXPECT_EQ(0u, Replaces.getShiftedCodePosition(1)); //  |nt   i;
  EXPECT_EQ(1u, Replaces.getShiftedCodePosition(2)); //  i|t   i;
  EXPECT_EQ(2u, Replaces.getShiftedCodePosition(3)); //  in|   i;
  EXPECT_EQ(3u, Replaces.getShiftedCodePosition(4)); //  int|  i;
  EXPECT_EQ(3u, Replaces.getShiftedCodePosition(5)); //  int | i;
  EXPECT_EQ(3u, Replaces.getShiftedCodePosition(6)); //  int  |i;
  EXPECT_EQ(4u, Replaces.getShiftedCodePosition(7)); //  int   |;
  EXPECT_EQ(5u, Replaces.getShiftedCodePosition(8)); //  int   i|
}

TEST(ShiftedCodePositionTest, FindsNewCodePositionWithInserts) {
  Replacements Replaces = toReplacements({Replacement("", 4, 0, "\"\n\"")});
  // Assume '"12345678"' is turned into '"1234"\n"5678"'.
  EXPECT_EQ(3u, Replaces.getShiftedCodePosition(3)); // "123|5678"
  EXPECT_EQ(7u, Replaces.getShiftedCodePosition(4)); // "1234|678"
  EXPECT_EQ(8u, Replaces.getShiftedCodePosition(5)); // "12345|78"
}

TEST(ShiftedCodePositionTest, FindsNewCodePositionInReplacedText) {
  // Replace the first four characters with "abcd".
  auto Replaces = toReplacements({Replacement("", 0, 4, "abcd")});
  for (unsigned i = 0; i < 3; ++i)
    EXPECT_EQ(i, Replaces.getShiftedCodePosition(i));
}

TEST(ShiftedCodePositionTest, NoReplacementText) {
  Replacements Replaces = toReplacements({Replacement("", 0, 42, "")});
  EXPECT_EQ(0u, Replaces.getShiftedCodePosition(0));
  EXPECT_EQ(0u, Replaces.getShiftedCodePosition(39));
  EXPECT_EQ(3u, Replaces.getShiftedCodePosition(45));
  EXPECT_EQ(0u, Replaces.getShiftedCodePosition(42));
}

class FlushRewrittenFilesTest : public ::testing::Test {
public:
   FlushRewrittenFilesTest() {}

   ~FlushRewrittenFilesTest() override {
    for (llvm::StringMap<std::string>::iterator I = TemporaryFiles.begin(),
                                                E = TemporaryFiles.end();
         I != E; ++I) {
      llvm::StringRef Name = I->second;
      std::error_code EC = llvm::sys::fs::remove(Name);
      (void)EC;
      assert(!EC);
    }
  }

  FileID createFile(llvm::StringRef Name, llvm::StringRef Content) {
    SmallString<1024> Path;
    int FD;
    std::error_code EC = llvm::sys::fs::createTemporaryFile(Name, "", FD, Path);
    assert(!EC);
    (void)EC;

    llvm::raw_fd_ostream OutStream(FD, true);
    OutStream << Content;
    OutStream.close();
    auto File = Context.Files.getOptionalFileRef(Path);
    assert(File);

    StringRef Found =
        TemporaryFiles.insert(std::make_pair(Name, std::string(Path.str())))
            .first->second;
    assert(Found == Path);
    (void)Found;
    return Context.Sources.createFileID(*File, SourceLocation(),
                                        SrcMgr::C_User);
  }

  std::string getFileContentFromDisk(llvm::StringRef Name) {
    std::string Path = TemporaryFiles.lookup(Name);
    assert(!Path.empty());
    // We need to read directly from the FileManager without relaying through
    // a FileEntry, as otherwise we'd read through an already opened file
    // descriptor, which might not see the changes made.
    // FIXME: Figure out whether there is a way to get the SourceManger to
    // reopen the file.
    auto FileBuffer = Context.Files.getBufferForFile(Path);
    return std::string((*FileBuffer)->getBuffer());
  }

  llvm::StringMap<std::string> TemporaryFiles;
  RewriterTestContext Context;
};

TEST_F(FlushRewrittenFilesTest, StoresChangesOnDisk) {
  FileID ID = createFile("input.cpp", "line1\nline2\nline3\nline4");
  Replacements Replaces = toReplacements({Replacement(
      Context.Sources, Context.getLocation(ID, 2, 1), 5, "replaced")});
  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_FALSE(Context.Rewrite.overwriteChangedFiles());
  EXPECT_EQ("line1\nreplaced\nline3\nline4",
            getFileContentFromDisk("input.cpp"));
}

namespace {
template <typename T>
class TestVisitor : public clang::RecursiveASTVisitor<T> {
public:
  bool runOver(StringRef Code) {
    return runToolOnCode(std::make_unique<TestAction>(this), Code);
  }

protected:
  clang::SourceManager *SM;
  clang::ASTContext *Context;

private:
  class FindConsumer : public clang::ASTConsumer {
  public:
    FindConsumer(TestVisitor *Visitor) : Visitor(Visitor) {}

    void HandleTranslationUnit(clang::ASTContext &Context) override {
      Visitor->TraverseDecl(Context.getTranslationUnitDecl());
    }

  private:
    TestVisitor *Visitor;
  };

  class TestAction : public clang::ASTFrontendAction {
  public:
    TestAction(TestVisitor *Visitor) : Visitor(Visitor) {}

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &compiler,
                      llvm::StringRef dummy) override {
      Visitor->SM = &compiler.getSourceManager();
      Visitor->Context = &compiler.getASTContext();
      /// TestConsumer will be deleted by the framework calling us.
      return std::make_unique<FindConsumer>(Visitor);
    }

  private:
    TestVisitor *Visitor;
  };
};
} // end namespace

void expectReplacementAt(const Replacement &Replace,
                         StringRef File, unsigned Offset, unsigned Length) {
  ASSERT_TRUE(Replace.isApplicable());
  EXPECT_EQ(File, Replace.getFilePath());
  EXPECT_EQ(Offset, Replace.getOffset());
  EXPECT_EQ(Length, Replace.getLength());
}

class ClassDeclXVisitor : public TestVisitor<ClassDeclXVisitor> {
public:
  bool VisitCXXRecordDecl(CXXRecordDecl *Record) {
    if (Record->getName() == "X") {
      Replace = Replacement(*SM, Record, "");
    }
    return true;
  }
  Replacement Replace;
};

TEST(Replacement, CanBeConstructedFromNode) {
  ClassDeclXVisitor ClassDeclX;
  EXPECT_TRUE(ClassDeclX.runOver("     class X;"));
  expectReplacementAt(ClassDeclX.Replace, "input.cc", 5, 7);
}

TEST(Replacement, ReplacesAtSpellingLocation) {
  ClassDeclXVisitor ClassDeclX;
  EXPECT_TRUE(ClassDeclX.runOver("#define A(Y) Y\nA(class X);"));
  expectReplacementAt(ClassDeclX.Replace, "input.cc", 17, 7);
}

class CallToFVisitor : public TestVisitor<CallToFVisitor> {
public:
  bool VisitCallExpr(CallExpr *Call) {
    if (Call->getDirectCallee()->getName() == "F") {
      Replace = Replacement(*SM, Call, "");
    }
    return true;
  }
  Replacement Replace;
};

TEST(Replacement, FunctionCall) {
  CallToFVisitor CallToF;
  EXPECT_TRUE(CallToF.runOver("void F(); void G() { F(); }"));
  expectReplacementAt(CallToF.Replace, "input.cc", 21, 3);
}

TEST(Replacement, TemplatedFunctionCall) {
  CallToFVisitor CallToF;
  EXPECT_TRUE(CallToF.runOver(
        "template <typename T> void F(); void G() { F<int>(); }"));
  expectReplacementAt(CallToF.Replace, "input.cc", 43, 8);
}

class NestedNameSpecifierAVisitor
    : public TestVisitor<NestedNameSpecifierAVisitor> {
public:
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNSLoc) {
    if (NNSLoc.getNestedNameSpecifier()) {
      if (const NamespaceDecl* NS = NNSLoc.getNestedNameSpecifier()->getAsNamespace()) {
        if (NS->getName() == "a") {
          Replace = Replacement(*SM, &NNSLoc, "", Context->getLangOpts());
        }
      }
    }
    return TestVisitor<NestedNameSpecifierAVisitor>::TraverseNestedNameSpecifierLoc(
        NNSLoc);
  }
  Replacement Replace;
};

TEST(Replacement, ColonColon) {
  NestedNameSpecifierAVisitor VisitNNSA;
  EXPECT_TRUE(VisitNNSA.runOver("namespace a { void f() { ::a::f(); } }"));
  expectReplacementAt(VisitNNSA.Replace, "input.cc", 25, 5);
}

TEST(Range, overlaps) {
  EXPECT_TRUE(Range(10, 10).overlapsWith(Range(0, 11)));
  EXPECT_TRUE(Range(0, 11).overlapsWith(Range(10, 10)));
  EXPECT_FALSE(Range(10, 10).overlapsWith(Range(0, 10)));
  EXPECT_FALSE(Range(0, 10).overlapsWith(Range(10, 10)));
  EXPECT_TRUE(Range(0, 10).overlapsWith(Range(2, 6)));
  EXPECT_TRUE(Range(2, 6).overlapsWith(Range(0, 10)));
}

TEST(Range, contains) {
  EXPECT_TRUE(Range(0, 10).contains(Range(0, 10)));
  EXPECT_TRUE(Range(0, 10).contains(Range(2, 6)));
  EXPECT_FALSE(Range(2, 6).contains(Range(0, 10)));
  EXPECT_FALSE(Range(0, 10).contains(Range(0, 11)));
}

TEST(Range, CalculateRangesOfReplacements) {
  // Before: aaaabbbbbbz
  // After : bbbbbbzzzzzzoooooooooooooooo
  Replacements Replaces = toReplacements(
      {Replacement("foo", 0, 4, ""), Replacement("foo", 10, 1, "zzzzzz"),
       Replacement("foo", 11, 0, "oooooooooooooooo")});

  std::vector<Range> Ranges = Replaces.getAffectedRanges();

  EXPECT_EQ(2ul, Ranges.size());
  EXPECT_TRUE(Ranges[0].getOffset() == 0);
  EXPECT_TRUE(Ranges[0].getLength() == 0);
  EXPECT_TRUE(Ranges[1].getOffset() == 6);
  EXPECT_TRUE(Ranges[1].getLength() == 22);
}

TEST(Range, CalculateRangesOfInsertionAroundReplacement) {
  Replacements Replaces = toReplacements(
      {Replacement("foo", 0, 2, ""), Replacement("foo", 0, 0, "ba")});

  std::vector<Range> Ranges = Replaces.getAffectedRanges();

  EXPECT_EQ(1ul, Ranges.size());
  EXPECT_EQ(0u, Ranges[0].getOffset());
  EXPECT_EQ(2u, Ranges[0].getLength());
}

TEST(Range, RangesAfterEmptyReplacements) {
  std::vector<Range> Ranges = {Range(5, 6), Range(10, 5)};
  Replacements Replaces;
  std::vector<Range> Expected = {Range(5, 10)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, RangesAfterReplacements) {
  std::vector<Range> Ranges = {Range(5, 2), Range(10, 5)};
  Replacements Replaces = toReplacements({Replacement("foo", 0, 2, "1234")});
  std::vector<Range> Expected = {Range(0, 4), Range(7, 2), Range(12, 5)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, RangesBeforeReplacements) {
  std::vector<Range> Ranges = {Range(5, 2), Range(10, 5)};
  Replacements Replaces = toReplacements({Replacement("foo", 20, 2, "1234")});
  std::vector<Range> Expected = {Range(5, 2), Range(10, 5), Range(20, 4)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, NotAffectedByReplacements) {
  std::vector<Range> Ranges = {Range(0, 2), Range(5, 2), Range(10, 5)};
  Replacements Replaces = toReplacements({Replacement("foo", 3, 2, "12"),
                                          Replacement("foo", 12, 2, "12"),
                                          Replacement("foo", 20, 5, "")});
  std::vector<Range> Expected = {Range(0, 2), Range(3, 4), Range(10, 5),
                                 Range(20, 0)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, RangesWithNonOverlappingReplacements) {
  std::vector<Range> Ranges = {Range(0, 2), Range(5, 2), Range(10, 5)};
  Replacements Replaces = toReplacements({Replacement("foo", 3, 1, ""),
                                          Replacement("foo", 6, 1, "123"),
                                          Replacement("foo", 20, 2, "12345")});
  std::vector<Range> Expected = {Range(0, 2), Range(3, 0), Range(4, 4),
                                 Range(11, 5), Range(21, 5)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, RangesWithOverlappingReplacements) {
  std::vector<Range> Ranges = {Range(0, 2), Range(5, 2), Range(15, 5),
                               Range(30, 5)};
  Replacements Replaces = toReplacements(
      {Replacement("foo", 1, 3, ""), Replacement("foo", 6, 1, "123"),
       Replacement("foo", 13, 3, "1"), Replacement("foo", 25, 15, "")});
  std::vector<Range> Expected = {Range(0, 1), Range(2, 4), Range(12, 5),
                                 Range(22, 0)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, MergeIntoOneRange) {
  std::vector<Range> Ranges = {Range(0, 2), Range(5, 2), Range(15, 5)};
  Replacements Replaces =
      toReplacements({Replacement("foo", 1, 15, "1234567890")});
  std::vector<Range> Expected = {Range(0, 15)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, ReplacementsStartingAtRangeOffsets) {
  std::vector<Range> Ranges = {Range(0, 2), Range(5, 5), Range(15, 5)};
  Replacements Replaces = toReplacements(
      {Replacement("foo", 0, 2, "12"), Replacement("foo", 5, 1, "123"),
       Replacement("foo", 7, 4, "12345"), Replacement("foo", 15, 10, "12")});
  std::vector<Range> Expected = {Range(0, 2), Range(5, 9), Range(18, 2)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, ReplacementsEndingAtRangeEnds) {
  std::vector<Range> Ranges = {Range(0, 2), Range(5, 2), Range(15, 5)};
  Replacements Replaces = toReplacements(
      {Replacement("foo", 6, 1, "123"), Replacement("foo", 17, 3, "12")});
  std::vector<Range> Expected = {Range(0, 2), Range(5, 4), Range(17, 4)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, AjacentReplacements) {
  std::vector<Range> Ranges = {Range(0, 0), Range(15, 5)};
  Replacements Replaces = toReplacements(
      {Replacement("foo", 1, 2, "123"), Replacement("foo", 12, 3, "1234")});
  std::vector<Range> Expected = {Range(0, 0), Range(1, 3), Range(13, 9)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, MergeRangesAfterReplacements) {
  std::vector<Range> Ranges = {Range(8, 0), Range(5, 2), Range(9, 0), Range(0, 1)};
  Replacements Replaces = toReplacements({Replacement("foo", 1, 3, ""),
                                          Replacement("foo", 7, 0, "12"),
                                          Replacement("foo", 9, 2, "")});
  std::vector<Range> Expected = {Range(0, 1), Range(2, 4), Range(7, 0),
                                 Range(8, 0)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

TEST(Range, ConflictingRangesBeforeReplacements) {
  std::vector<Range> Ranges = {Range(8, 3), Range(5, 4), Range(9, 1)};
  Replacements Replaces = toReplacements({Replacement("foo", 1, 3, "")});
  std::vector<Range> Expected = {Range(1, 0), Range(2, 6)};
  EXPECT_EQ(Expected, calculateRangesAfterReplacements(Replaces, Ranges));
}

class MergeReplacementsTest : public ::testing::Test {
protected:
  void mergeAndTestRewrite(StringRef Code, StringRef Intermediate,
                           StringRef Result, const Replacements &First,
                           const Replacements &Second) {
    // These are mainly to verify the test itself and make it easier to read.
    auto AfterFirst = applyAllReplacements(Code, First);
    EXPECT_TRUE(static_cast<bool>(AfterFirst));
    auto InSequenceRewrite = applyAllReplacements(*AfterFirst, Second);
    EXPECT_TRUE(static_cast<bool>(InSequenceRewrite));
    EXPECT_EQ(Intermediate, *AfterFirst);
    EXPECT_EQ(Result, *InSequenceRewrite);

    tooling::Replacements Merged = First.merge(Second);
    auto MergedRewrite = applyAllReplacements(Code, Merged);
    EXPECT_TRUE(static_cast<bool>(MergedRewrite));
    EXPECT_EQ(*InSequenceRewrite, *MergedRewrite);
    if (*InSequenceRewrite != *MergedRewrite)
      for (tooling::Replacement M : Merged)
        llvm::errs() << M.getOffset() << " " << M.getLength() << " "
                     << M.getReplacementText() << "\n";
  }
  void mergeAndTestRewrite(StringRef Code, const Replacements &First,
                           const Replacements &Second) {
    auto AfterFirst = applyAllReplacements(Code, First);
    EXPECT_TRUE(static_cast<bool>(AfterFirst));
    auto InSequenceRewrite = applyAllReplacements(*AfterFirst, Second);
    tooling::Replacements Merged = First.merge(Second);
    auto MergedRewrite = applyAllReplacements(Code, Merged);
    EXPECT_TRUE(static_cast<bool>(MergedRewrite));
    EXPECT_EQ(*InSequenceRewrite, *MergedRewrite);
    if (*InSequenceRewrite != *MergedRewrite)
      for (tooling::Replacement M : Merged)
        llvm::errs() << M.getOffset() << " " << M.getLength() << " "
                     << M.getReplacementText() << "\n";
  }
};

TEST_F(MergeReplacementsTest, Offsets) {
  mergeAndTestRewrite("aaa", "aabab", "cacabab",
                      toReplacements({{"", 2, 0, "b"}, {"", 3, 0, "b"}}),
                      toReplacements({{"", 0, 0, "c"}, {"", 1, 0, "c"}}));
  mergeAndTestRewrite("aaa", "babaa", "babacac",
                      toReplacements({{"", 0, 0, "b"}, {"", 1, 0, "b"}}),
                      toReplacements({{"", 4, 0, "c"}, {"", 5, 0, "c"}}));
  mergeAndTestRewrite("aaaa", "aaa", "aac", toReplacements({{"", 1, 1, ""}}),
                      toReplacements({{"", 2, 1, "c"}}));

  mergeAndTestRewrite("aa", "bbabba", "bbabcba",
                      toReplacements({{"", 0, 0, "bb"}, {"", 1, 0, "bb"}}),
                      toReplacements({{"", 4, 0, "c"}}));
}

TEST_F(MergeReplacementsTest, Concatenations) {
  // Basic concatenations. It is important to merge these into a single
  // replacement to ensure the correct order.
  {
    auto First = toReplacements({{"", 0, 0, "a"}});
    auto Second = toReplacements({{"", 1, 0, "b"}});
    EXPECT_EQ(toReplacements({{"", 0, 0, "ab"}}), First.merge(Second));
  }
  {
    auto First = toReplacements({{"", 0, 0, "a"}});
    auto Second = toReplacements({{"", 0, 0, "b"}});
    EXPECT_EQ(toReplacements({{"", 0, 0, "ba"}}), First.merge(Second));
  }
  mergeAndTestRewrite("", "a", "ab", toReplacements({{"", 0, 0, "a"}}),
                      toReplacements({{"", 1, 0, "b"}}));
  mergeAndTestRewrite("", "a", "ba", toReplacements({{"", 0, 0, "a"}}),
                      toReplacements({{"", 0, 0, "b"}}));
}

TEST_F(MergeReplacementsTest, NotChangingLengths) {
  mergeAndTestRewrite("aaaa", "abba", "acca",
                      toReplacements({{"", 1, 2, "bb"}}),
                      toReplacements({{"", 1, 2, "cc"}}));
  mergeAndTestRewrite("aaaa", "abba", "abcc",
                      toReplacements({{"", 1, 2, "bb"}}),
                      toReplacements({{"", 2, 2, "cc"}}));
  mergeAndTestRewrite("aaaa", "abba", "ccba",
                      toReplacements({{"", 1, 2, "bb"}}),
                      toReplacements({{"", 0, 2, "cc"}}));
  mergeAndTestRewrite("aaaaaa", "abbdda", "abccda",
                      toReplacements({{"", 1, 2, "bb"}, {"", 3, 2, "dd"}}),
                      toReplacements({{"", 2, 2, "cc"}}));
}

TEST_F(MergeReplacementsTest, OverlappingRanges) {
  mergeAndTestRewrite("aaa", "bbd", "bcbcd",
                      toReplacements({{"", 0, 1, "bb"}, {"", 1, 2, "d"}}),
                      toReplacements({{"", 1, 0, "c"}, {"", 2, 0, "c"}}));

  mergeAndTestRewrite("aaaa", "aabbaa", "acccca",
                      toReplacements({{"", 2, 0, "bb"}}),
                      toReplacements({{"", 1, 4, "cccc"}}));
  mergeAndTestRewrite("aaaa", "aababa", "acccca",
                      toReplacements({{"", 2, 0, "b"}, {"", 3, 0, "b"}}),
                      toReplacements({{"", 1, 4, "cccc"}}));
  mergeAndTestRewrite("aaaaaa", "abbbba", "abba",
                      toReplacements({{"", 1, 4, "bbbb"}}),
                      toReplacements({{"", 2, 2, ""}}));
  mergeAndTestRewrite("aaaa", "aa", "cc",
                      toReplacements({{"", 1, 1, ""}, {"", 2, 1, ""}}),
                      toReplacements({{"", 0, 2, "cc"}}));
  mergeAndTestRewrite("aa", "abbba", "abcbcba",
                      toReplacements({{"", 1, 0, "bbb"}}),
                      toReplacements({{"", 2, 0, "c"}, {"", 3, 0, "c"}}));

  mergeAndTestRewrite(
      "aaa", "abbab", "ccdd",
      toReplacements({{"", 0, 1, ""}, {"", 2, 0, "bb"}, {"", 3, 0, "b"}}),
      toReplacements({{"", 0, 2, "cc"}, {"", 2, 3, "dd"}}));
  mergeAndTestRewrite(
      "aa", "babbab", "ccdd",
      toReplacements({{"", 0, 0, "b"}, {"", 1, 0, "bb"}, {"", 2, 0, "b"}}),
      toReplacements({{"", 0, 3, "cc"}, {"", 3, 3, "dd"}}));
}

TEST(DeduplicateByFileTest, PathsWithDots) {
  std::map<std::string, Replacements> FileToReplaces;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS(
      new llvm::vfs::InMemoryFileSystem());
  FileManager FileMgr(FileSystemOptions(), VFS);
#if !defined(_WIN32)
  StringRef Path1 = "a/b/.././c.h";
  StringRef Path2 = "a/c.h";
#else
  StringRef Path1 = "a\\b\\..\\.\\c.h";
  StringRef Path2 = "a\\c.h";
#endif
  EXPECT_TRUE(VFS->addFile(Path1, 0, llvm::MemoryBuffer::getMemBuffer("")));
  EXPECT_TRUE(VFS->addFile(Path2, 0, llvm::MemoryBuffer::getMemBuffer("")));
  FileToReplaces[std::string(Path1)] = Replacements();
  FileToReplaces[std::string(Path2)] = Replacements();
  FileToReplaces = groupReplacementsByFile(FileMgr, FileToReplaces);
  EXPECT_EQ(1u, FileToReplaces.size());
  EXPECT_EQ(Path1, FileToReplaces.begin()->first);
}

TEST(DeduplicateByFileTest, PathWithDotSlash) {
  std::map<std::string, Replacements> FileToReplaces;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS(
      new llvm::vfs::InMemoryFileSystem());
  FileManager FileMgr(FileSystemOptions(), VFS);
#if !defined(_WIN32)
  StringRef Path1 = "./a/b/c.h";
  StringRef Path2 = "a/b/c.h";
#else
  StringRef Path1 = ".\\a\\b\\c.h";
  StringRef Path2 = "a\\b\\c.h";
#endif
  EXPECT_TRUE(VFS->addFile(Path1, 0, llvm::MemoryBuffer::getMemBuffer("")));
  EXPECT_TRUE(VFS->addFile(Path2, 0, llvm::MemoryBuffer::getMemBuffer("")));
  FileToReplaces[std::string(Path1)] = Replacements();
  FileToReplaces[std::string(Path2)] = Replacements();
  FileToReplaces = groupReplacementsByFile(FileMgr, FileToReplaces);
  EXPECT_EQ(1u, FileToReplaces.size());
  EXPECT_EQ(Path1, FileToReplaces.begin()->first);
}

TEST(DeduplicateByFileTest, NonExistingFilePath) {
  std::map<std::string, Replacements> FileToReplaces;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS(
      new llvm::vfs::InMemoryFileSystem());
  FileManager FileMgr(FileSystemOptions(), VFS);
#if !defined(_WIN32)
  StringRef Path1 = "./a/b/c.h";
  StringRef Path2 = "a/b/c.h";
#else
  StringRef Path1 = ".\\a\\b\\c.h";
  StringRef Path2 = "a\\b\\c.h";
#endif
  FileToReplaces[std::string(Path1)] = Replacements();
  FileToReplaces[std::string(Path2)] = Replacements();
  FileToReplaces = groupReplacementsByFile(FileMgr, FileToReplaces);
  EXPECT_TRUE(FileToReplaces.empty());
}

class AtomicChangeTest : public ::testing::Test {
  protected:
    void SetUp() override {
      DefaultFileID = Context.createInMemoryFile("input.cpp", DefaultCode);
      DefaultLoc = Context.Sources.getLocForStartOfFile(DefaultFileID)
                       .getLocWithOffset(20);
      assert(DefaultLoc.isValid() && "Default location must be valid.");
    }

    RewriterTestContext Context;
    std::string DefaultCode = std::string(100, 'a');
    unsigned DefaultOffset = 20;
    SourceLocation DefaultLoc;
    FileID DefaultFileID;
};

TEST_F(AtomicChangeTest, AtomicChangeToYAML) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  llvm::Error Err =
      Change.insert(Context.Sources, DefaultLoc, "aa", /*InsertAfter=*/false);
  ASSERT_TRUE(!Err);
  Err = Change.insert(Context.Sources, DefaultLoc.getLocWithOffset(10), "bb",
                    /*InsertAfter=*/false);
  ASSERT_TRUE(!Err);
  Change.addHeader("a.h");
  Change.removeHeader("b.h");
  std::string YAMLString = Change.toYAMLString();

  // NOTE: If this test starts to fail for no obvious reason, check whitespace.
  ASSERT_STREQ("---\n"
               "Key:             'input.cpp:20'\n"
               "FilePath:        input.cpp\n"
               "Error:           ''\n"
               "InsertedHeaders:\n"
               "  - a.h\n"
               "RemovedHeaders:\n"
               "  - b.h\n"
               "Replacements:\n"
               "  - FilePath:        input.cpp\n"
               "    Offset:          20\n"
               "    Length:          0\n"
               "    ReplacementText: aa\n"
               "  - FilePath:        input.cpp\n"
               "    Offset:          30\n"
               "    Length:          0\n"
               "    ReplacementText: bb\n"
               "...\n",
               YAMLString.c_str());
}

TEST_F(AtomicChangeTest, YAMLToAtomicChange) {
  std::string YamlContent = "---\n"
                            "Key:             'input.cpp:20'\n"
                            "FilePath:        input.cpp\n"
                            "Error:           'ok'\n"
                            "InsertedHeaders:\n"
                            "  - a.h\n"
                            "RemovedHeaders:\n"
                            "  - b.h\n"
                            "Replacements:\n"
                            "  - FilePath:        input.cpp\n"
                            "    Offset:          20\n"
                            "    Length:          0\n"
                            "    ReplacementText: aa\n"
                            "  - FilePath:        input.cpp\n"
                            "    Offset:          30\n"
                            "    Length:          0\n"
                            "    ReplacementText: bb\n"
                            "...\n";
  AtomicChange ExpectedChange(Context.Sources, DefaultLoc);
  llvm::Error Err = ExpectedChange.insert(Context.Sources, DefaultLoc, "aa",
                                        /*InsertAfter=*/false);
  ASSERT_TRUE(!Err);
  Err = ExpectedChange.insert(Context.Sources, DefaultLoc.getLocWithOffset(10),
                            "bb", /*InsertAfter=*/false);
  ASSERT_TRUE(!Err);

  ExpectedChange.addHeader("a.h");
  ExpectedChange.removeHeader("b.h");
  ExpectedChange.setError("ok");

  AtomicChange ActualChange = AtomicChange::convertFromYAML(YamlContent);
  EXPECT_EQ(ExpectedChange.getKey(), ActualChange.getKey());
  EXPECT_EQ(ExpectedChange.getFilePath(), ActualChange.getFilePath());
  EXPECT_EQ(ExpectedChange.getError(), ActualChange.getError());
  EXPECT_EQ(ExpectedChange.getInsertedHeaders(),
            ActualChange.getInsertedHeaders());
  EXPECT_EQ(ExpectedChange.getRemovedHeaders(),
            ActualChange.getRemovedHeaders());
  EXPECT_EQ(ExpectedChange.getReplacements().size(),
            ActualChange.getReplacements().size());
  EXPECT_EQ(2u, ActualChange.getReplacements().size());
  EXPECT_EQ(*ExpectedChange.getReplacements().begin(),
            *ActualChange.getReplacements().begin());
  EXPECT_EQ(*(++ExpectedChange.getReplacements().begin()),
            *(++ActualChange.getReplacements().begin()));
}

TEST_F(AtomicChangeTest, CheckKeyAndKeyFile) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  EXPECT_EQ("input.cpp:20", Change.getKey());
  EXPECT_EQ("input.cpp", Change.getFilePath());
}

TEST_F(AtomicChangeTest, Replace) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  llvm::Error Err = Change.replace(Context.Sources, DefaultLoc, 2, "aa");
  ASSERT_TRUE(!Err);
  EXPECT_EQ(Change.getReplacements().size(), 1u);
  EXPECT_EQ(*Change.getReplacements().begin(),
            Replacement(Context.Sources, DefaultLoc, 2, "aa"));

  // Add a new replacement that conflicts with the existing one.
  Err = Change.replace(Context.Sources, DefaultLoc, 3, "ab");
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(Change.getReplacements().size(), 1u);
}

TEST_F(AtomicChangeTest, ReplaceWithRange) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  SourceLocation End = DefaultLoc.getLocWithOffset(20);
  llvm::Error Err = Change.replace(
      Context.Sources, CharSourceRange::getCharRange(DefaultLoc, End), "aa");
  ASSERT_TRUE(!Err);
  EXPECT_EQ(Change.getReplacements().size(), 1u);
  EXPECT_EQ(*Change.getReplacements().begin(),
            Replacement(Context.Sources, DefaultLoc, 20, "aa"));
}

TEST_F(AtomicChangeTest, InsertBefore) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  llvm::Error Err = Change.insert(Context.Sources, DefaultLoc, "aa");
  ASSERT_TRUE(!Err);
  EXPECT_EQ(Change.getReplacements().size(), 1u);
  EXPECT_EQ(*Change.getReplacements().begin(),
            Replacement(Context.Sources, DefaultLoc, 0, "aa"));
  Err = Change.insert(Context.Sources, DefaultLoc, "b", /*InsertAfter=*/false);
  ASSERT_TRUE(!Err);
  EXPECT_EQ(Change.getReplacements().size(), 1u);
  EXPECT_EQ(*Change.getReplacements().begin(),
            Replacement(Context.Sources, DefaultLoc, 0, "baa"));
}

TEST_F(AtomicChangeTest, InsertAfter) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  llvm::Error Err = Change.insert(Context.Sources, DefaultLoc, "aa");
  ASSERT_TRUE(!Err);
  EXPECT_EQ(Change.getReplacements().size(), 1u);
  EXPECT_EQ(*Change.getReplacements().begin(),
            Replacement(Context.Sources, DefaultLoc, 0, "aa"));
  Err = Change.insert(Context.Sources, DefaultLoc, "b");
  ASSERT_TRUE(!Err);
  EXPECT_EQ(Change.getReplacements().size(), 1u);
  EXPECT_EQ(*Change.getReplacements().begin(),
            Replacement(Context.Sources, DefaultLoc, 0, "aab"));
}

TEST_F(AtomicChangeTest, InsertBeforeWithInvalidLocation) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  llvm::Error Err =
      Change.insert(Context.Sources, DefaultLoc, "a", /*InsertAfter=*/false);
  ASSERT_TRUE(!Err);

  // Invalid location.
  Err = Change.insert(Context.Sources, SourceLocation(), "a",
                    /*InsertAfter=*/false);
  ASSERT_TRUE((bool)Err);
  EXPECT_TRUE(checkReplacementError(
      std::move(Err), replacement_error::wrong_file_path,
      Replacement(Context.Sources, DefaultLoc, 0, "a"),
      Replacement(Context.Sources, SourceLocation(), 0, "a")));
}

TEST_F(AtomicChangeTest, InsertBeforeToWrongFile) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  llvm::Error Err =
      Change.insert(Context.Sources, DefaultLoc, "a", /*InsertAfter=*/false);
  ASSERT_TRUE(!Err);

  // Inserting at a different file.
  FileID NewID = Context.createInMemoryFile("extra.cpp", DefaultCode);
  SourceLocation NewLoc = Context.Sources.getLocForStartOfFile(NewID);
  Err = Change.insert(Context.Sources, NewLoc, "b", /*InsertAfter=*/false);
  ASSERT_TRUE((bool)Err);
  EXPECT_TRUE(
      checkReplacementError(std::move(Err), replacement_error::wrong_file_path,
                            Replacement(Context.Sources, DefaultLoc, 0, "a"),
                            Replacement(Context.Sources, NewLoc, 0, "b")));
}

TEST_F(AtomicChangeTest, InsertAfterWithInvalidLocation) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  llvm::Error Err = Change.insert(Context.Sources, DefaultLoc, "a");
  ASSERT_TRUE(!Err);

  // Invalid location.
  Err = Change.insert(Context.Sources, SourceLocation(), "b");
  ASSERT_TRUE((bool)Err);
  EXPECT_TRUE(checkReplacementError(
      std::move(Err), replacement_error::wrong_file_path,
      Replacement(Context.Sources, DefaultLoc, 0, "a"),
      Replacement(Context.Sources, SourceLocation(), 0, "b")));
}

TEST_F(AtomicChangeTest, Metadata) {
  AtomicChange Change(Context.Sources, DefaultLoc, 17);
  const llvm::Any &Metadata = Change.getMetadata();
  ASSERT_TRUE(llvm::any_isa<int>(Metadata));
  EXPECT_EQ(llvm::any_cast<int>(Metadata), 17);
}

TEST_F(AtomicChangeTest, NoMetadata) {
  AtomicChange Change(Context.Sources, DefaultLoc);
  EXPECT_FALSE(Change.getMetadata().hasValue());
}

class ApplyAtomicChangesTest : public ::testing::Test {
protected:
  ApplyAtomicChangesTest() : FilePath("file.cc") {
    Spec.Cleanup = true;
    Spec.Format = ApplyChangesSpec::kAll;
    Spec.Style = format::getLLVMStyle();
  }

  ~ApplyAtomicChangesTest() override {}

  void setInput(llvm::StringRef Input) {
    Code = std::string(Input);
    FID = Context.createInMemoryFile(FilePath, Code);
  }

  SourceLocation getLoc(unsigned Offset) const {
    return Context.Sources.getLocForStartOfFile(FID).getLocWithOffset(Offset);
  }

  AtomicChange replacementToAtomicChange(llvm::StringRef Key, unsigned Offset,
                                         unsigned Length,
                                         llvm::StringRef Text) {
    AtomicChange Change(FilePath, Key);
    llvm::Error Err =
        Change.replace(Context.Sources, getLoc(Offset), Length, Text);
    EXPECT_FALSE(Err);
    return Change;
  }

  std::string rewrite(bool FailureExpected = false) {
    llvm::Expected<std::string> ChangedCode =
        applyAtomicChanges(FilePath, Code, Changes, Spec);
    EXPECT_EQ(FailureExpected, !ChangedCode);
    if (!ChangedCode) {
      llvm::errs() << "Failed to apply changes: "
                   << llvm::toString(ChangedCode.takeError()) << "\n";
      return "";
    }
    return *ChangedCode;
  }

  RewriterTestContext Context;
  FileID FID;
  ApplyChangesSpec Spec;
  std::string Code;
  std::string FilePath;
  llvm::SmallVector<AtomicChange, 8> Changes;
};

TEST_F(ApplyAtomicChangesTest, BasicRefactoring) {
  setInput("int a;");
  AtomicChange Change(FilePath, "key1");
  Changes.push_back(replacementToAtomicChange("key1", 4, 1, "b"));
  EXPECT_EQ("int b;", rewrite());
}

TEST_F(ApplyAtomicChangesTest, SeveralRefactorings) {
  setInput("int a;\n"
           "int b;");
  Changes.push_back(replacementToAtomicChange("key1", 0, 3, "float"));
  Changes.push_back(replacementToAtomicChange("key2", 4, 1, "f"));
  Changes.push_back(replacementToAtomicChange("key3", 11, 1, "g"));
  Changes.push_back(replacementToAtomicChange("key4", 7, 3, "float"));
  EXPECT_EQ("float f;\n"
            "float g;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, IgnorePathsInRefactorings) {
  setInput("int a;\n"
           "int b;");
  Changes.push_back(replacementToAtomicChange("key1", 4, 1, "aa"));

  FileID ID = Context.createInMemoryFile("AnotherFile", "12345678912345");
  Changes.emplace_back("AnotherFile", "key2");
  auto Err = Changes.back().replace(
      Context.Sources,
      Context.Sources.getLocForStartOfFile(ID).getLocWithOffset(11), 1, "bb");
  ASSERT_TRUE(!Err);
  EXPECT_EQ("int aa;\n"
            "int bb;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, AppliesDuplicateInsertions) {
  setInput("int a;");
  Changes.push_back(replacementToAtomicChange("key1", 5, 0, "b"));
  Changes.push_back(replacementToAtomicChange("key2", 5, 0, "b"));
  EXPECT_EQ("int abb;", rewrite());
}

TEST_F(ApplyAtomicChangesTest, BailsOnOverlappingRefactorings) {
  setInput("int a;");
  Changes.push_back(replacementToAtomicChange("key1", 0, 5, "float f"));
  Changes.push_back(replacementToAtomicChange("key2", 4, 1, "b"));
  EXPECT_EQ("", rewrite(/*FailureExpected=*/true));
}

TEST_F(ApplyAtomicChangesTest, BasicReformatting) {
  setInput("int  a;");
  Changes.push_back(replacementToAtomicChange("key1", 5, 1, "b"));
  EXPECT_EQ("int b;", rewrite());
}

TEST_F(ApplyAtomicChangesTest, OnlyFormatWhenViolateColumnLimits) {
  Spec.Format = ApplyChangesSpec::kViolations;
  Spec.Style.ColumnLimit = 8;
  setInput("int  a;\n"
           "int    a;\n"
           "int  aaaaaaaa;\n");
  Changes.push_back(replacementToAtomicChange("key1", 5, 1, "x"));
  Changes.push_back(replacementToAtomicChange("key2", 15, 1, "x"));
  Changes.push_back(replacementToAtomicChange("key3", 23, 8, "xx"));
  EXPECT_EQ("int  x;\n"
            "int x;\n"
            "int  xx;\n",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, LastLineViolateColumnLimits) {
  Spec.Format = ApplyChangesSpec::kViolations;
  Spec.Style.ColumnLimit = 8;
  setInput("int  a;\n"
           "int    a;");
  Changes.push_back(replacementToAtomicChange("key1", 0, 1, "i"));
  Changes.push_back(replacementToAtomicChange("key2", 15, 2, "y;"));
  EXPECT_EQ("int  a;\n"
            "int y;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, LastLineWithNewlineViolateColumnLimits) {
  Spec.Format = ApplyChangesSpec::kViolations;
  Spec.Style.ColumnLimit = 8;
  setInput("int  a;\n"
           "int   a;\n");
  Changes.push_back(replacementToAtomicChange("key1", 0, 1, "i"));
  Changes.push_back(replacementToAtomicChange("key2", 14, 3, "y;\n"));
  EXPECT_EQ("int  a;\n"
            "int   y;\n",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, Longer) {
  setInput("int  a;");
  Changes.push_back(replacementToAtomicChange("key1", 5, 1, "bbb"));
  EXPECT_EQ("int bbb;", rewrite());
}

TEST_F(ApplyAtomicChangesTest, Shorter) {
  setInput("int  aaa;");
  Changes.push_back(replacementToAtomicChange("key1", 5, 3, "b"));
  EXPECT_EQ("int b;", rewrite());
}

TEST_F(ApplyAtomicChangesTest, OnlyFormatChangedLines) {
  setInput("int  aaa;\n"
           "int a = b;\n"
           "int  bbb;");
  Changes.push_back(replacementToAtomicChange("key1", 14, 1, "b"));
  EXPECT_EQ("int  aaa;\n"
            "int b = b;\n"
            "int  bbb;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, DisableFormatting) {
  Spec.Format = ApplyChangesSpec::kNone;
  setInput("int  aaa;\n"
           "int a   = b;\n"
           "int  bbb;");
  Changes.push_back(replacementToAtomicChange("key1", 14, 1, "b"));
  EXPECT_EQ("int  aaa;\n"
            "int b   = b;\n"
            "int  bbb;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, AdaptsToLocalPointerStyle) {
  setInput("int *aaa;\n"
           "int *bbb;");
  Changes.push_back(replacementToAtomicChange("key1", 0, 0, "int* ccc;\n"));
  EXPECT_EQ("int *ccc;\n"
            "int *aaa;\n"
            "int *bbb;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, AcceptsSurroundingFormatting) {
  setInput("   int  aaa;\n"
           "   int a = b;\n"
           "   int  bbb;");
  Changes.push_back(replacementToAtomicChange("key1", 20, 1, "b"));
  EXPECT_EQ("   int  aaa;\n"
            "   int b = b;\n"
            "   int  bbb;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, BailsOutOnConflictingChanges) {
  setInput("int c;\n"
           "int f;");
  // Insertions at the same offset are only allowed in the same AtomicChange.
  Changes.push_back(replacementToAtomicChange("key1", 0, 0, "int a;\n"));
  Changes.push_back(replacementToAtomicChange("key2", 0, 0, "int b;\n"));
  EXPECT_EQ("", rewrite(/*FailureExpected=*/true));
}

TEST_F(ApplyAtomicChangesTest, InsertsNewIncludesInRightOrder) {
  setInput("int a;");
  Changes.emplace_back(FilePath, "key1");
  Changes.back().addHeader("b");
  Changes.back().addHeader("c");
  Changes.emplace_back(FilePath, "key2");
  Changes.back().addHeader("a");
  EXPECT_EQ("#include \"a\"\n"
            "#include \"b\"\n"
            "#include \"c\"\n"
            "int a;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, RemoveAndSortIncludes) {
  setInput("#include \"a\"\n"
           "#include \"b\"\n"
           "#include \"c\"\n"
           "\n"
           "int a;");
  Changes.emplace_back(FilePath, "key1");
  Changes.back().removeHeader("b");
  EXPECT_EQ("#include \"a\"\n"
            "#include \"c\"\n"
            "\n"
            "int a;",
            rewrite());
}
TEST_F(ApplyAtomicChangesTest, InsertsSystemIncludes) {
  setInput("#include <asys>\n"
           "#include <csys>\n"
           "\n"
           "#include \"a\"\n"
           "#include \"c\"\n");
  Changes.emplace_back(FilePath, "key1");
  Changes.back().addHeader("<asys>"); // Already exists.
  Changes.back().addHeader("<b>");
  Changes.back().addHeader("<d>");
  Changes.back().addHeader("\"b-already-escaped\"");
  EXPECT_EQ("#include <asys>\n"
            "#include <b>\n"
            "#include <csys>\n"
            "#include <d>\n"
            "\n"
            "#include \"a\"\n"
            "#include \"b-already-escaped\"\n"
            "#include \"c\"\n",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, RemoveSystemIncludes) {
  setInput("#include <a>\n"
           "#include <b>\n"
           "\n"
           "#include \"c\""
           "\n"
           "int a;");
  Changes.emplace_back(FilePath, "key1");
  Changes.back().removeHeader("<a>");
  EXPECT_EQ("#include <b>\n"
            "\n"
            "#include \"c\""
            "\n"
            "int a;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest,
       DoNotFormatFollowingLinesIfSeparatedWithNewline) {
  setInput("#ifndef __H__\n"
           "#define __H__\n"
           "#include \"b\"\n"
           "\n"
           "int  a;\n"
           "int  a;\n"
           "int  a;\n"
           "#endif // __H__\n");
  Changes.push_back(replacementToAtomicChange("key1",
                                              llvm::StringRef("#ifndef __H__\n"
                                                              "#define __H__\n"
                                                              "\n"
                                                              "#include \"b\"\n"
                                                              "int  a;\n"
                                                              "int  ")
                                                  .size(),
                                              1, "b"));
  Changes.back().addHeader("a");
  EXPECT_EQ("#ifndef __H__\n"
            "#define __H__\n"
            "#include \"a\"\n"
            "#include \"b\"\n"
            "\n"
            "int  a;\n"
            "int b;\n"
            "int  a;\n"
            "#endif // __H__\n",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, FormatsCorrectLineWhenHeaderIsRemoved) {
  setInput("#include \"a\"\n"
           "\n"
           "int  a;\n"
           "int  a;\n"
           "int  a;");
  Changes.push_back(replacementToAtomicChange("key1", 27, 1, "b"));
  Changes.back().removeHeader("a");
  EXPECT_EQ("\n"
            "int  a;\n"
            "int b;\n"
            "int  a;",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, CleansUpCtorInitializers) {
  setInput("A::A() : a(), b() {}\n"
           "A::A() : a(), b() {}\n"
           "A::A() : a(), b() {}\n"
           "A::A() : a()/**/, b() {}\n"
           "A::A() : a()  ,// \n"
           "   /**/    b()    {}");
  Changes.emplace_back(FilePath, "key1");
  auto Err = Changes.back().replace(Context.Sources, getLoc(9), 3, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(35), 3, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(51), 3, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(56), 3, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(72), 3, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(97), 3, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(118), 3, "");
  ASSERT_TRUE(!Err);
  EXPECT_EQ("A::A() : b() {}\n"
            "A::A() : a() {}\n"
            "A::A() {}\n"
            "A::A() : b() {}\n"
            "A::A() {}",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, CleansUpParameterLists) {
  setInput("void f(int i, float f, string s);\n"
           "f(1, 2.0f, \"a\");\n"
           "g(1, 1);");
  Changes.emplace_back(FilePath, "key1");
  auto Err = Changes.back().replace(Context.Sources, getLoc(7), 5, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(23), 8, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(36), 1, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(45), 3, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(53), 1, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(56), 1, "");
  ASSERT_TRUE(!Err);
  EXPECT_EQ("void f(float f);\n"
            "f(2.0f);\n"
            "g();",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, DisableCleanup) {
  Spec.Cleanup = false;
  setInput("void f(int i, float f, string s);\n"
           "f(1, 2.0f, \"a\");\n"
           "g(1, 1);");
  Changes.emplace_back(FilePath, "key1");
  auto Err = Changes.back().replace(Context.Sources, getLoc(7), 5, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(23), 8, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(36), 1, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(45), 3, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(53), 1, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(56), 1, "");
  ASSERT_TRUE(!Err);
  EXPECT_EQ("void f(, float f, );\n"
            "f(, 2.0f, );\n"
            "g(, );",
            rewrite());
}

TEST_F(ApplyAtomicChangesTest, EverythingDeleted) {
  setInput("int a;");
  Changes.push_back(replacementToAtomicChange("key1", 0, 6, ""));
  EXPECT_EQ("", rewrite());
}

TEST_F(ApplyAtomicChangesTest, DoesNotDeleteInserts) {
  setInput("int a;\n"
           "int b;");
  Changes.emplace_back(FilePath, "key1");
  auto Err = Changes.back().replace(Context.Sources, getLoc(4), 1, "");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(4), 0, "b");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(11), 0, "a");
  ASSERT_TRUE(!Err);
  Err = Changes.back().replace(Context.Sources, getLoc(11), 1, "");
  ASSERT_TRUE(!Err);
  EXPECT_EQ("int b;\n"
            "int a;",
            rewrite());
}

} // end namespace tooling
} // end namespace clang
