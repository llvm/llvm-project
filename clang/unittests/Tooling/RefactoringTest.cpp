//===- unittest/Tooling/RefactoringTest.cpp - Refactoring unit tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

class ReplacementTest : public ::testing::Test {
 protected:
  Replacement createReplacement(SourceLocation Start, unsigned Length,
                                llvm::StringRef ReplacementText) {
    return Replacement(Context.Sources, Start, Length, ReplacementText);
  }

  RewriterTestContext Context;
};

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

TEST_F(ReplacementTest, CanApplyReplacements) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  Replacements Replaces;
  Replaces.insert(Replacement(Context.Sources, Context.getLocation(ID, 2, 1),
                              5, "replaced"));
  Replaces.insert(Replacement(Context.Sources, Context.getLocation(ID, 3, 1),
                              5, "other"));
  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("line1\nreplaced\nother\nline4", Context.getRewrittenText(ID));
}

// FIXME: Remove this test case when Replacements is implemented as std::vector
// instead of std::set. The other ReplacementTest tests will need to be updated
// at that point as well.
TEST_F(ReplacementTest, VectorCanApplyReplacements) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  std::vector<Replacement> Replaces;
  Replaces.push_back(Replacement(Context.Sources, Context.getLocation(ID, 2, 1),
                                 5, "replaced"));
  Replaces.push_back(
      Replacement(Context.Sources, Context.getLocation(ID, 3, 1), 5, "other"));
  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("line1\nreplaced\nother\nline4", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, SkipsDuplicateReplacements) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  Replacements Replaces;
  Replaces.insert(Replacement(Context.Sources, Context.getLocation(ID, 2, 1),
                              5, "replaced"));
  Replaces.insert(Replacement(Context.Sources, Context.getLocation(ID, 2, 1),
                              5, "replaced"));
  Replaces.insert(Replacement(Context.Sources, Context.getLocation(ID, 2, 1),
                              5, "replaced"));
  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("line1\nreplaced\nline3\nline4", Context.getRewrittenText(ID));
}

TEST_F(ReplacementTest, ApplyAllFailsIfOneApplyFails) {
  // This test depends on the value of the file name of an invalid source
  // location being in the range ]a, z[.
  FileID IDa = Context.createInMemoryFile("a.cpp", "text");
  FileID IDz = Context.createInMemoryFile("z.cpp", "text");
  Replacements Replaces;
  Replaces.insert(Replacement(Context.Sources, Context.getLocation(IDa, 1, 1),
                              4, "a"));
  Replaces.insert(Replacement(Context.Sources, SourceLocation(),
                              5, "2"));
  Replaces.insert(Replacement(Context.Sources, Context.getLocation(IDz, 1, 1),
                              4, "z"));
  EXPECT_FALSE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("a", Context.getRewrittenText(IDa));
  EXPECT_EQ("z", Context.getRewrittenText(IDz));
}

TEST(ShiftedCodePositionTest, FindsNewCodePosition) {
  Replacements Replaces;
  Replaces.insert(Replacement("", 0, 1, ""));
  Replaces.insert(Replacement("", 4, 3, " "));
  // Assume ' int   i;' is turned into 'int i;' and cursor is located at '|'.
  EXPECT_EQ(0u, shiftedCodePosition(Replaces, 0)); // |int   i;
  EXPECT_EQ(0u, shiftedCodePosition(Replaces, 1)); //  |nt   i;
  EXPECT_EQ(1u, shiftedCodePosition(Replaces, 2)); //  i|t   i;
  EXPECT_EQ(2u, shiftedCodePosition(Replaces, 3)); //  in|   i;
  EXPECT_EQ(3u, shiftedCodePosition(Replaces, 4)); //  int|  i;
  EXPECT_EQ(3u, shiftedCodePosition(Replaces, 5)); //  int | i;
  EXPECT_EQ(3u, shiftedCodePosition(Replaces, 6)); //  int  |i;
  EXPECT_EQ(4u, shiftedCodePosition(Replaces, 7)); //  int   |;
  EXPECT_EQ(5u, shiftedCodePosition(Replaces, 8)); //  int   i|
}

// FIXME: Remove this test case when Replacements is implemented as std::vector
// instead of std::set. The other ReplacementTest tests will need to be updated
// at that point as well.
TEST(ShiftedCodePositionTest, VectorFindsNewCodePositionWithInserts) {
  std::vector<Replacement> Replaces;
  Replaces.push_back(Replacement("", 0, 1, ""));
  Replaces.push_back(Replacement("", 4, 3, " "));
  // Assume ' int   i;' is turned into 'int i;' and cursor is located at '|'.
  EXPECT_EQ(0u, shiftedCodePosition(Replaces, 0)); // |int   i;
  EXPECT_EQ(0u, shiftedCodePosition(Replaces, 1)); //  |nt   i;
  EXPECT_EQ(1u, shiftedCodePosition(Replaces, 2)); //  i|t   i;
  EXPECT_EQ(2u, shiftedCodePosition(Replaces, 3)); //  in|   i;
  EXPECT_EQ(3u, shiftedCodePosition(Replaces, 4)); //  int|  i;
  EXPECT_EQ(3u, shiftedCodePosition(Replaces, 5)); //  int | i;
  EXPECT_EQ(3u, shiftedCodePosition(Replaces, 6)); //  int  |i;
  EXPECT_EQ(4u, shiftedCodePosition(Replaces, 7)); //  int   |;
  EXPECT_EQ(5u, shiftedCodePosition(Replaces, 8)); //  int   i|
}

TEST(ShiftedCodePositionTest, FindsNewCodePositionWithInserts) {
  Replacements Replaces;
  Replaces.insert(Replacement("", 4, 0, "\"\n\""));
  // Assume '"12345678"' is turned into '"1234"\n"5678"'.
  EXPECT_EQ(3u, shiftedCodePosition(Replaces, 3)); // "123|5678"
  EXPECT_EQ(7u, shiftedCodePosition(Replaces, 4)); // "1234|678"
  EXPECT_EQ(8u, shiftedCodePosition(Replaces, 5)); // "12345|78"
}

TEST(ShiftedCodePositionTest, FindsNewCodePositionInReplacedText) {
  Replacements Replaces;
  // Replace the first four characters with "abcd".
  Replaces.insert(Replacement("", 0, 4, "abcd"));
  for (unsigned i = 0; i < 3; ++i)
    EXPECT_EQ(i, shiftedCodePosition(Replaces, i));
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
    const FileEntry *File = Context.Files.getFile(Path);
    assert(File != nullptr);

    StringRef Found =
        TemporaryFiles.insert(std::make_pair(Name, Path.str())).first->second;
    assert(Found == Path);
    (void)Found;
    return Context.Sources.createFileID(File, SourceLocation(), SrcMgr::C_User);
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
    return (*FileBuffer)->getBuffer();
  }

  llvm::StringMap<std::string> TemporaryFiles;
  RewriterTestContext Context;
};

TEST_F(FlushRewrittenFilesTest, StoresChangesOnDisk) {
  FileID ID = createFile("input.cpp", "line1\nline2\nline3\nline4");
  Replacements Replaces;
  Replaces.insert(Replacement(Context.Sources, Context.getLocation(ID, 2, 1),
                              5, "replaced"));
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
    return runToolOnCode(new TestAction(this), Code);
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
      return llvm::make_unique<FindConsumer>(Visitor);
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

TEST(DeduplicateTest, removesDuplicates) {
  std::vector<Replacement> Input;
  Input.push_back(Replacement("fileA", 50, 0, " foo "));
  Input.push_back(Replacement("fileA", 10, 3, " bar "));
  Input.push_back(Replacement("fileA", 10, 2, " bar ")); // Length differs
  Input.push_back(Replacement("fileA", 9,  3, " bar ")); // Offset differs
  Input.push_back(Replacement("fileA", 50, 0, " foo ")); // Duplicate
  Input.push_back(Replacement("fileA", 51, 3, " bar "));
  Input.push_back(Replacement("fileB", 51, 3, " bar ")); // Filename differs!
  Input.push_back(Replacement("fileB", 60, 1, " bar "));
  Input.push_back(Replacement("fileA", 60, 2, " bar "));
  Input.push_back(Replacement("fileA", 51, 3, " moo ")); // Replacement text
                                                         // differs!

  std::vector<Replacement> Expected;
  Expected.push_back(Replacement("fileA", 9,  3, " bar "));
  Expected.push_back(Replacement("fileA", 10, 2, " bar "));
  Expected.push_back(Replacement("fileA", 10, 3, " bar "));
  Expected.push_back(Replacement("fileA", 50, 0, " foo "));
  Expected.push_back(Replacement("fileA", 51, 3, " bar "));
  Expected.push_back(Replacement("fileA", 51, 3, " moo "));
  Expected.push_back(Replacement("fileB", 60, 1, " bar "));
  Expected.push_back(Replacement("fileA", 60, 2, " bar "));

  std::vector<Range> Conflicts; // Ignored for this test
  deduplicate(Input, Conflicts);

  EXPECT_EQ(3U, Conflicts.size());
  EXPECT_EQ(Expected, Input);
}

TEST(DeduplicateTest, detectsConflicts) {
  {
    std::vector<Replacement> Input;
    Input.push_back(Replacement("fileA", 0, 5, " foo "));
    Input.push_back(Replacement("fileA", 0, 5, " foo ")); // Duplicate not a
                                                          // conflict.
    Input.push_back(Replacement("fileA", 2, 6, " bar "));
    Input.push_back(Replacement("fileA", 7, 3, " moo "));

    std::vector<Range> Conflicts;
    deduplicate(Input, Conflicts);

    // One duplicate is removed and the remaining three items form one
    // conflicted range.
    ASSERT_EQ(3u, Input.size());
    ASSERT_EQ(1u, Conflicts.size());
    ASSERT_EQ(0u, Conflicts.front().getOffset());
    ASSERT_EQ(3u, Conflicts.front().getLength());
  }
  {
    std::vector<Replacement> Input;

    // Expected sorted order is shown. It is the sorted order to which the
    // returned conflict info refers to.
    Input.push_back(Replacement("fileA", 0,  5, " foo "));  // 0
    Input.push_back(Replacement("fileA", 5,  5, " bar "));  // 1
    Input.push_back(Replacement("fileA", 6,  0, " bar "));  // 3
    Input.push_back(Replacement("fileA", 5,  5, " moo "));  // 2
    Input.push_back(Replacement("fileA", 7,  2, " bar "));  // 4
    Input.push_back(Replacement("fileA", 15, 5, " golf ")); // 5
    Input.push_back(Replacement("fileA", 16, 5, " bag "));  // 6
    Input.push_back(Replacement("fileA", 10, 3, " club ")); // 7

    // #3 is special in that it is completely contained by another conflicting
    // Replacement. #4 ensures #3 hasn't messed up the conflicting range size.

    std::vector<Range> Conflicts;
    deduplicate(Input, Conflicts);

    // No duplicates
    ASSERT_EQ(8u, Input.size());
    ASSERT_EQ(2u, Conflicts.size());
    ASSERT_EQ(1u, Conflicts[0].getOffset());
    ASSERT_EQ(4u, Conflicts[0].getLength());
    ASSERT_EQ(6u, Conflicts[1].getOffset());
    ASSERT_EQ(2u, Conflicts[1].getLength());
  }
}

class MergeReplacementsTest : public ::testing::Test {
protected:
  void mergeAndTestRewrite(StringRef Code, StringRef Intermediate,
                           StringRef Result, const Replacements &First,
                           const Replacements &Second) {
    // These are mainly to verify the test itself and make it easier to read.
    std::string AfterFirst = applyAllReplacements(Code, First);
    std::string InSequenceRewrite = applyAllReplacements(AfterFirst, Second);
    EXPECT_EQ(Intermediate, AfterFirst);
    EXPECT_EQ(Result, InSequenceRewrite);

    tooling::Replacements Merged = mergeReplacements(First, Second);
    std::string MergedRewrite = applyAllReplacements(Code, Merged);
    EXPECT_EQ(InSequenceRewrite, MergedRewrite);
    if (InSequenceRewrite != MergedRewrite)
      for (tooling::Replacement M : Merged)
        llvm::errs() << M.getOffset() << " " << M.getLength() << " "
                     << M.getReplacementText() << "\n";
  }
  void mergeAndTestRewrite(StringRef Code, const Replacements &First,
                           const Replacements &Second) {
    std::string InSequenceRewrite =
        applyAllReplacements(applyAllReplacements(Code, First), Second);
    tooling::Replacements Merged = mergeReplacements(First, Second);
    std::string MergedRewrite = applyAllReplacements(Code, Merged);
    EXPECT_EQ(InSequenceRewrite, MergedRewrite);
    if (InSequenceRewrite != MergedRewrite)
      for (tooling::Replacement M : Merged)
        llvm::errs() << M.getOffset() << " " << M.getLength() << " "
                     << M.getReplacementText() << "\n";
  }
};

TEST_F(MergeReplacementsTest, Offsets) {
  mergeAndTestRewrite("aaa", "aabab", "cacabab",
                      {{"", 2, 0, "b"}, {"", 3, 0, "b"}},
                      {{"", 0, 0, "c"}, {"", 1, 0, "c"}});
  mergeAndTestRewrite("aaa", "babaa", "babacac",
                      {{"", 0, 0, "b"}, {"", 1, 0, "b"}},
                      {{"", 4, 0, "c"}, {"", 5, 0, "c"}});
  mergeAndTestRewrite("aaaa", "aaa", "aac", {{"", 1, 1, ""}},
                      {{"", 2, 1, "c"}});

  mergeAndTestRewrite("aa", "bbabba", "bbabcba",
                      {{"", 0, 0, "bb"}, {"", 1, 0, "bb"}}, {{"", 4, 0, "c"}});
}

TEST_F(MergeReplacementsTest, Concatenations) {
  // Basic concatenations. It is important to merge these into a single
  // replacement to ensure the correct order.
  EXPECT_EQ((Replacements{{"", 0, 0, "ab"}}),
            mergeReplacements({{"", 0, 0, "a"}}, {{"", 1, 0, "b"}}));
  EXPECT_EQ((Replacements{{"", 0, 0, "ba"}}),
            mergeReplacements({{"", 0, 0, "a"}}, {{"", 0, 0, "b"}}));
  mergeAndTestRewrite("", "a", "ab", {{"", 0, 0, "a"}}, {{"", 1, 0, "b"}});
  mergeAndTestRewrite("", "a", "ba", {{"", 0, 0, "a"}}, {{"", 0, 0, "b"}});
}

TEST_F(MergeReplacementsTest, NotChangingLengths) {
  mergeAndTestRewrite("aaaa", "abba", "acca", {{"", 1, 2, "bb"}},
                      {{"", 1, 2, "cc"}});
  mergeAndTestRewrite("aaaa", "abba", "abcc", {{"", 1, 2, "bb"}},
                      {{"", 2, 2, "cc"}});
  mergeAndTestRewrite("aaaa", "abba", "ccba", {{"", 1, 2, "bb"}},
                      {{"", 0, 2, "cc"}});
  mergeAndTestRewrite("aaaaaa", "abbdda", "abccda",
                      {{"", 1, 2, "bb"}, {"", 3, 2, "dd"}}, {{"", 2, 2, "cc"}});
}

TEST_F(MergeReplacementsTest, OverlappingRanges) {
  mergeAndTestRewrite("aaa", "bbd", "bcbcd",
                      {{"", 0, 1, "bb"}, {"", 1, 2, "d"}},
                      {{"", 1, 0, "c"}, {"", 2, 0, "c"}});

  mergeAndTestRewrite("aaaa", "aabbaa", "acccca", {{"", 2, 0, "bb"}},
                      {{"", 1, 4, "cccc"}});
  mergeAndTestRewrite("aaaa", "aababa", "acccca",
                      {{"", 2, 0, "b"}, {"", 3, 0, "b"}}, {{"", 1, 4, "cccc"}});
  mergeAndTestRewrite("aaaaaa", "abbbba", "abba", {{"", 1, 4, "bbbb"}},
                      {{"", 2, 2, ""}});
  mergeAndTestRewrite("aaaa", "aa", "cc", {{"", 1, 1, ""}, {"", 2, 1, ""}},
                      {{"", 0, 2, "cc"}});
  mergeAndTestRewrite("aa", "abbba", "abcbcba", {{"", 1, 0, "bbb"}},
                      {{"", 2, 0, "c"}, {"", 3, 0, "c"}});

  mergeAndTestRewrite("aaa", "abbab", "ccdd",
                      {{"", 0, 1, ""}, {"", 2, 0, "bb"}, {"", 3, 0, "b"}},
                      {{"", 0, 2, "cc"}, {"", 2, 3, "dd"}});
  mergeAndTestRewrite("aa", "babbab", "ccdd",
                      {{"", 0, 0, "b"}, {"", 1, 0, "bb"}, {"", 2, 0, "b"}},
                      {{"", 0, 3, "cc"}, {"", 3, 3, "dd"}});
}

} // end namespace tooling
} // end namespace clang
