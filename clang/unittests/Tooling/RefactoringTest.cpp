//===- unittest/Tooling/RefactoringTest.cpp - Refactoring unit tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
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

TEST_F(ReplacementTest, FailAddReplacements) {
  Replacements Replaces;
  auto Err = Replaces.add(Replacement("x.cc", 0, 10, "3"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 0, 2, ""));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 2, 2, ""));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("y.cc", 20, 2, ""));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));
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
  Err = Replaces.add(Replacement("x.cc", 5, 5, "fail"));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE((bool)Err);
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

TEST_F(ReplacementTest, FailAddInsertAtOtherInsert) {
  Replacements Replaces;
  auto Err = Replaces.add(Replacement("x.cc", 10, 0, "a"));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, "b"));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));

  Replaces.clear();
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));

  Replaces.clear();
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 3, ""));
  EXPECT_TRUE(!Err);
  llvm::consumeError(std::move(Err));
  Err = Replaces.add(Replacement("x.cc", 10, 0, ""));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));
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

TEST_F(ReplacementTest, SkipsDuplicateReplacements) {
  FileID ID = Context.createInMemoryFile("input.cpp",
                                         "line1\nline2\nline3\nline4");
  auto Replaces = toReplacements({Replacement(
      Context.Sources, Context.getLocation(ID, 2, 1), 5, "replaced")});

  auto Err = Replaces.add(Replacement(
      Context.Sources, Context.getLocation(ID, 2, 1), 5, "replaced"));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));

  Err = Replaces.add(Replacement(Context.Sources, Context.getLocation(ID, 2, 1),
                                 5, "replaced"));
  EXPECT_TRUE((bool)Err);
  llvm::consumeError(std::move(Err));

  EXPECT_TRUE(applyAllReplacements(Replaces, Context.Rewrite));
  EXPECT_EQ("line1\nreplaced\nline3\nline4", Context.getRewrittenText(ID));
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
  FileToReplaces[File1] = toReplacements(
      {tooling::Replacement(Context.Sources, Context.getLocation(ID1, 1, 1), 6,
                            "auto "),
       tooling::Replacement(Context.Sources, Context.getLocation(ID1, 3, 10), 1,
                            "12345678901")});
  FileToReplaces[File2] = toReplacements(
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

} // end namespace tooling
} // end namespace clang
