//===- unittest/AST/RandstructTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for Clang's structure field layout randomization.
//
//===----------------------------------------------------------------------===//

/*
 * Build this test suite by running `make ASTTests` in the build folder.
 *
 * Run this test suite by running the following in the build folder:
 * ` ./tools/clang/unittests/AST/ASTTests
 * --gtest_filter=RecordLayoutRandomization*`
 */

#include "clang/AST/Randstruct.h"
#include "gtest/gtest.h"

#include "DeclMatcher.h"
#include "clang/AST/RecordLayout.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/ToolOutputFile.h"

#include <vector>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::randstruct;

using field_names = std::vector<std::string>;

constexpr const char Seed[] = "1234567890abcdef";

static RecordDecl *getRecordDeclFromAST(const ASTContext &C,
                                        const std::string &Name) {
  RecordDecl *RD = FirstDeclMatcher<RecordDecl>().match(
      C.getTranslationUnitDecl(), recordDecl(hasName(Name)));
  return RD;
}

static std::vector<std::string> getFieldNamesFromRecord(const RecordDecl *RD) {
  std::vector<std::string> Fields;

  Fields.reserve(8);
  for (auto *Field : RD->fields())
    Fields.push_back(Field->getNameAsString());

  return Fields;
}

static bool isSubsequence(const field_names &Seq, const field_names &Subseq) {
  unsigned SeqLen = Seq.size();
  unsigned SubLen = Subseq.size();

  bool IsSubseq = false;
  for (unsigned I = 0; I < SeqLen; ++I)
    if (Seq[I] == Subseq[0]) {
      IsSubseq = true;
      for (unsigned J = 0; J + I < SeqLen && J < SubLen; ++J) {
        if (Seq[J + I] != Subseq[J]) {
          IsSubseq = false;
          break;
        }
      }
    }

  return IsSubseq;
}

static bool recordsEqual(const std::unique_ptr<ASTUnit> &LHS,
                         const std::unique_ptr<ASTUnit> &RHS,
                         const std::string &RecordName) {
  const RecordDecl *LHSRD =
      getRecordDeclFromAST(LHS->getASTContext(), RecordName);
  const RecordDecl *RHSRD =
      getRecordDeclFromAST(LHS->getASTContext(), RecordName);

  return getFieldNamesFromRecord(LHSRD) == getFieldNamesFromRecord(RHSRD);
}

static std::unique_ptr<ASTUnit>
makeAST(const std::string &SourceCode, bool ExpectError = false,
        std::vector<std::string> RecordNames = std::vector<std::string>()) {
  std::vector<std::string> Args = getCommandLineArgsForTesting(Lang_C99);
  Args.push_back("-frandomize-layout-seed=" + std::string(Seed));

  IgnoringDiagConsumer IgnoringConsumer = IgnoringDiagConsumer();

  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
      SourceCode, Args, "input.c", "clang-tool",
      std::make_shared<PCHContainerOperations>(),
      tooling::getClangStripDependencyFileAdjuster(),
      tooling::FileContentMappings(), &IgnoringConsumer);

  int SeedFileFD = -1;
  llvm::SmallString<256> SeedFilename;
  EXPECT_FALSE(llvm::sys::fs::createTemporaryFile("seed", "rng", SeedFileFD,
                                                  SeedFilename));
  llvm::ToolOutputFile SeedFile(SeedFilename, SeedFileFD);
  SeedFile.os() << Seed << "\n";

  Args.clear();
  Args = getCommandLineArgsForTesting(Lang_C99);
  Args.push_back("-frandomize-layout-seed-file=" +
                 SeedFile.getFilename().str());

  std::unique_ptr<ASTUnit> ASTFileSeed = tooling::buildASTFromCodeWithArgs(
      SourceCode, Args, "input.c", "clang-tool",
      std::make_shared<PCHContainerOperations>(),
      tooling::getClangStripDependencyFileAdjuster(),
      tooling::FileContentMappings(), &IgnoringConsumer);

  if (!ExpectError) {
    if (RecordNames.empty())
      RecordNames.push_back("test");

    for (std::string Name : RecordNames)
      EXPECT_TRUE(recordsEqual(AST, ASTFileSeed, Name));
  }

  return AST;
}

namespace clang {
namespace ast_matchers {

long declCount(const RecordDecl *RD) {
  return llvm::count_if(RD->decls(), [&](const Decl *D) {
    return isa<FieldDecl>(D) || isa<RecordDecl>(D);
  });
}

#define RANDSTRUCT_TEST_SUITE_TEST RecordLayoutRandomizationTestSuiteTest

TEST(RANDSTRUCT_TEST_SUITE_TEST, CanDetermineIfSubsequenceExists) {
  const field_names Seq = {"a", "b", "c", "d"};

  EXPECT_TRUE(isSubsequence(Seq, {"b", "c"}));
  EXPECT_TRUE(isSubsequence(Seq, {"a", "b", "c", "d"}));
  EXPECT_TRUE(isSubsequence(Seq, {"b", "c", "d"}));
  EXPECT_TRUE(isSubsequence(Seq, {"a"}));
  EXPECT_FALSE(isSubsequence(Seq, {"a", "d"}));
}

#define RANDSTRUCT_TEST RecordLayoutRandomization

TEST(RANDSTRUCT_TEST, UnmarkedStruct) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    };
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_FALSE(RD->hasAttr<RandomizeLayoutAttr>());
  EXPECT_FALSE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
}

TEST(RANDSTRUCT_TEST, MarkedNoRandomize) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((no_randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_TRUE(RD->hasAttr<NoRandomizeLayoutAttr>());
  EXPECT_FALSE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
}

TEST(RANDSTRUCT_TEST, MarkedRandomize) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_TRUE(RD->hasAttr<RandomizeLayoutAttr>());
  EXPECT_TRUE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
}

TEST(RANDSTRUCT_TEST, MismatchedAttrsDeclVsDef) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test __attribute__((randomize_layout));
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((no_randomize_layout));
  )c",
                                         true);

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const DiagnosticsEngine &Diags = AST->getDiagnostics();

  EXPECT_FALSE(Diags.hasFatalErrorOccurred());
  EXPECT_FALSE(Diags.hasUncompilableErrorOccurred());
  EXPECT_FALSE(Diags.hasUnrecoverableErrorOccurred());
  EXPECT_EQ(Diags.getNumWarnings(), 1u);
  EXPECT_EQ(Diags.getNumErrors(), 0u);
}

TEST(RANDSTRUCT_TEST, MismatchedAttrsRandomizeVsNoRandomize) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((randomize_layout)) __attribute__((no_randomize_layout));
  )c",
                                         true);

  EXPECT_TRUE(AST->getDiagnostics().hasErrorOccurred());

  const DiagnosticsEngine &Diags = AST->getDiagnostics();

  EXPECT_TRUE(Diags.hasUncompilableErrorOccurred());
  EXPECT_TRUE(Diags.hasUnrecoverableErrorOccurred());
  EXPECT_EQ(Diags.getNumWarnings(), 0u);
  EXPECT_EQ(Diags.getNumErrors(), 1u);
}

TEST(RANDSTRUCT_TEST, MismatchedAttrsNoRandomizeVsRandomize) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test3 {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((no_randomize_layout)) __attribute__((randomize_layout));
  )c",
                                         true);

  EXPECT_TRUE(AST->getDiagnostics().hasErrorOccurred());

  const DiagnosticsEngine &Diags = AST->getDiagnostics();

  EXPECT_TRUE(Diags.hasUncompilableErrorOccurred());
  EXPECT_TRUE(Diags.hasUnrecoverableErrorOccurred());
  EXPECT_EQ(Diags.getNumWarnings(), 0u);
  EXPECT_EQ(Diags.getNumErrors(), 1u);
}

TEST(RANDSTRUCT_TEST, CheckAdjacentBitfieldsRemainAdjacentAfterRandomization) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a;
        int b;
        int x : 1;
        int y : 1;
        int z : 1;
        int c;
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  const field_names Actual = getFieldNamesFromRecord(RD);
  const field_names Subseq = {"x", "y", "z"};

  EXPECT_TRUE(RD->isRandomized());
  EXPECT_TRUE(isSubsequence(Actual, Subseq));
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
}

TEST(RANDSTRUCT_TEST, CheckFlexibleArrayMemberRemainsAtEndOfStructure1) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a;
        int b;
        int c;
        int d;
        int e;
        int f;
        int g;
        int h;
        char name[];
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_TRUE(RD->hasFlexibleArrayMember());
  EXPECT_TRUE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
  EXPECT_EQ(getFieldNamesFromRecord(RD).back(), "name");
}

TEST(RANDSTRUCT_TEST, CheckFlexibleArrayMemberRemainsAtEndOfStructure2) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a;
        int b;
        int c;
        int d;
        int e;
        int f;
        int g;
        int h;
        char name[0];
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_FALSE(RD->hasFlexibleArrayMember());
  EXPECT_TRUE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
  EXPECT_EQ(getFieldNamesFromRecord(RD).back(), "name");
}

TEST(RANDSTRUCT_TEST, CheckFlexibleArrayMemberRemainsAtEndOfStructure3) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a;
        int b;
        int c;
        int d;
        int e;
        int f;
        int g;
        int h;
        char name[1];
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_FALSE(RD->hasFlexibleArrayMember());
  EXPECT_TRUE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
  EXPECT_EQ(getFieldNamesFromRecord(RD).back(), "name");
}

TEST(RANDSTRUCT_TEST, RandstructDoesNotOverrideThePackedAttr) {
  std::unique_ptr<ASTUnit> AST =
      makeAST(R"c(
    struct test_struct {
        char a;
        float b[3];
        short c;
        int d;
    } __attribute__((packed, randomize_layout));

    struct another_struct {
        char a;
        char b[5];
        int c;
    } __attribute__((packed, randomize_layout));

    struct last_struct {
        char a;
        long long b;
        int c[];
    } __attribute__((packed, randomize_layout));
  )c",
              false,
              std::vector<std::string>(
                  {"test_struct", "another_struct", "last_struct"}));

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  // FIXME (?): calling getASTRecordLayout is probably a necessary evil so that
  // Clang's RecordBuilders can actually flesh out the information like
  // alignment, etc.
  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "test_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);
    long OriginalDeclCount = declCount(RD);

    EXPECT_TRUE(RD->isRandomized());
    EXPECT_EQ(19, Layout->getSize().getQuantity());
    EXPECT_EQ(OriginalDeclCount, declCount(RD));
  }

  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "another_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);
    long OriginalDeclCount = declCount(RD);

    EXPECT_TRUE(RD->isRandomized());
    EXPECT_EQ(10, Layout->getSize().getQuantity());
    EXPECT_EQ(OriginalDeclCount, declCount(RD));
  }

  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "last_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);
    long OriginalDeclCount = declCount(RD);

    EXPECT_TRUE(RD->isRandomized());
    EXPECT_EQ(9, Layout->getSize().getQuantity());
    EXPECT_EQ(OriginalDeclCount, declCount(RD));
  }
}

TEST(RANDSTRUCT_TEST, ZeroWidthBitfieldsSeparateAllocationUnits) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a : 1;
        int   : 0;
        int b : 1;
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_TRUE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
}

TEST(RANDSTRUCT_TEST, RandstructDoesNotRandomizeUnionFieldOrder) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    union test {
        int a;
        int b;
        int c;
        int d;
        int e;
        int f;
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_FALSE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
}

TEST(RANDSTRUCT_TEST, AnonymousStructsAndUnionsRetainFieldOrder) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a;
        struct sub_struct {
            int b;
            int c;
            int d;
            int e;
            int f;
        } __attribute__((randomize_layout)) s;
        int f;
        struct {
            int g;
            int h;
            int i;
            int j;
            int k;
        };
        int l;
        union {
            int m;
            int n;
            int o;
            int p;
            int q;
        };
        int r;
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_TRUE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));

  bool AnonStructTested = false;
  bool AnonUnionTested = false;

  for (const Decl *D : RD->decls())
    if (const FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
      if (const auto *Record = FD->getType()->getAs<RecordType>()) {
        RD = Record->getDecl();
        if (RD->isAnonymousStructOrUnion()) {
          // These field orders shouldn't change.
          if (RD->isUnion()) {
            const field_names Expected = {"m", "n", "o", "p", "q"};

            EXPECT_EQ(Expected, getFieldNamesFromRecord(RD));
            AnonUnionTested = true;
          } else {
            const field_names Expected = {"g", "h", "i", "j", "k"};

            EXPECT_EQ(Expected, getFieldNamesFromRecord(RD));
            AnonStructTested = true;
          }
        }
      }
    }

  EXPECT_TRUE(AnonStructTested);
  EXPECT_TRUE(AnonUnionTested);
}

TEST(RANDSTRUCT_TEST, AnonymousStructsAndUnionsReferenced) {
  std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int bacon;
        long lettuce;
        struct { double avocado; char blech; };
        long long tomato;
        union { char toast[8]; unsigned toast_thing; };
        float mayonnaise;
    } __attribute__((randomize_layout));

    int foo(struct test *t) {
      return t->blech;
    }

    char *bar(struct test *t) {
      return t->toast;
    }
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  long OriginalDeclCount = declCount(RD);

  EXPECT_TRUE(RD->isRandomized());
  EXPECT_EQ(OriginalDeclCount, declCount(RD));
}

} // namespace ast_matchers
} // namespace clang
