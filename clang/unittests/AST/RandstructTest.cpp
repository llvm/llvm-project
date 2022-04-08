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
 * --gtest_filter=StructureLayoutRandomization*`
 */

#include "clang/AST/Randstruct.h"
#include "gtest/gtest.h"

#include "DeclMatcher.h"
#include "clang/AST/RecordLayout.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Tooling/Tooling.h"

#include <vector>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::randstruct;

using field_names = std::vector<std::string>;

namespace {

std::unique_ptr<ASTUnit> makeAST(const std::string &SourceCode,
                                 bool ExpectErr = false) {
  std::vector<std::string> Args = getCommandLineArgsForTesting(Lang_C99);
  Args.push_back("-frandomize-layout-seed=1234567890abcdef");

  IgnoringDiagConsumer IgnoringConsumer = IgnoringDiagConsumer();

  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
      SourceCode, Args, "input.c", "clang-tool",
      std::make_shared<PCHContainerOperations>(),
      tooling::getClangStripDependencyFileAdjuster(),
      tooling::FileContentMappings(), &IgnoringConsumer);

  if (ExpectErr)
    EXPECT_TRUE(AST->getDiagnostics().hasErrorOccurred());
  else
    EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  return AST;
}

RecordDecl *getRecordDeclFromAST(const ASTContext &C, const std::string &Name) {
  RecordDecl *RD = FirstDeclMatcher<RecordDecl>().match(
      C.getTranslationUnitDecl(), recordDecl(hasName(Name)));
  return RD;
}

std::vector<std::string> getFieldNamesFromRecord(const RecordDecl *RD) {
  std::vector<std::string> Fields;

  Fields.reserve(8);
  for (auto *Field : RD->fields())
    Fields.push_back(Field->getNameAsString());

  return Fields;
}

bool isSubsequence(const field_names &Seq, const field_names &Subseq) {
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

} // end anonymous namespace

namespace clang {
namespace ast_matchers {

#define RANDSTRUCT_TEST_SUITE_TEST StructureLayoutRandomizationTestSuiteTest

TEST(RANDSTRUCT_TEST_SUITE_TEST, CanDetermineIfSubsequenceExists) {
  const field_names Seq = {"a", "b", "c", "d"};

  ASSERT_TRUE(isSubsequence(Seq, {"b", "c"}));
  ASSERT_TRUE(isSubsequence(Seq, {"a", "b", "c", "d"}));
  ASSERT_TRUE(isSubsequence(Seq, {"b", "c", "d"}));
  ASSERT_TRUE(isSubsequence(Seq, {"a"}));
  ASSERT_FALSE(isSubsequence(Seq, {"a", "d"}));
}

#define RANDSTRUCT_TEST StructureLayoutRandomization

TEST(RANDSTRUCT_TEST, UnmarkedStruct) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    };
  )c");

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  const field_names Expected = {"bacon", "lettuce", "tomato", "mayonnaise"};

  ASSERT_FALSE(RD->hasAttr<RandomizeLayoutAttr>());
  ASSERT_FALSE(RD->isRandomized());
  ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
}

TEST(RANDSTRUCT_TEST, MarkedNoRandomize) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((no_randomize_layout));
  )c");

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  const field_names Expected = {"bacon", "lettuce", "tomato", "mayonnaise"};

  ASSERT_TRUE(RD->hasAttr<NoRandomizeLayoutAttr>());
  ASSERT_FALSE(RD->isRandomized());
  ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
}

TEST(RANDSTRUCT_TEST, MarkedRandomize) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((randomize_layout));
  )c");

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
#ifdef _WIN32
  const field_names Expected = {"lettuce", "bacon", "mayonnaise", "tomato"};
#else
  const field_names Expected = {"mayonnaise", "bacon", "tomato", "lettuce"};
#endif

  ASSERT_TRUE(RD->hasAttr<RandomizeLayoutAttr>());
  ASSERT_TRUE(RD->isRandomized());
  ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
}

TEST(RANDSTRUCT_TEST, MismatchedAttrsDeclVsDef) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test __attribute__((randomize_layout));
    struct test {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((no_randomize_layout));
  )c");

  DiagnosticsEngine &Diags = AST->getDiagnostics();

  EXPECT_FALSE(Diags.hasFatalErrorOccurred());
  EXPECT_FALSE(Diags.hasUncompilableErrorOccurred());
  EXPECT_FALSE(Diags.hasUnrecoverableErrorOccurred());
  EXPECT_EQ(Diags.getNumWarnings(), 1u);
  EXPECT_EQ(Diags.getNumErrors(), 0u);
}

TEST(RANDSTRUCT_TEST, MismatchedAttrsRandomizeVsNoRandomize) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test2 {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((randomize_layout)) __attribute__((no_randomize_layout));
  )c", true);

  DiagnosticsEngine &Diags = AST->getDiagnostics();

  EXPECT_TRUE(Diags.hasUncompilableErrorOccurred());
  EXPECT_TRUE(Diags.hasUnrecoverableErrorOccurred());
  EXPECT_EQ(Diags.getNumWarnings(), 0u);
  EXPECT_EQ(Diags.getNumErrors(), 1u);
}

TEST(RANDSTRUCT_TEST, MismatchedAttrsNoRandomizeVsRandomize) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test3 {
        int bacon;
        long lettuce;
        long long tomato;
        float mayonnaise;
    } __attribute__((no_randomize_layout)) __attribute__((randomize_layout));
  )c", true);

  DiagnosticsEngine &Diags = AST->getDiagnostics();

  EXPECT_TRUE(Diags.hasUncompilableErrorOccurred());
  EXPECT_TRUE(Diags.hasUnrecoverableErrorOccurred());
  EXPECT_EQ(Diags.getNumWarnings(), 0u);
  EXPECT_EQ(Diags.getNumErrors(), 1u);
}

TEST(RANDSTRUCT_TEST, CheckAdjacentBitfieldsRemainAdjacentAfterRandomization) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a;
        int b;
        int x : 1;
        int y : 1;
        int z : 1;
        int c;
    } __attribute__((randomize_layout));
  )c");

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");

#ifdef _WIN32
  const field_names Expected = {"b", "a", "c", "x", "y", "z"};
#else
  const field_names Expected = {"c", "x", "y", "z", "b", "a"};
#endif
  const field_names Subseq = {"x", "y", "z"};
  const field_names Actual = getFieldNamesFromRecord(RD);

  ASSERT_TRUE(isSubsequence(Actual, Subseq));
  ASSERT_EQ(Expected, Actual);
}

TEST(RANDSTRUCT_TEST, CheckVariableLengthArrayMemberRemainsAtEndOfStructure) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a;
        double b;
        short c;
        char name[];
    } __attribute__((randomize_layout));
  )c");

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
#ifdef _WIN32
  const field_names Expected = {"b", "a", "c", "name"};
#else
  const field_names Expected = {"b", "c", "a", "name"};
#endif

  ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
}

TEST(RANDSTRUCT_TEST, RandstructDoesNotOverrideThePackedAttr) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
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
  )c");

  // FIXME (?): calling getASTRecordLayout is probably a necessary evil so that
  // Clang's RecordBuilders can actually flesh out the information like
  // alignment, etc.
  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "test_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);
#ifdef _WIN32
    const field_names Expected = {"a", "c", "d", "b"};
#else
    const field_names Expected = {"c", "a", "d", "b"};
#endif

    ASSERT_EQ(19, Layout->getSize().getQuantity());
    ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
  }

  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "another_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);
#ifdef _WIN32
    const field_names Expected = {"a", "b", "c"};
#else
    const field_names Expected = {"c", "a", "b"};
#endif

    ASSERT_EQ(10, Layout->getSize().getQuantity());
    ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
  }

  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "last_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);
    const field_names Expected = {"b", "c", "a"};

    ASSERT_EQ(9, Layout->getSize().getQuantity());
    ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
  }
}

TEST(RANDSTRUCT_TEST, ZeroWidthBitfieldsSeparateAllocationUnits) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test_struct {
        int a : 1;
        int   : 0;
        int b : 1;
    } __attribute__((randomize_layout));
  )c");

  const RecordDecl *RD =
      getRecordDeclFromAST(AST->getASTContext(), "test_struct");
#ifdef _WIN32
  const field_names Expected = {"b", "a", ""};
#else
  const field_names Expected = {"", "a", "b"};
#endif

  ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
}

TEST(RANDSTRUCT_TEST, RandstructDoesNotRandomizeUnionFieldOrder) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    union test_union {
        int a;
        int b;
        int c;
        int d;
        int e;
        int f;
    } __attribute__((randomize_layout));
  )c");

  const RecordDecl *RD =
      getRecordDeclFromAST(AST->getASTContext(), "test_union");
  const field_names Expected = {"a", "b", "c", "d", "e", "f"};

  ASSERT_FALSE(RD->isRandomized());
  ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
}

TEST(RANDSTRUCT_TEST, AnonymousStructsAndUnionsRetainFieldOrder) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test_struct {
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

  const RecordDecl *RD =
      getRecordDeclFromAST(AST->getASTContext(), "test_struct");
#ifdef _WIN32
  const field_names Expected = {"", "s", "l", "", "r", "a", "f"};
#else
  const field_names Expected = {"f", "a", "l", "", "", "s", "r"};
#endif

  ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));

  bool AnonStructTested = false;
  bool AnonUnionTested = false;
  for (const Decl *D : RD->decls())
    if (const FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
      if (const auto *Record = FD->getType()->getAs<RecordType>()) {
        RD = Record->getDecl();
        if (RD->isAnonymousStructOrUnion()) {
          if (RD->isUnion()) {
            const field_names Expected = {"m", "n", "o", "p", "q"};

            ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
            AnonUnionTested = true;
          } else {
            const field_names Expected = {"g", "h", "i", "j", "k"};

            ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
            AnonStructTested = true;
          }
        } else if (RD->isStruct()) {
#ifdef _WIN32
          const field_names Expected = {"b", "c", "f", "d", "e"};
#else
          const field_names Expected = {"d", "e", "f", "c", "b"};
#endif
          ASSERT_EQ(Expected, getFieldNamesFromRecord(RD));
        }
      }
    }

  ASSERT_TRUE(AnonStructTested);
  ASSERT_TRUE(AnonUnionTested);
}

} // namespace ast_matchers
} // namespace clang
