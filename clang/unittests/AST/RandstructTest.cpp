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

static std::unique_ptr<ASTUnit> makeAST(const std::string &SourceCode) {
  std::vector<std::string> Args = getCommandLineArgsForTesting(Lang_C99);
  Args.push_back("-frandomize-layout-seed=1234567890abcdef");

  IgnoringDiagConsumer IgnoringConsumer = IgnoringDiagConsumer();

  return tooling::buildASTFromCodeWithArgs(
      SourceCode, Args, "input.c", "clang-tool",
      std::make_shared<PCHContainerOperations>(),
      tooling::getClangStripDependencyFileAdjuster(),
      tooling::FileContentMappings(), &IgnoringConsumer);
}

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

namespace clang {
namespace ast_matchers {

#define RANDSTRUCT_TEST_SUITE_TEST StructureLayoutRandomizationTestSuiteTest

TEST(RANDSTRUCT_TEST_SUITE_TEST, CanDetermineIfSubsequenceExists) {
  const field_names Seq = {"a", "b", "c", "d"};

  EXPECT_TRUE(isSubsequence(Seq, {"b", "c"}));
  EXPECT_TRUE(isSubsequence(Seq, {"a", "b", "c", "d"}));
  EXPECT_TRUE(isSubsequence(Seq, {"b", "c", "d"}));
  EXPECT_TRUE(isSubsequence(Seq, {"a"}));
  EXPECT_FALSE(isSubsequence(Seq, {"a", "d"}));
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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");

  EXPECT_FALSE(RD->hasAttr<RandomizeLayoutAttr>());
  EXPECT_FALSE(RD->isRandomized());
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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");

  EXPECT_TRUE(RD->hasAttr<NoRandomizeLayoutAttr>());
  EXPECT_FALSE(RD->isRandomized());
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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");

  EXPECT_TRUE(RD->hasAttr<RandomizeLayoutAttr>());
  EXPECT_TRUE(RD->isRandomized());
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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

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
  )c");

  EXPECT_TRUE(AST->getDiagnostics().hasErrorOccurred());

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
  )c");

  EXPECT_TRUE(AST->getDiagnostics().hasErrorOccurred());

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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");
  const field_names Actual = getFieldNamesFromRecord(RD);
  const field_names Subseq = {"x", "y", "z"};

  EXPECT_TRUE(RD->isRandomized());
  EXPECT_TRUE(isSubsequence(Actual, Subseq));
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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");

  EXPECT_TRUE(RD->isRandomized());
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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  // FIXME (?): calling getASTRecordLayout is probably a necessary evil so that
  // Clang's RecordBuilders can actually flesh out the information like
  // alignment, etc.
  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "test_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);

    EXPECT_TRUE(RD->isRandomized());
    EXPECT_EQ(19, Layout->getSize().getQuantity());
  }

  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "another_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);

    EXPECT_TRUE(RD->isRandomized());
    EXPECT_EQ(10, Layout->getSize().getQuantity());
  }

  {
    const RecordDecl *RD =
        getRecordDeclFromAST(AST->getASTContext(), "last_struct");
    const ASTRecordLayout *Layout =
        &AST->getASTContext().getASTRecordLayout(RD);

    EXPECT_TRUE(RD->isRandomized());
    EXPECT_EQ(9, Layout->getSize().getQuantity());
  }
}

TEST(RANDSTRUCT_TEST, ZeroWidthBitfieldsSeparateAllocationUnits) {
  const std::unique_ptr<ASTUnit> AST = makeAST(R"c(
    struct test {
        int a : 1;
        int   : 0;
        int b : 1;
    } __attribute__((randomize_layout));
  )c");

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD = getRecordDeclFromAST(AST->getASTContext(), "test");

  EXPECT_TRUE(RD->isRandomized());
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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD =
      getRecordDeclFromAST(AST->getASTContext(), "test_union");

  EXPECT_FALSE(RD->isRandomized());
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

  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  const RecordDecl *RD =
      getRecordDeclFromAST(AST->getASTContext(), "test_struct");

  EXPECT_TRUE(RD->isRandomized());

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

} // namespace ast_matchers
} // namespace clang
