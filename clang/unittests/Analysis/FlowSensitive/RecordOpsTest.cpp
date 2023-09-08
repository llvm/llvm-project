//===- unittests/Analysis/FlowSensitive/RecordOpsTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/RecordOps.h"
#include "TestingSupport.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

namespace clang {
namespace dataflow {
namespace test {
namespace {

void runDataflow(
    llvm::StringRef Code,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults,
    LangStandard::Kind Std = LangStandard::lang_cxx17,
    llvm::StringRef TargetFun = "target") {
  ASSERT_THAT_ERROR(
      checkDataflowWithNoopAnalysis(Code, VerifyResults,
                                    DataflowAnalysisOptions{BuiltinOptions{}},
                                    Std, TargetFun),
      llvm::Succeeded());
}

TEST(RecordOpsTest, CopyRecord) {
  std::string Code = R"(
    struct S {
      int outer_int;
      int &ref;
      struct {
        int inner_int;
      } inner;
    };
    void target(S s1, S s2) {
      (void)s1.outer_int;
      (void)s1.ref;
      (void)s1.inner.inner_int;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        Environment Env = getEnvironmentAtAnnotation(Results, "p").fork();

        const ValueDecl *OuterIntDecl = findValueDecl(ASTCtx, "outer_int");
        const ValueDecl *RefDecl = findValueDecl(ASTCtx, "ref");
        const ValueDecl *InnerDecl = findValueDecl(ASTCtx, "inner");
        const ValueDecl *InnerIntDecl = findValueDecl(ASTCtx, "inner_int");

        auto &S1 = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "s1");
        auto &S2 = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "s2");
        auto &Inner1 = *cast<RecordStorageLocation>(S1.getChild(*InnerDecl));
        auto &Inner2 = *cast<RecordStorageLocation>(S2.getChild(*InnerDecl));

        EXPECT_NE(getFieldValue(&S1, *OuterIntDecl, Env),
                  getFieldValue(&S2, *OuterIntDecl, Env));
        EXPECT_NE(S1.getChild(*RefDecl), S2.getChild(*RefDecl));
        EXPECT_NE(getFieldValue(&Inner1, *InnerIntDecl, Env),
                  getFieldValue(&Inner2, *InnerIntDecl, Env));

        auto *S1Val = cast<RecordValue>(Env.getValue(S1));
        auto *S2Val = cast<RecordValue>(Env.getValue(S2));
        EXPECT_NE(S1Val, S2Val);

        S1Val->setProperty("prop", Env.getBoolLiteralValue(true));

        copyRecord(S1, S2, Env);

        EXPECT_EQ(getFieldValue(&S1, *OuterIntDecl, Env),
                  getFieldValue(&S2, *OuterIntDecl, Env));
        EXPECT_EQ(S1.getChild(*RefDecl), S2.getChild(*RefDecl));
        EXPECT_EQ(getFieldValue(&Inner1, *InnerIntDecl, Env),
                  getFieldValue(&Inner2, *InnerIntDecl, Env));

        S1Val = cast<RecordValue>(Env.getValue(S1));
        S2Val = cast<RecordValue>(Env.getValue(S2));
        EXPECT_NE(S1Val, S2Val);

        EXPECT_EQ(S2Val->getProperty("prop"), &Env.getBoolLiteralValue(true));
      });
}

TEST(RecordOpsTest, RecordsEqual) {
  std::string Code = R"(
    struct S {
      int outer_int;
      int &ref;
      struct {
        int inner_int;
      } inner;
    };
    void target(S s1, S s2) {
      (void)s1.outer_int;
      (void)s1.ref;
      (void)s1.inner.inner_int;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        Environment Env = getEnvironmentAtAnnotation(Results, "p").fork();

        const ValueDecl *OuterIntDecl = findValueDecl(ASTCtx, "outer_int");
        const ValueDecl *RefDecl = findValueDecl(ASTCtx, "ref");
        const ValueDecl *InnerDecl = findValueDecl(ASTCtx, "inner");
        const ValueDecl *InnerIntDecl = findValueDecl(ASTCtx, "inner_int");

        auto &S1 = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "s1");
        auto &S2 = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "s2");
        auto &Inner2 = *cast<RecordStorageLocation>(S2.getChild(*InnerDecl));

        cast<RecordValue>(Env.getValue(S1))
            ->setProperty("prop", Env.getBoolLiteralValue(true));

        // Strategy: Create two equal records, then verify each of the various
        // ways in which records can differ causes recordsEqual to return false.
        // changes we can make to the record.

        // This test reuses the same objects for multiple checks, which isn't
        // great, but seems better than duplicating the setup code for every
        // check.

        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S2 has a different outer_int.
        Env.setValue(*S2.getChild(*OuterIntDecl), Env.create<IntegerValue>());
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S2 doesn't have outer_int at all.
        Env.clearValue(*S2.getChild(*OuterIntDecl));
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S2 has a different ref.
        S2.setChild(*RefDecl, &Env.createStorageLocation(
                                  RefDecl->getType().getNonReferenceType()));
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S2 as a different inner_int.
        Env.setValue(*Inner2.getChild(*InnerIntDecl),
                     Env.create<IntegerValue>());
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S1 and S2 have the same property with different values.
        cast<RecordValue>(Env.getValue(S2))
            ->setProperty("prop", Env.getBoolLiteralValue(false));
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S1 has a property that S2 doesn't have.
        cast<RecordValue>(Env.getValue(S1))
            ->setProperty("other_prop", Env.getBoolLiteralValue(false));
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        // We modified S1 this time, so need to copy back the other way.
        copyRecord(S2, S1, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S2 has a property that S1 doesn't have.
        cast<RecordValue>(Env.getValue(S2))
            ->setProperty("other_prop", Env.getBoolLiteralValue(false));
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S1 and S2 have the same number of properties, but with different
        // names.
        cast<RecordValue>(Env.getValue(S1))
            ->setProperty("prop1", Env.getBoolLiteralValue(false));
        cast<RecordValue>(Env.getValue(S2))
            ->setProperty("prop2", Env.getBoolLiteralValue(false));
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
      });
}

TEST(TransferTest, CopyRecordFromDerivedToBase) {
  std::string Code = R"(
    struct A {
      int i;
    };

    struct B : public A {
    };

    void target(A a, B b) {
      (void)a.i;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        Environment Env = getEnvironmentAtAnnotation(Results, "p").fork();

        const ValueDecl *IDecl = findValueDecl(ASTCtx, "i");
        auto &A = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "a");
        auto &B = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "b");

        EXPECT_NE(Env.getValue(*A.getChild(*IDecl)),
                  Env.getValue(*B.getChild(*IDecl)));

        copyRecord(B, A, Env);

        EXPECT_EQ(Env.getValue(*A.getChild(*IDecl)),
                  Env.getValue(*B.getChild(*IDecl)));
      });
}

} // namespace
} // namespace test
} // namespace dataflow
} // namespace clang
