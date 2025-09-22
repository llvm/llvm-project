//===- unittests/Analysis/FlowSensitive/RecordOpsTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/RecordOps.h"
#include "TestingSupport.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <string>

namespace clang {
namespace dataflow {
namespace test {
namespace {

void runDataflow(
    llvm::StringRef Code,
    std::function<llvm::StringMap<QualType>(QualType)> SyntheticFieldCallback,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults) {
  ASSERT_THAT_ERROR(checkDataflowWithNoopAnalysis(
                        Code, ast_matchers::hasName("target"), VerifyResults,
                        {BuiltinOptions()}, LangStandard::lang_cxx17,
                        SyntheticFieldCallback),
                    llvm::Succeeded());
}

const FieldDecl *getFieldNamed(RecordDecl *RD, llvm::StringRef Name) {
  for (const FieldDecl *FD : RD->fields())
    if (FD->getName() == Name)
      return FD;
  assert(false);
  return nullptr;
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
      [](QualType Ty) -> llvm::StringMap<QualType> {
        if (Ty.getAsString() != "S")
          return {};
        QualType IntTy =
            getFieldNamed(Ty->getAsRecordDecl(), "outer_int")->getType();
        return {{"synth_int", IntTy}};
      },
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
        EXPECT_NE(Env.getValue(S1.getSyntheticField("synth_int")),
                  Env.getValue(S2.getSyntheticField("synth_int")));

        copyRecord(S1, S2, Env);

        EXPECT_EQ(getFieldValue(&S1, *OuterIntDecl, Env),
                  getFieldValue(&S2, *OuterIntDecl, Env));
        EXPECT_EQ(S1.getChild(*RefDecl), S2.getChild(*RefDecl));
        EXPECT_EQ(getFieldValue(&Inner1, *InnerIntDecl, Env),
                  getFieldValue(&Inner2, *InnerIntDecl, Env));
        EXPECT_EQ(Env.getValue(S1.getSyntheticField("synth_int")),
                  Env.getValue(S2.getSyntheticField("synth_int")));
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
      [](QualType Ty) -> llvm::StringMap<QualType> {
        if (Ty.getAsString() != "S")
          return {};
        QualType IntTy =
            getFieldNamed(Ty->getAsRecordDecl(), "outer_int")->getType();
        return {{"synth_int", IntTy}};
      },
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

        Env.setValue(S1.getSyntheticField("synth_int"),
                     Env.create<IntegerValue>());

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

        // S2 has a different synth_int.
        Env.setValue(S2.getSyntheticField("synth_int"),
                     Env.create<IntegerValue>());
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));

        // S2 doesn't have a value for synth_int.
        Env.clearValue(S2.getSyntheticField("synth_int"));
        EXPECT_FALSE(recordsEqual(S1, S2, Env));
        copyRecord(S1, S2, Env);
        EXPECT_TRUE(recordsEqual(S1, S2, Env));
      });
}

TEST(RecordOpsTest, CopyRecordBetweenDerivedAndBase) {
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
  auto SyntheticFieldCallback = [](QualType Ty) -> llvm::StringMap<QualType> {
    CXXRecordDecl *ADecl = nullptr;
    if (Ty.getAsString() == "A")
      ADecl = Ty->getAsCXXRecordDecl();
    else if (Ty.getAsString() == "B")
      ADecl = Ty->getAsCXXRecordDecl()
                  ->bases_begin()
                  ->getType()
                  ->getAsCXXRecordDecl();
    else
      return {};
    QualType IntTy = getFieldNamed(ADecl, "i")->getType();
    return {{"synth_int", IntTy}};
  };
  // Test copying derived to base class.
  runDataflow(
      Code, SyntheticFieldCallback,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        Environment Env = getEnvironmentAtAnnotation(Results, "p").fork();

        const ValueDecl *IDecl = findValueDecl(ASTCtx, "i");
        auto &A = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "a");
        auto &B = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "b");

        EXPECT_NE(Env.getValue(*A.getChild(*IDecl)),
                  Env.getValue(*B.getChild(*IDecl)));
        EXPECT_NE(Env.getValue(A.getSyntheticField("synth_int")),
                  Env.getValue(B.getSyntheticField("synth_int")));

        copyRecord(B, A, Env);

        EXPECT_EQ(Env.getValue(*A.getChild(*IDecl)),
                  Env.getValue(*B.getChild(*IDecl)));
        EXPECT_EQ(Env.getValue(A.getSyntheticField("synth_int")),
                  Env.getValue(B.getSyntheticField("synth_int")));
      });
  // Test copying base to derived class.
  runDataflow(
      Code, SyntheticFieldCallback,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        Environment Env = getEnvironmentAtAnnotation(Results, "p").fork();

        const ValueDecl *IDecl = findValueDecl(ASTCtx, "i");
        auto &A = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "a");
        auto &B = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "b");

        EXPECT_NE(Env.getValue(*A.getChild(*IDecl)),
                  Env.getValue(*B.getChild(*IDecl)));
        EXPECT_NE(Env.getValue(A.getSyntheticField("synth_int")),
                  Env.getValue(B.getSyntheticField("synth_int")));

        copyRecord(A, B, Env);

        EXPECT_EQ(Env.getValue(*A.getChild(*IDecl)),
                  Env.getValue(*B.getChild(*IDecl)));
        EXPECT_EQ(Env.getValue(A.getSyntheticField("synth_int")),
                  Env.getValue(B.getSyntheticField("synth_int")));
      });
}

TEST(RecordOpsTest, CopyRecordWithExplicitSharedBaseTypeToCopy) {
  std::string Code = R"(
    struct Base {
      bool BaseField;
      char UnmodeledField;
    };

    struct DerivedOne : public Base {
      int DerivedOneField;
    };

    struct DerivedTwo : public Base {
      int DerivedTwoField;
    };

    void target(Base B, DerivedOne D1, DerivedTwo D2) {
      (void) B.BaseField;
      // [[p]]
    }
  )";
  auto SyntheticFieldCallback = [](QualType Ty) -> llvm::StringMap<QualType> {
    CXXRecordDecl *BaseDecl = nullptr;
    std::string TypeAsString = Ty.getAsString();
    if (TypeAsString == "Base")
      BaseDecl = Ty->getAsCXXRecordDecl();
    else if (TypeAsString == "DerivedOne" || TypeAsString == "DerivedTwo")
      BaseDecl = Ty->getAsCXXRecordDecl()
                     ->bases_begin()
                     ->getType()
                     ->getAsCXXRecordDecl();
    else
      return {};
    QualType FieldType = getFieldNamed(BaseDecl, "BaseField")->getType();
    return {{"synth_field", FieldType}};
  };
  // Test copying derived to base class.
  runDataflow(
      Code, SyntheticFieldCallback,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        Environment Env = getEnvironmentAtAnnotation(Results, "p").fork();

        const ValueDecl *BaseFieldDecl = findValueDecl(ASTCtx, "BaseField");
        auto &B = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "B");
        auto &D1 = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "D1");
        auto &D2 = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "D2");

        EXPECT_NE(Env.getValue(*D1.getChild(*BaseFieldDecl)),
                  Env.getValue(*D2.getChild(*BaseFieldDecl)));
        EXPECT_NE(Env.getValue(D1.getSyntheticField("synth_field")),
                  Env.getValue(D2.getSyntheticField("synth_field")));

        copyRecord(D1, D2, Env, B.getType());

        EXPECT_EQ(Env.getValue(*D1.getChild(*BaseFieldDecl)),
                  Env.getValue(*D2.getChild(*BaseFieldDecl)));
        EXPECT_EQ(Env.getValue(D1.getSyntheticField("synth_field")),
                  Env.getValue(D2.getSyntheticField("synth_field")));
      });
}

} // namespace
} // namespace test
} // namespace dataflow
} // namespace clang
