//===- unittests/Analysis/FlowSensitive/TransferTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Analysis/FlowSensitive/RecordOps.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/LangStandard.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>
#include <string>
#include <utility>

namespace {

using namespace clang;
using namespace dataflow;
using namespace test;
using ::testing::Eq;
using ::testing::IsNull;
using ::testing::Ne;
using ::testing::NotNull;
using ::testing::UnorderedElementsAre;

void runDataflow(
    llvm::StringRef Code,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults,
    DataflowAnalysisOptions Options,
    LangStandard::Kind Std = LangStandard::lang_cxx17,
    llvm::StringRef TargetFun = "target") {
  ASSERT_THAT_ERROR(checkDataflowWithNoopAnalysis(Code, VerifyResults, Options,
                                                  Std, TargetFun),
                    llvm::Succeeded());
}

void runDataflow(
    llvm::StringRef Code,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults,
    LangStandard::Kind Std = LangStandard::lang_cxx17,
    bool ApplyBuiltinTransfer = true, llvm::StringRef TargetFun = "target") {
  runDataflow(Code, std::move(VerifyResults),
              {ApplyBuiltinTransfer ? BuiltinOptions{}
                                    : std::optional<BuiltinOptions>()},
              Std, TargetFun);
}

void runDataflowOnLambda(
    llvm::StringRef Code,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults,
    DataflowAnalysisOptions Options,
    LangStandard::Kind Std = LangStandard::lang_cxx17) {
  ASSERT_THAT_ERROR(
      checkDataflowWithNoopAnalysis(
          Code,
          ast_matchers::hasDeclContext(
              ast_matchers::cxxRecordDecl(ast_matchers::isLambda())),
          VerifyResults, Options, Std),
      llvm::Succeeded());
}

void runDataflowOnLambda(
    llvm::StringRef Code,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults,
    LangStandard::Kind Std = LangStandard::lang_cxx17,
    bool ApplyBuiltinTransfer = true) {
  runDataflowOnLambda(Code, std::move(VerifyResults),
                      {ApplyBuiltinTransfer ? BuiltinOptions{}
                                            : std::optional<BuiltinOptions>()},
                      Std);
}

const Formula &getFormula(const ValueDecl &D, const Environment &Env) {
  return cast<BoolValue>(Env.getValue(D))->formula();
}

TEST(TransferTest, CNotSupported) {
  std::string Code = R"(
    void target() {}
  )";
  ASSERT_THAT_ERROR(checkDataflowWithNoopAnalysis(
                        Code, [](const auto &, auto &) {}, {BuiltinOptions{}},
                        LangStandard::lang_c89),
                    llvm::FailedWithMessage("Can only analyze C++"));
}

TEST(TransferTest, IntVarDeclNotTrackedWhenTransferDisabled) {
  std::string Code = R"(
    void target() {
      int Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        EXPECT_EQ(Env.getStorageLocation(*FooDecl), nullptr);
      },
      LangStandard::lang_cxx17,
      /*ApplyBuiltinTransfer=*/false);
}

TEST(TransferTest, BoolVarDecl) {
  std::string Code = R"(
    void target() {
      bool Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<BoolValue>(FooVal));
      });
}

TEST(TransferTest, IntVarDecl) {
  std::string Code = R"(
    void target() {
      int Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));
      });
}

TEST(TransferTest, StructIncomplete) {
  std::string Code = R"(
    struct A;

    void target() {
      A* Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        auto *FooValue = dyn_cast_or_null<PointerValue>(Env.getValue(*FooDecl));
        ASSERT_THAT(FooValue, NotNull());

        EXPECT_TRUE(isa<RecordStorageLocation>(FooValue->getPointeeLoc()));
        auto *FooPointeeValue = Env.getValue(FooValue->getPointeeLoc());
        ASSERT_THAT(FooPointeeValue, NotNull());
        EXPECT_TRUE(isa<RecordValue>(FooPointeeValue));
      });
}

// As a memory optimization, we prevent modeling fields nested below a certain
// level (currently, depth 3). This test verifies this lack of modeling. We also
// include a regression test for the case that the unmodeled field is a
// reference to a struct; previously, we crashed when accessing such a field.
TEST(TransferTest, StructFieldUnmodeled) {
  std::string Code = R"(
    struct S { int X; };
    S GlobalS;
    struct A { S &Unmodeled = GlobalS; };
    struct B { A F3; };
    struct C { B F2; };
    struct D { C F1; };

    void target() {
      D Bar;
      A &Foo = Bar.F1.F2.F3;
      int Zab = Foo.Unmodeled.X;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        QualType FooReferentType = FooDecl->getType()->getPointeeType();
        ASSERT_TRUE(FooReferentType->isStructureType());
        auto FooFields = FooReferentType->getAsRecordDecl()->fields();

        FieldDecl *UnmodeledDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Unmodeled") {
            UnmodeledDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(UnmodeledDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *UnmodeledLoc = FooLoc->getChild(*UnmodeledDecl);
        ASSERT_TRUE(isa<RecordStorageLocation>(UnmodeledLoc));
        EXPECT_THAT(Env.getValue(*UnmodeledLoc), IsNull());

        const ValueDecl *ZabDecl = findValueDecl(ASTCtx, "Zab");
        ASSERT_THAT(ZabDecl, NotNull());
        EXPECT_THAT(Env.getValue(*ZabDecl), NotNull());
      });
}

TEST(TransferTest, StructVarDecl) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target() {
      A Foo;
      (void)Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        EXPECT_TRUE(isa<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env)));
      });
}

TEST(TransferTest, StructVarDeclWithInit) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    A Gen();

    void target() {
      A Foo = Gen();
      (void)Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        EXPECT_TRUE(isa<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env)));
      });
}

TEST(TransferTest, StructArrayVarDecl) {
  std::string Code = R"(
    struct A {};

    void target() {
      A Array[2];
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *ArrayDecl = findValueDecl(ASTCtx, "Array");

        // We currently don't create values for arrays.
        ASSERT_THAT(Env.getValue(*ArrayDecl), IsNull());
      });
}

TEST(TransferTest, ClassVarDecl) {
  std::string Code = R"(
    class A {
     public:
      int Bar;
    };

    void target() {
      A Foo;
      (void)Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isClassType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        EXPECT_TRUE(isa<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env)));
      });
}

TEST(TransferTest, ReferenceVarDecl) {
  std::string Code = R"(
    struct A {};

    A &getA();

    void target() {
      A &Foo = getA();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<RecordStorageLocation>(FooLoc));

        const Value *FooReferentVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<RecordValue>(FooReferentVal));
      });
}

TEST(TransferTest, SelfReferentialReferenceVarDecl) {
  std::string Code = R"(
    struct A;

    struct B {};

    struct C {
      A &FooRef;
      A *FooPtr;
      B &BazRef;
      B *BazPtr;
    };

    struct A {
      C &Bar;
    };

    A &getA();

    void target() {
      A &Foo = getA();
      (void)Foo.Bar.FooRef;
      (void)Foo.Bar.FooPtr;
      (void)Foo.Bar.BazRef;
      (void)Foo.Bar.BazPtr;
      // [[p]]
    }
  )";
  runDataflow(Code, [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                           &Results,
                       ASTContext &ASTCtx) {
    ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
    const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    ASSERT_TRUE(FooDecl->getType()->isReferenceType());
    ASSERT_TRUE(FooDecl->getType().getNonReferenceType()->isStructureType());
    const auto FooFields =
        FooDecl->getType().getNonReferenceType()->getAsRecordDecl()->fields();

    FieldDecl *BarDecl = nullptr;
    for (FieldDecl *Field : FooFields) {
      if (Field->getNameAsString() == "Bar") {
        BarDecl = Field;
      } else {
        FAIL() << "Unexpected field: " << Field->getNameAsString();
      }
    }
    ASSERT_THAT(BarDecl, NotNull());

    ASSERT_TRUE(BarDecl->getType()->isReferenceType());
    ASSERT_TRUE(BarDecl->getType().getNonReferenceType()->isStructureType());
    const auto BarFields =
        BarDecl->getType().getNonReferenceType()->getAsRecordDecl()->fields();

    FieldDecl *FooRefDecl = nullptr;
    FieldDecl *FooPtrDecl = nullptr;
    FieldDecl *BazRefDecl = nullptr;
    FieldDecl *BazPtrDecl = nullptr;
    for (FieldDecl *Field : BarFields) {
      if (Field->getNameAsString() == "FooRef") {
        FooRefDecl = Field;
      } else if (Field->getNameAsString() == "FooPtr") {
        FooPtrDecl = Field;
      } else if (Field->getNameAsString() == "BazRef") {
        BazRefDecl = Field;
      } else if (Field->getNameAsString() == "BazPtr") {
        BazPtrDecl = Field;
      } else {
        FAIL() << "Unexpected field: " << Field->getNameAsString();
      }
    }
    ASSERT_THAT(FooRefDecl, NotNull());
    ASSERT_THAT(FooPtrDecl, NotNull());
    ASSERT_THAT(BazRefDecl, NotNull());
    ASSERT_THAT(BazPtrDecl, NotNull());

    const auto &FooLoc =
        *cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));

    const auto &BarLoc =
        *cast<RecordStorageLocation>(FooLoc.getChild(*BarDecl));

    const auto &FooReferentLoc =
        *cast<RecordStorageLocation>(BarLoc.getChild(*FooRefDecl));
    EXPECT_THAT(Env.getValue(FooReferentLoc), NotNull());
    EXPECT_THAT(getFieldValue(&FooReferentLoc, *BarDecl, Env), IsNull());

    const auto &FooPtrVal =
        *cast<PointerValue>(getFieldValue(&BarLoc, *FooPtrDecl, Env));
    const auto &FooPtrPointeeLoc =
        cast<RecordStorageLocation>(FooPtrVal.getPointeeLoc());
    EXPECT_THAT(Env.getValue(FooPtrPointeeLoc), NotNull());
    EXPECT_THAT(getFieldValue(&FooPtrPointeeLoc, *BarDecl, Env), IsNull());

    EXPECT_THAT(getFieldValue(&BarLoc, *BazRefDecl, Env), NotNull());

    const auto &BazPtrVal =
        *cast<PointerValue>(getFieldValue(&BarLoc, *BazPtrDecl, Env));
    const StorageLocation &BazPtrPointeeLoc = BazPtrVal.getPointeeLoc();
    EXPECT_THAT(Env.getValue(BazPtrPointeeLoc), NotNull());
  });
}

TEST(TransferTest, PointerVarDecl) {
  std::string Code = R"(
    struct A {};

    A *getA();

    void target() {
      A *Foo = getA();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const PointerValue *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        EXPECT_TRUE(isa<RecordStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        EXPECT_TRUE(isa_and_nonnull<RecordValue>(FooPointeeVal));
      });
}

TEST(TransferTest, SelfReferentialPointerVarDecl) {
  std::string Code = R"(
    struct A;

    struct B {};

    struct C {
      A &FooRef;
      A *FooPtr;
      B &BazRef;
      B *BazPtr;
    };

    struct A {
      C *Bar;
    };

    A *getA();

    void target() {
      A *Foo = getA();
      (void)Foo->Bar->FooRef;
      (void)Foo->Bar->FooPtr;
      (void)Foo->Bar->BazRef;
      (void)Foo->Bar->BazPtr;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isPointerType());
        ASSERT_TRUE(FooDecl->getType()
                        ->getAs<PointerType>()
                        ->getPointeeType()
                        ->isStructureType());
        const auto FooFields = FooDecl->getType()
                                   ->getAs<PointerType>()
                                   ->getPointeeType()
                                   ->getAsRecordDecl()
                                   ->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        ASSERT_TRUE(BarDecl->getType()->isPointerType());
        ASSERT_TRUE(BarDecl->getType()
                        ->getAs<PointerType>()
                        ->getPointeeType()
                        ->isStructureType());
        const auto BarFields = BarDecl->getType()
                                   ->getAs<PointerType>()
                                   ->getPointeeType()
                                   ->getAsRecordDecl()
                                   ->fields();

        FieldDecl *FooRefDecl = nullptr;
        FieldDecl *FooPtrDecl = nullptr;
        FieldDecl *BazRefDecl = nullptr;
        FieldDecl *BazPtrDecl = nullptr;
        for (FieldDecl *Field : BarFields) {
          if (Field->getNameAsString() == "FooRef") {
            FooRefDecl = Field;
          } else if (Field->getNameAsString() == "FooPtr") {
            FooPtrDecl = Field;
          } else if (Field->getNameAsString() == "BazRef") {
            BazRefDecl = Field;
          } else if (Field->getNameAsString() == "BazPtr") {
            BazPtrDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(FooRefDecl, NotNull());
        ASSERT_THAT(FooPtrDecl, NotNull());
        ASSERT_THAT(BazRefDecl, NotNull());
        ASSERT_THAT(BazPtrDecl, NotNull());

        const auto &FooLoc =
            *cast<ScalarStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto &FooVal = *cast<PointerValue>(Env.getValue(FooLoc));
        const auto &FooPointeeLoc =
            cast<RecordStorageLocation>(FooVal.getPointeeLoc());

        const auto &BarVal =
            *cast<PointerValue>(getFieldValue(&FooPointeeLoc, *BarDecl, Env));
        const auto &BarPointeeLoc =
            cast<RecordStorageLocation>(BarVal.getPointeeLoc());

        EXPECT_THAT(getFieldValue(&BarPointeeLoc, *FooRefDecl, Env), NotNull());

        const auto &FooPtrVal = *cast<PointerValue>(
            getFieldValue(&BarPointeeLoc, *FooPtrDecl, Env));
        const auto &FooPtrPointeeLoc =
            cast<RecordStorageLocation>(FooPtrVal.getPointeeLoc());
        EXPECT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

        EXPECT_THAT(getFieldValue(&BarPointeeLoc, *BazRefDecl, Env), NotNull());

        const auto &BazPtrVal = *cast<PointerValue>(
            getFieldValue(&BarPointeeLoc, *BazPtrDecl, Env));
        const StorageLocation &BazPtrPointeeLoc = BazPtrVal.getPointeeLoc();
        EXPECT_THAT(Env.getValue(BazPtrPointeeLoc), NotNull());
      });
}

TEST(TransferTest, DirectlySelfReferentialReference) {
  std::string Code = R"(
    struct target {
      target() {
        (void)0;
        // [[p]]
      }
      target &self = *this;
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        const ValueDecl *SelfDecl = findValueDecl(ASTCtx, "self");

        auto *ThisLoc = Env.getThisPointeeStorageLocation();
        ASSERT_EQ(ThisLoc->getChild(*SelfDecl), ThisLoc);
      });
}

TEST(TransferTest, MultipleVarsDecl) {
  std::string Code = R"(
    void target() {
      int Foo, Bar;
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const Value *BarVal = Env.getValue(*BarLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));
      });
}

TEST(TransferTest, JoinVarDecl) {
  std::string Code = R"(
    void target(bool B) {
      int Foo;
      // [[p1]]
      if (B) {
        int Bar;
        // [[p2]]
      } else {
        int Baz;
        // [[p3]]
      }
      (void)0;
      // [[p4]]
    }
  )";
  runDataflow(Code, [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                           &Results,
                       ASTContext &ASTCtx) {
    ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2", "p3", "p4"));

    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
    ASSERT_THAT(BarDecl, NotNull());

    const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
    ASSERT_THAT(BazDecl, NotNull());

    const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");

    const StorageLocation *FooLoc = Env1.getStorageLocation(*FooDecl);
    EXPECT_THAT(FooLoc, NotNull());
    EXPECT_THAT(Env1.getStorageLocation(*BarDecl), IsNull());
    EXPECT_THAT(Env1.getStorageLocation(*BazDecl), IsNull());

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    EXPECT_EQ(Env2.getStorageLocation(*FooDecl), FooLoc);
    EXPECT_THAT(Env2.getStorageLocation(*BarDecl), NotNull());
    EXPECT_THAT(Env2.getStorageLocation(*BazDecl), IsNull());

    const Environment &Env3 = getEnvironmentAtAnnotation(Results, "p3");
    EXPECT_EQ(Env3.getStorageLocation(*FooDecl), FooLoc);
    EXPECT_THAT(Env3.getStorageLocation(*BarDecl), IsNull());
    EXPECT_THAT(Env3.getStorageLocation(*BazDecl), NotNull());

    const Environment &Env4 = getEnvironmentAtAnnotation(Results, "p4");
    EXPECT_EQ(Env4.getStorageLocation(*FooDecl), FooLoc);
    EXPECT_THAT(Env4.getStorageLocation(*BarDecl), IsNull());
    EXPECT_THAT(Env4.getStorageLocation(*BazDecl), IsNull());
  });
}

TEST(TransferTest, BinaryOperatorAssign) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar;
      (Bar) = (Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Value *FooVal = Env.getValue(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BarDecl), FooVal);
      });
}

TEST(TransferTest, BinaryOperatorAssignIntegerLiteral) {
  std::string Code = R"(
    void target() {
      int Foo = 1;
      // [[before]]
      Foo = 2;
      // [[after]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Before =
            getEnvironmentAtAnnotation(Results, "before");
        const Environment &After = getEnvironmentAtAnnotation(Results, "after");

        const auto &ValBefore =
            getValueForDecl<IntegerValue>(ASTCtx, Before, "Foo");
        const auto &ValAfter =
            getValueForDecl<IntegerValue>(ASTCtx, After, "Foo");
        EXPECT_NE(&ValBefore, &ValAfter);
      });
}

TEST(TransferTest, VarDeclInitAssign) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Value *FooVal = Env.getValue(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BarDecl), FooVal);
      });
}

TEST(TransferTest, VarDeclInitAssignChained) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar;
      int Baz = (Bar = Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Value *FooVal = Env.getValue(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BarDecl), FooVal);
        EXPECT_EQ(Env.getValue(*BazDecl), FooVal);
      });
}

TEST(TransferTest, VarDeclInitAssignPtrDeref) {
  std::string Code = R"(
    void target() {
      int Foo;
      int *Bar;
      *(Bar) = Foo;
      int Baz = *(Bar);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Value *FooVal = Env.getValue(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarVal = cast<PointerValue>(Env.getValue(*BarDecl));
        EXPECT_EQ(Env.getValue(BarVal->getPointeeLoc()), FooVal);

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl), FooVal);
      });
}

TEST(TransferTest, AssignToAndFromReference) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar;
      int &Baz = Foo;
      // [[p1]]
      Baz = Bar;
      int Qux = Baz;
      int &Quux = Baz;
      // [[p2]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Value *FooVal = Env1.getValue(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const Value *BarVal = Env1.getValue(*BarDecl);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env1.getValue(*BazDecl), FooVal);

        EXPECT_EQ(Env2.getValue(*BazDecl), BarVal);
        EXPECT_EQ(Env2.getValue(*FooDecl), BarVal);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());
        EXPECT_EQ(Env2.getValue(*QuxDecl), BarVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env2.getValue(*QuuxDecl), BarVal);
      });
}

TEST(TransferTest, MultipleParamDecls) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));
      });
}

TEST(TransferTest, StructParamDecl) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target(A Foo) {
      (void)Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        EXPECT_TRUE(isa<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env)));
      });
}

TEST(TransferTest, ReferenceParamDecl) {
  std::string Code = R"(
    struct A {};

    void target(A &Foo) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<RecordStorageLocation>(FooLoc));

        const Value *FooReferentVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<RecordValue>(FooReferentVal));
      });
}

TEST(TransferTest, PointerParamDecl) {
  std::string Code = R"(
    struct A {};

    void target(A *Foo) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const PointerValue *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        EXPECT_TRUE(isa<RecordStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        EXPECT_TRUE(isa_and_nonnull<RecordValue>(FooPointeeVal));
      });
}

TEST(TransferTest, StructMember) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target(A Foo) {
      int Baz = Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *BarVal =
            cast<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl), BarVal);
      });
}

TEST(TransferTest, StructMemberEnum) {
  std::string Code = R"(
    struct A {
      int Bar;
      enum E { ONE, TWO };
    };

    void target(A Foo) {
      A::E Baz = Foo.ONE;
      // [[p]]
    }
  )";
  // Minimal expectations -- we're just testing that it doesn't crash, since
  // enums aren't interpreted.
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        EXPECT_THAT(Results.keys(), UnorderedElementsAre("p"));
      });
}

TEST(TransferTest, DerivedBaseMemberClass) {
  std::string Code = R"(
    class A {
      int ADefault;
    protected:
      int AProtected;
    private:
      int APrivate;
    public:
      int APublic;

    private:
      friend void target();
    };

    class B : public A {
      int BDefault;
    protected:
      int BProtected;
    private:
      int BPrivate;

    private:
      friend void target();
    };

    void target() {
      B Foo;
      (void)Foo.ADefault;
      (void)Foo.AProtected;
      (void)Foo.APrivate;
      (void)Foo.APublic;
      (void)Foo.BDefault;
      (void)Foo.BProtected;
      (void)Foo.BPrivate;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        ASSERT_TRUE(FooDecl->getType()->isRecordType());

        // Derived-class fields.
        const FieldDecl *BDefaultDecl = nullptr;
        const FieldDecl *BProtectedDecl = nullptr;
        const FieldDecl *BPrivateDecl = nullptr;
        for (const FieldDecl *Field :
             FooDecl->getType()->getAsRecordDecl()->fields()) {
          if (Field->getNameAsString() == "BDefault") {
            BDefaultDecl = Field;
          } else if (Field->getNameAsString() == "BProtected") {
            BProtectedDecl = Field;
          } else if (Field->getNameAsString() == "BPrivate") {
            BPrivateDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BDefaultDecl, NotNull());
        ASSERT_THAT(BProtectedDecl, NotNull());
        ASSERT_THAT(BPrivateDecl, NotNull());

        // Base-class fields.
        const FieldDecl *ADefaultDecl = nullptr;
        const FieldDecl *APrivateDecl = nullptr;
        const FieldDecl *AProtectedDecl = nullptr;
        const FieldDecl *APublicDecl = nullptr;
        for (const clang::CXXBaseSpecifier &Base :
             FooDecl->getType()->getAsCXXRecordDecl()->bases()) {
          QualType BaseType = Base.getType();
          ASSERT_TRUE(BaseType->isRecordType());
          for (const FieldDecl *Field : BaseType->getAsRecordDecl()->fields()) {
            if (Field->getNameAsString() == "ADefault") {
              ADefaultDecl = Field;
            } else if (Field->getNameAsString() == "AProtected") {
              AProtectedDecl = Field;
            } else if (Field->getNameAsString() == "APrivate") {
              APrivateDecl = Field;
            } else if (Field->getNameAsString() == "APublic") {
              APublicDecl = Field;
            } else {
              FAIL() << "Unexpected field: " << Field->getNameAsString();
            }
          }
        }
        ASSERT_THAT(ADefaultDecl, NotNull());
        ASSERT_THAT(AProtectedDecl, NotNull());
        ASSERT_THAT(APrivateDecl, NotNull());
        ASSERT_THAT(APublicDecl, NotNull());

        ASSERT_TRUE(
            isa<RecordStorageLocation>(Env.getStorageLocation(*FooDecl)));
      });
}

static void derivedBaseMemberExpectations(
    const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
    ASTContext &ASTCtx) {
  ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
  const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

  const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
  ASSERT_THAT(FooDecl, NotNull());

  ASSERT_TRUE(FooDecl->getType()->isRecordType());
  const FieldDecl *BarDecl = nullptr;
  for (const clang::CXXBaseSpecifier &Base :
       FooDecl->getType()->getAsCXXRecordDecl()->bases()) {
    QualType BaseType = Base.getType();
    ASSERT_TRUE(BaseType->isStructureType());

    for (const FieldDecl *Field : BaseType->getAsRecordDecl()->fields()) {
      if (Field->getNameAsString() == "Bar") {
        BarDecl = Field;
      } else {
        FAIL() << "Unexpected field: " << Field->getNameAsString();
      }
    }
  }
  ASSERT_THAT(BarDecl, NotNull());

  const auto &FooLoc =
      *cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
  const auto &FooVal = *cast<RecordValue>(Env.getValue(FooLoc));
  EXPECT_EQ(&FooVal.getLoc(), &FooLoc);
}

TEST(TransferTest, DerivedBaseMemberStructDefault) {
  std::string Code = R"(
    struct A {
      int Bar;
    };
    struct B : public A {
    };

    void target() {
      B Foo;
      (void)Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(Code, derivedBaseMemberExpectations);
}

TEST(TransferTest, DerivedBaseMemberPrivateFriend) {
  // Include an access to `Foo.Bar` to verify the analysis doesn't crash on that
  // access.
  std::string Code = R"(
    struct A {
    private:
      friend void target();
      int Bar;
    };
    struct B : public A {
    };

    void target() {
      B Foo;
      (void)Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(Code, derivedBaseMemberExpectations);
}

TEST(TransferTest, ClassMember) {
  std::string Code = R"(
    class A {
    public:
      int Bar;
    };

    void target(A Foo) {
      int Baz = Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isClassType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *BarVal =
            cast<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl), BarVal);
      });
}

TEST(TransferTest, BaseClassInitializer) {
  using ast_matchers::cxxConstructorDecl;
  using ast_matchers::hasName;
  using ast_matchers::ofClass;

  std::string Code = R"(
    class A {
    public:
      A(int I) : Bar(I) {}
      int Bar;
    };

    class B : public A {
    public:
      B(int I) : A(I) {
        (void)0;
        // [[p]]
      }
    };
  )";
  ASSERT_THAT_ERROR(
      checkDataflow<NoopAnalysis>(
          AnalysisInputs<NoopAnalysis>(
              Code, cxxConstructorDecl(ofClass(hasName("B"))),
              [](ASTContext &C, Environment &) { return NoopAnalysis(C); })
              .withASTBuildArgs(
                  {"-fsyntax-only", "-fno-delayed-template-parsing",
                   "-std=" + std::string(LangStandard::getLangStandardForKind(
                                             LangStandard::lang_cxx17)
                                             .getName())}),
          /*VerifyResults=*/
          [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
             const AnalysisOutputs &) {
            // Regression test to verify that base-class initializers do not
            // trigger an assertion. If we add support for such initializers in
            // the future, we can expand this test to check more specific
            // properties.
            EXPECT_THAT(Results.keys(), UnorderedElementsAre("p"));
          }),
      llvm::Succeeded());
}

TEST(TransferTest, FieldsDontHaveValuesInConstructor) {
  // In a constructor, unlike in regular member functions, we don't want fields
  // to be pre-initialized with values, because doing so is the job of the
  // constructor.
  std::string Code = R"(
    struct target {
      target() {
        0;
        // [[p]]
        // Mention the field so it is modeled;
        Val;
      }

      int Val;
    };
 )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        EXPECT_EQ(getFieldValue(Env.getThisPointeeStorageLocation(), "Val",
                                ASTCtx, Env),
                  nullptr);
      });
}

TEST(TransferTest, FieldsDontHaveValuesInConstructorWithBaseClass) {
  // See above, but for a class with a base class.
  std::string Code = R"(
    struct Base {
        int BaseVal;
    };

    struct target  : public Base {
      target() {
        0;
        // [[p]]
        // Mention the fields so they are modeled.
        BaseVal;
        Val;
      }

      int Val;
    };
 )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        // FIXME: The field of the base class should already have been
        // initialized with a value by the base constructor. This test documents
        // the current buggy behavior.
        EXPECT_EQ(getFieldValue(Env.getThisPointeeStorageLocation(), "BaseVal",
                                ASTCtx, Env),
                  nullptr);
        EXPECT_EQ(getFieldValue(Env.getThisPointeeStorageLocation(), "Val",
                                ASTCtx, Env),
                  nullptr);
      });
}

TEST(TransferTest, StructModeledFieldsWithAccessor) {
  std::string Code = R"(
    class S {
      int *Ptr;
      int *PtrNonConst;
      int Int;
      int IntWithInc;
      int IntNotAccessed;
      int IntRef;
    public:
      int *getPtr() const { return Ptr; }
      int *getPtrNonConst() { return PtrNonConst; }
      int getInt(int i) const { return Int; }
      int getWithInc(int i) { IntWithInc += i; return IntWithInc; }
      int getIntNotAccessed() const { return IntNotAccessed; }
      int getIntNoDefinition() const;
      int &getIntRef() { return IntRef; }
      void returnVoid() const { return; }
    };

    void target() {
      S s;
      int *p1 = s.getPtr();
      int *p2 = s.getPtrNonConst();
      int i1 = s.getInt(1);
      int i2 = s.getWithInc(1);
      int i3 = s.getIntNoDefinition();
      int &iref = s.getIntRef();

      // Regression test: Don't crash on an indirect call (which doesn't have
      // an associated `CXXMethodDecl`).
      auto ptr_to_member_fn = &S::getPtr;
      p1 = (s.*ptr_to_member_fn)();

      // Regression test: Don't crash on a return statement without a value.
      s.returnVoid();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env =
              getEnvironmentAtAnnotation(Results, "p");
        auto &SLoc = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "s");
        std::vector<const ValueDecl*> Fields;
        for (auto [Field, _] : SLoc.children())
          Fields.push_back(Field);
        // Only the fields that have simple accessor methods (that have a
        // single statement body that returns the member variable) should be
        // modeled.
        ASSERT_THAT(Fields, UnorderedElementsAre(
            findValueDecl(ASTCtx, "Ptr"), findValueDecl(ASTCtx, "PtrNonConst"),
            findValueDecl(ASTCtx, "Int"), findValueDecl(ASTCtx, "IntRef")));
      });
}

TEST(TransferTest, StructModeledFieldsWithComplicatedInheritance) {
  std::string Code = R"(
    struct Base1 {
      int base1_1;
      int base1_2;
    };
    struct Intermediate : Base1 {
      int intermediate_1;
      int intermediate_2;
    };
    struct Base2 {
      int base2_1;
      int base2_2;
    };
    struct MostDerived : public Intermediate, Base2 {
      int most_derived_1;
      int most_derived_2;
    };

    void target() {
      MostDerived MD;
      MD.base1_2 = 1;
      MD.intermediate_2 = 1;
      MD.base2_2 = 1;
      MD.most_derived_2 = 1;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env =
              getEnvironmentAtAnnotation(Results, "p");

        // Only the accessed fields should exist in the model.
        auto &MDLoc = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "MD");
        std::vector<const ValueDecl*> Fields;
        for (auto [Field, _] : MDLoc.children())
          Fields.push_back(Field);
        ASSERT_THAT(Fields, UnorderedElementsAre(
            findValueDecl(ASTCtx, "base1_2"),
            findValueDecl(ASTCtx, "intermediate_2"),
            findValueDecl(ASTCtx, "base2_2"),
            findValueDecl(ASTCtx, "most_derived_2")));
      });
}

TEST(TransferTest, StructInitializerListWithComplicatedInheritance) {
  std::string Code = R"(
    struct Base1 {
      int base1;
    };
    struct Intermediate : Base1 {
      int intermediate;
    };
    struct Base2 {
      int base2;
    };
    struct MostDerived : public Intermediate, Base2 {
      int most_derived;
    };

    void target() {
      MostDerived MD = {};
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env =
              getEnvironmentAtAnnotation(Results, "p");

        // When a struct is initialized with a initializer list, all the
        // fields are considered "accessed", and therefore do exist.
        auto &MD = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "MD");
        ASSERT_THAT(cast<IntegerValue>(
            getFieldValue(&MD, *findValueDecl(ASTCtx, "base1"), Env)),
            NotNull());
        ASSERT_THAT(cast<IntegerValue>(
            getFieldValue(&MD, *findValueDecl(ASTCtx, "intermediate"), Env)),
            NotNull());
        ASSERT_THAT(cast<IntegerValue>(
            getFieldValue(&MD, *findValueDecl(ASTCtx, "base2"), Env)),
            NotNull());
        ASSERT_THAT(cast<IntegerValue>(
            getFieldValue(&MD, *findValueDecl(ASTCtx, "most_derived"), Env)),
            NotNull());
      });
}

TEST(TransferTest, ReferenceMember) {
  std::string Code = R"(
    struct A {
      int &Bar;
    };

    void target(A Foo) {
      int Baz = Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *BarReferentVal =
            cast<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl), BarReferentVal);
      });
}

TEST(TransferTest, StructThisMember) {
  std::string Code = R"(
    struct A {
      int Bar;

      struct B {
        int Baz;
      };

      B Qux;

      void target() {
        int Foo = Bar;
        int Quux = Qux.Baz;
        // [[p]]
      }
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const auto *ThisLoc = Env.getThisPointeeStorageLocation();
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl), BarVal);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        ASSERT_TRUE(QuxDecl->getType()->isStructureType());
        auto QuxFields = QuxDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BazDecl = nullptr;
        for (FieldDecl *Field : QuxFields) {
          if (Field->getNameAsString() == "Baz") {
            BazDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BazDecl, NotNull());

        const auto *QuxLoc =
            cast<RecordStorageLocation>(ThisLoc->getChild(*QuxDecl));
        EXPECT_THAT(dyn_cast<RecordValue>(Env.getValue(*QuxLoc)), NotNull());

        const auto *BazVal =
            cast<IntegerValue>(getFieldValue(QuxLoc, *BazDecl, Env));

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuuxDecl), BazVal);
      });
}

TEST(TransferTest, ClassThisMember) {
  std::string Code = R"(
    class A {
      int Bar;

      class B {
      public:
        int Baz;
      };

      B Qux;

      void target() {
        int Foo = Bar;
        int Quux = Qux.Baz;
        // [[p]]
      }
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const auto *ThisLoc = Env.getThisPointeeStorageLocation();

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl), BarVal);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        ASSERT_TRUE(QuxDecl->getType()->isClassType());
        auto QuxFields = QuxDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BazDecl = nullptr;
        for (FieldDecl *Field : QuxFields) {
          if (Field->getNameAsString() == "Baz") {
            BazDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BazDecl, NotNull());

        const auto *QuxLoc =
            cast<RecordStorageLocation>(ThisLoc->getChild(*QuxDecl));
        EXPECT_THAT(dyn_cast<RecordValue>(Env.getValue(*QuxLoc)), NotNull());

        const auto *BazVal =
            cast<IntegerValue>(getFieldValue(QuxLoc, *BazDecl, Env));

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuuxDecl), BazVal);
      });
}

TEST(TransferTest, UnionThisMember) {
  std::string Code = R"(
    union A {
      int Foo;
      int Bar;

      void target() {
        A a;
        // Mention the fields to ensure they're included in the analysis.
        (void)a.Foo;
        (void)a.Bar;
        // [[p]]
      }
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const auto *ThisLoc = Env.getThisPointeeStorageLocation();
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooLoc =
            cast<ScalarStorageLocation>(ThisLoc->getChild(*FooDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));
      });
}

TEST(TransferTest, StructThisInLambda) {
  std::string ThisCaptureCode = R"(
    struct A {
      void frob() {
        [this]() {
          int Foo = Bar;
          // [[p1]]
        }();
      }

      int Bar;
    };
  )";
  runDataflow(
      ThisCaptureCode,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p1");

        const auto *ThisLoc = Env.getThisPointeeStorageLocation();
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl), BarVal);
      },
      LangStandard::lang_cxx17, /*ApplyBuiltinTransfer=*/true, "operator()");

  std::string RefCaptureDefaultCode = R"(
    struct A {
      void frob() {
        [&]() {
          int Foo = Bar;
          // [[p2]]
        }();
      }

      int Bar;
    };
  )";
  runDataflow(
      RefCaptureDefaultCode,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p2"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p2");

        const auto *ThisLoc = Env.getThisPointeeStorageLocation();
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl), BarVal);
      },
      LangStandard::lang_cxx17, /*ApplyBuiltinTransfer=*/true, "operator()");

  std::string FreeFunctionLambdaCode = R"(
    void foo() {
      int Bar;
      [&]() {
        int Foo = Bar;
        // [[p3]]
      }();
    }
  )";
  runDataflow(
      FreeFunctionLambdaCode,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p3"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p3");

        EXPECT_THAT(Env.getThisPointeeStorageLocation(), IsNull());
      },
      LangStandard::lang_cxx17, /*ApplyBuiltinTransfer=*/true, "operator()");
}

TEST(TransferTest, ConstructorInitializer) {
  std::string Code = R"(
    struct target {
      int Bar;

      target(int Foo) : Bar(Foo) {
        int Qux = Bar;
        // [[p]]
      }
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const auto *ThisLoc = Env.getThisPointeeStorageLocation();
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooVal = cast<IntegerValue>(Env.getValue(*FooDecl));

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuxDecl), FooVal);
      });
}

TEST(TransferTest, DefaultInitializer) {
  std::string Code = R"(
    struct target {
      int Bar;
      int Baz = Bar;

      target(int Foo) : Bar(Foo) {
        int Qux = Baz;
        // [[p]]
      }
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const auto *ThisLoc = Env.getThisPointeeStorageLocation();
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooVal = cast<IntegerValue>(Env.getValue(*FooDecl));

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuxDecl), FooVal);
      });
}

TEST(TransferTest, DefaultInitializerReference) {
  std::string Code = R"(
    struct target {
      int &Bar;
      int &Baz = Bar;

      target(int &Foo) : Bar(Foo) {
        int &Qux = Baz;
        // [[p]]
      }
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const auto *ThisLoc = Env.getThisPointeeStorageLocation();
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooLoc = Env.getStorageLocation(*FooDecl);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        const auto *QuxLoc = Env.getStorageLocation(*QuxDecl);
        EXPECT_EQ(QuxLoc, FooLoc);
      });
}

TEST(TransferTest, TemporaryObject) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target() {
      A Foo = A();
      (void)Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        EXPECT_TRUE(isa<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env)));
      });
}

TEST(TransferTest, ElidableConstructor) {
  // This test is effectively the same as TransferTest.TemporaryObject, but
  // the code is compiled as C++14.
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target() {
      A Foo = A();
      (void)Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        EXPECT_TRUE(isa<IntegerValue>(getFieldValue(FooLoc, *BarDecl, Env)));
      },
      LangStandard::lang_cxx14);
}

TEST(TransferTest, AssignmentOperator) {
  std::string Code = R"(
    struct A {
      int Baz;
    };

    void target() {
      A Foo = { 1 };
      A Bar = { 2 };
      // [[p1]]
      Foo = Bar;
      // [[p2]]
      Foo.Baz = 3;
      // [[p3]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        // Before copy assignment.
        {
          const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");

          const auto *FooLoc1 =
              cast<RecordStorageLocation>(Env1.getStorageLocation(*FooDecl));
          const auto *BarLoc1 =
              cast<RecordStorageLocation>(Env1.getStorageLocation(*BarDecl));
          EXPECT_FALSE(recordsEqual(*FooLoc1, *BarLoc1, Env1));

          const auto *FooBazVal1 =
              cast<IntegerValue>(getFieldValue(FooLoc1, *BazDecl, Env1));
          const auto *BarBazVal1 =
              cast<IntegerValue>(getFieldValue(BarLoc1, *BazDecl, Env1));
          EXPECT_NE(FooBazVal1, BarBazVal1);
        }

        // After copy assignment.
        {
          const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

          const auto *FooLoc2 =
              cast<RecordStorageLocation>(Env2.getStorageLocation(*FooDecl));
          const auto *BarLoc2 =
              cast<RecordStorageLocation>(Env2.getStorageLocation(*BarDecl));

          const auto *FooVal2 = cast<RecordValue>(Env2.getValue(*FooLoc2));
          const auto *BarVal2 = cast<RecordValue>(Env2.getValue(*BarLoc2));
          EXPECT_NE(FooVal2, BarVal2);

          EXPECT_TRUE(recordsEqual(*FooLoc2, *BarLoc2, Env2));

          const auto *FooBazVal2 =
              cast<IntegerValue>(getFieldValue(FooLoc2, *BazDecl, Env2));
          const auto *BarBazVal2 =
              cast<IntegerValue>(getFieldValue(BarLoc2, *BazDecl, Env2));
          EXPECT_EQ(FooBazVal2, BarBazVal2);
        }

        // After value update.
        {
          const Environment &Env3 = getEnvironmentAtAnnotation(Results, "p3");

          const auto *FooLoc3 =
              cast<RecordStorageLocation>(Env3.getStorageLocation(*FooDecl));
          const auto *BarLoc3 =
              cast<RecordStorageLocation>(Env3.getStorageLocation(*BarDecl));
          EXPECT_FALSE(recordsEqual(*FooLoc3, *BarLoc3, Env3));

          const auto *FooBazVal3 =
              cast<IntegerValue>(getFieldValue(FooLoc3, *BazDecl, Env3));
          const auto *BarBazVal3 =
              cast<IntegerValue>(getFieldValue(BarLoc3, *BazDecl, Env3));
          EXPECT_NE(FooBazVal3, BarBazVal3);
        }
      });
}

// It's legal for the assignment operator to take its source parameter by value.
// Check that we handle this correctly. (This is a repro -- we used to
// assert-fail on this.)
TEST(TransferTest, AssignmentOperator_ArgByValue) {
  std::string Code = R"(
    struct A {
      int Baz;
      A &operator=(A);
    };

    void target() {
      A Foo = { 1 };
      A Bar = { 2 };
      Foo = Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");

        const auto &FooLoc =
            getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "Foo");
        const auto &BarLoc =
            getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "Bar");

        const auto *FooBazVal =
            cast<IntegerValue>(getFieldValue(&FooLoc, *BazDecl, Env));
        const auto *BarBazVal =
            cast<IntegerValue>(getFieldValue(&BarLoc, *BazDecl, Env));
        EXPECT_EQ(FooBazVal, BarBazVal);
      });
}

TEST(TransferTest, AssignmentOperatorFromBase) {
  // This is a crash repro. We don't model the copy this case, so no
  // expectations on the copied field of the base class are checked.
  std::string Code = R"(
    struct Base {
      int base;
    };
    struct Derived : public Base {
      using Base::operator=;
      int derived;
    };
    void target(Base B, Derived D) {
      D.base = 1;
      D.derived = 1;
      D = B;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {});
}

TEST(TransferTest, AssignmentOperatorFromCallResult) {
  std::string Code = R"(
    struct A {};
    A ReturnA();

    void target() {
      A MyA;
      MyA = ReturnA();
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        // As of this writing, we don't produce a `Value` for the call
        // `ReturnA()`. The only condition we're testing for is that the
        // analysis should not crash in this case.
      });
}

TEST(TransferTest, AssignmentOperatorWithInitAndInheritance) {
  // This is a crash repro.
  std::string Code = R"(
    struct B { int Foo; };
    struct S : public B {};
    void target() {
      S S1 = { 1 };
      S S2;
      S S3;
      S1 = S2;  // Only Dst has InitListExpr.
      S3 = S1;  // Only Src has InitListExpr.
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {});
}

TEST(TransferTest, AssignmentOperatorReturnsVoid) {
  // This is a crash repro.
  std::string Code = R"(
    struct S {
      void operator=(S&& other);
    };
    void target() {
      S s;
      s = S();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {});
}

TEST(TransferTest, AssignmentOperatorReturnsByValue) {
  // This is a crash repro.
  std::string Code = R"(
    struct S {
      S operator=(S&& other);
    };
    void target() {
      S s;
      s = S();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {});
}

TEST(TransferTest, InitListExprAsXValue) {
  // This is a crash repro.
  std::string Code = R"(
    void target() {
      bool&& Foo{false};
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        const auto &FooVal = getValueForDecl<BoolValue>(ASTCtx, Env, "Foo");
        ASSERT_TRUE(FooVal.formula().isLiteral(false));
      });
}

TEST(TransferTest, ArrayInitListExprOneRecordElement) {
  // This is a crash repro.
  std::string Code = R"cc(
    struct S {};

    void target() { S foo[] = {S()}; }
  )cc";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        // Just verify that it doesn't crash.
      });
}

TEST(TransferTest, InitListExprAsUnion) {
  // This is a crash repro.
  std::string Code = R"cc(
    class target {
      union {
        int *a;
        bool *b;
      } F;

     public:
      constexpr target() : F{nullptr} {
        int *null = nullptr;
        F.b;  // Make sure we reference 'b' so it is modeled.
        // [[p]]
      }
    };
  )cc";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto &FLoc = getFieldLoc<RecordStorageLocation>(
            *Env.getThisPointeeStorageLocation(), "F", ASTCtx);
        auto *AVal = cast<PointerValue>(getFieldValue(&FLoc, "a", ASTCtx, Env));
        EXPECT_EQ(AVal, &getValueForDecl<PointerValue>(ASTCtx, Env, "null"));
        EXPECT_EQ(getFieldValue(&FLoc, "b", ASTCtx, Env), nullptr);
      });
}

TEST(TransferTest, EmptyInitListExprForUnion) {
  // This is a crash repro.
  std::string Code = R"cc(
    class target {
      union {
        int *a;
        bool *b;
      } F;

     public:
      // Empty initializer list means that `F` is aggregate-initialized.
      // For a union, this has the effect that the first member of the union
      // is copy-initialized from an empty initializer list; in this specific
      // case, this has the effect of initializing `a` with null.
      constexpr target() : F{} {
        int *null = nullptr;
        F.b;  // Make sure we reference 'b' so it is modeled.
        // [[p]]
      }
    };
  )cc";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto &FLoc = getFieldLoc<RecordStorageLocation>(
            *Env.getThisPointeeStorageLocation(), "F", ASTCtx);
        auto *AVal = cast<PointerValue>(getFieldValue(&FLoc, "a", ASTCtx, Env));
        EXPECT_EQ(AVal, &getValueForDecl<PointerValue>(ASTCtx, Env, "null"));
        EXPECT_EQ(getFieldValue(&FLoc, "b", ASTCtx, Env), nullptr);
      });
}

TEST(TransferTest, EmptyInitListExprForStruct) {
  std::string Code = R"cc(
    class target {
      struct {
        int *a;
        bool *b;
      } F;

     public:
      constexpr target() : F{} {
        int *NullIntPtr = nullptr;
        bool *NullBoolPtr = nullptr;
        // [[p]]
      }
    };
  )cc";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto &FLoc = getFieldLoc<RecordStorageLocation>(
            *Env.getThisPointeeStorageLocation(), "F", ASTCtx);
        auto *AVal = cast<PointerValue>(getFieldValue(&FLoc, "a", ASTCtx, Env));
        EXPECT_EQ(AVal,
                  &getValueForDecl<PointerValue>(ASTCtx, Env, "NullIntPtr"));
        auto *BVal = cast<PointerValue>(getFieldValue(&FLoc, "b", ASTCtx, Env));
        EXPECT_EQ(BVal,
                  &getValueForDecl<PointerValue>(ASTCtx, Env, "NullBoolPtr"));
      });
}

TEST(TransferTest, CopyConstructor) {
  std::string Code = R"(
    struct A {
      int Baz;
    };

    void target() {
      A Foo = { 1 };
      A Bar = Foo;
      // [[after_copy]]
      Foo.Baz = 2;
      // [[after_update]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        // after_copy
        {
          const Environment &Env =
              getEnvironmentAtAnnotation(Results, "after_copy");

          const auto *FooLoc =
              cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
          const auto *BarLoc =
              cast<RecordStorageLocation>(Env.getStorageLocation(*BarDecl));

          // `Foo` and `Bar` have different `RecordValue`s associated with them.
          const auto *FooVal = cast<RecordValue>(Env.getValue(*FooLoc));
          const auto *BarVal = cast<RecordValue>(Env.getValue(*BarLoc));
          EXPECT_NE(FooVal, BarVal);

          // But the records compare equal.
          EXPECT_TRUE(recordsEqual(*FooLoc, *BarLoc, Env));

          // In particular, the value of `Baz` in both records is the same.
          const auto *FooBazVal =
              cast<IntegerValue>(getFieldValue(FooLoc, *BazDecl, Env));
          const auto *BarBazVal =
              cast<IntegerValue>(getFieldValue(BarLoc, *BazDecl, Env));
          EXPECT_EQ(FooBazVal, BarBazVal);
        }

        // after_update
        {
          const Environment &Env =
              getEnvironmentAtAnnotation(Results, "after_update");

          const auto *FooLoc =
              cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
          const auto *BarLoc =
              cast<RecordStorageLocation>(Env.getStorageLocation(*BarDecl));

          EXPECT_FALSE(recordsEqual(*FooLoc, *BarLoc, Env));

          const auto *FooBazVal =
              cast<IntegerValue>(getFieldValue(FooLoc, *BazDecl, Env));
          const auto *BarBazVal =
              cast<IntegerValue>(getFieldValue(BarLoc, *BazDecl, Env));
          EXPECT_NE(FooBazVal, BarBazVal);
        }
      });
}

TEST(TransferTest, CopyConstructorWithDefaultArgument) {
  std::string Code = R"(
    struct A {
      int Baz;
      A() = default;
      A(const A& a, bool def = true) { Baz = a.Baz; }
    };

    void target() {
      A Foo;
      (void)Foo.Baz;
      A Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *BarLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*BarDecl));
        EXPECT_TRUE(recordsEqual(*FooLoc, *BarLoc, Env));

        const auto *FooBazVal =
            cast<IntegerValue>(getFieldValue(FooLoc, *BazDecl, Env));
        const auto *BarBazVal =
            cast<IntegerValue>(getFieldValue(BarLoc, *BazDecl, Env));
        EXPECT_EQ(FooBazVal, BarBazVal);
      });
}

TEST(TransferTest, CopyConstructorWithParens) {
  std::string Code = R"(
    struct A {
      int Baz;
    };

    void target() {
      A Foo;
      (void)Foo.Baz;
      A Bar((A(Foo)));
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *BarLoc =
            cast<RecordStorageLocation>(Env.getStorageLocation(*BarDecl));
        EXPECT_TRUE(recordsEqual(*FooLoc, *BarLoc, Env));

        const auto *FooBazVal =
            cast<IntegerValue>(getFieldValue(FooLoc, *BazDecl, Env));
        const auto *BarBazVal =
            cast<IntegerValue>(getFieldValue(BarLoc, *BazDecl, Env));
        EXPECT_EQ(FooBazVal, BarBazVal);
      });
}

TEST(TransferTest, CopyConstructorWithInitializerListAsSyntacticSugar) {
  std::string Code = R"(
  struct A {
    int Baz;
  };
  void target() {
    A Foo = {3};
    (void)Foo.Baz;
    A Bar = {A(Foo)};
    // [[p]]
  }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");

        const auto &FooLoc =
            getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "Foo");
        const auto &BarLoc =
            getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "Bar");

        const auto *FooBazVal =
            cast<IntegerValue>(getFieldValue(&FooLoc, *BazDecl, Env));
        const auto *BarBazVal =
            cast<IntegerValue>(getFieldValue(&BarLoc, *BazDecl, Env));
        EXPECT_EQ(FooBazVal, BarBazVal);
      });
}

TEST(TransferTest, CopyConstructorArgIsRefReturnedByFunction) {
  // This is a crash repro.
  std::string Code = R"(
    struct S {};
    const S &returnsSRef();
    void target() {
      S s(returnsSRef());
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {});
}

TEST(TransferTest, MoveConstructor) {
  std::string Code = R"(
    namespace std {

    template <typename T> struct remove_reference      { using type = T; };
    template <typename T> struct remove_reference<T&>  { using type = T; };
    template <typename T> struct remove_reference<T&&> { using type = T; };

    template <typename T>
    using remove_reference_t = typename remove_reference<T>::type;

    template <typename T>
    std::remove_reference_t<T>&& move(T&& x);

    } // namespace std

    struct A {
      int Baz;
    };

    void target() {
      A Foo;
      A Bar;
      (void)Foo.Baz;
      // [[p1]]
      Foo = std::move(Bar);
      // [[p2]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooLoc1 =
            cast<RecordStorageLocation>(Env1.getStorageLocation(*FooDecl));
        const auto *BarLoc1 =
            cast<RecordStorageLocation>(Env1.getStorageLocation(*BarDecl));

        EXPECT_FALSE(recordsEqual(*FooLoc1, *BarLoc1, Env1));

        const auto *FooVal1 = cast<RecordValue>(Env1.getValue(*FooLoc1));
        const auto *BarVal1 = cast<RecordValue>(Env1.getValue(*BarLoc1));
        EXPECT_NE(FooVal1, BarVal1);

        const auto *FooBazVal1 =
            cast<IntegerValue>(getFieldValue(FooLoc1, *BazDecl, Env1));
        const auto *BarBazVal1 =
            cast<IntegerValue>(getFieldValue(BarLoc1, *BazDecl, Env1));
        EXPECT_NE(FooBazVal1, BarBazVal1);

        const auto *FooLoc2 =
            cast<RecordStorageLocation>(Env2.getStorageLocation(*FooDecl));
        const auto *FooVal2 = cast<RecordValue>(Env2.getValue(*FooLoc2));
        EXPECT_NE(FooVal2, BarVal1);
        EXPECT_TRUE(recordsEqual(*FooLoc2, Env2, *BarLoc1, Env1));

        const auto *FooBazVal2 =
            cast<IntegerValue>(getFieldValue(FooLoc1, *BazDecl, Env2));
        EXPECT_EQ(FooBazVal2, BarBazVal1);
      });
}

TEST(TransferTest, BindTemporary) {
  std::string Code = R"(
    struct A {
      virtual ~A() = default;

      int Baz;
    };

    void target(A Foo) {
      int Bar = A(Foo).Baz;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto &FooLoc =
            *cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *BarVal = cast<IntegerValue>(Env.getValue(*BarDecl));
        EXPECT_EQ(BarVal, getFieldValue(&FooLoc, *BazDecl, Env));
      });
}

TEST(TransferTest, ResultObjectLocation) {
  std::string Code = R"(
    struct A {
      virtual ~A() = default;
    };

    void target() {
      0, A();
      (void)0; // [[p]]
    }
  )";
  using ast_matchers::binaryOperator;
  using ast_matchers::cxxBindTemporaryExpr;
  using ast_matchers::cxxTemporaryObjectExpr;
  using ast_matchers::exprWithCleanups;
  using ast_matchers::has;
  using ast_matchers::hasOperatorName;
  using ast_matchers::hasRHS;
  using ast_matchers::match;
  using ast_matchers::selectFirst;
  using ast_matchers::traverse;
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        // The expression `0, A()` in the code above produces the following
        // structure, consisting of four prvalues of record type.
        // `Env.getResultObjectLocation()` should return the same location for
        // all of these.
        auto MatchResult = match(
            traverse(TK_AsIs,
                     exprWithCleanups(
                         has(binaryOperator(
                                 hasOperatorName(","),
                                 hasRHS(cxxBindTemporaryExpr(
                                            has(cxxTemporaryObjectExpr().bind(
                                                "toe")))
                                            .bind("bte")))
                                 .bind("comma")))
                         .bind("ewc")),
            ASTCtx);
        auto *TOE = selectFirst<CXXTemporaryObjectExpr>("toe", MatchResult);
        ASSERT_NE(TOE, nullptr);
        auto *Comma = selectFirst<BinaryOperator>("comma", MatchResult);
        ASSERT_NE(Comma, nullptr);
        auto *EWC = selectFirst<ExprWithCleanups>("ewc", MatchResult);
        ASSERT_NE(EWC, nullptr);
        auto *BTE = selectFirst<CXXBindTemporaryExpr>("bte", MatchResult);
        ASSERT_NE(BTE, nullptr);

        RecordStorageLocation &Loc = Env.getResultObjectLocation(*TOE);
        EXPECT_EQ(&Loc, &Env.getResultObjectLocation(*Comma));
        EXPECT_EQ(&Loc, &Env.getResultObjectLocation(*EWC));
        EXPECT_EQ(&Loc, &Env.getResultObjectLocation(*BTE));
      });
}

TEST(TransferTest, ResultObjectLocationForDefaultInitExpr) {
  std::string Code = R"(
    struct S {};
    struct target {
      target () {
        (void)0;
        // [[p]]
      }
      S s = {};
    };
  )";

  using ast_matchers::cxxCtorInitializer;
  using ast_matchers::match;
  using ast_matchers::selectFirst;
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *SField = findValueDecl(ASTCtx, "s");

        auto *CtorInit = selectFirst<CXXCtorInitializer>(
            "ctor_initializer",
            match(cxxCtorInitializer().bind("ctor_initializer"), ASTCtx));
        ASSERT_NE(CtorInit, nullptr);

        auto *DefaultInit = cast<CXXDefaultInitExpr>(CtorInit->getInit());

        RecordStorageLocation &Loc = Env.getResultObjectLocation(*DefaultInit);

        // FIXME: The result object location for the `CXXDefaultInitExpr` should
        // be the location of the member variable being initialized, but we
        // don't do this correctly yet; see also comments in
        // `builtinTransferInitializer()`.
        // For the time being, we just document the current erroneous behavior
        // here (this should be `EXPECT_EQ` when the behavior is fixed).
        EXPECT_NE(&Loc, Env.getThisPointeeStorageLocation()->getChild(*SField));
      });
}

// This test ensures that CXXOperatorCallExpr returning prvalues are correctly
// handled by the transfer functions, especially that `getResultObjectLocation`
// correctly returns a storage location for those.
TEST(TransferTest, ResultObjectLocationForCXXOperatorCallExpr) {
  std::string Code = R"(
    struct A {
      A operator+(int);
    };

    void target() {
      A a;
      a + 3;
      (void)0; // [[p]]
    }
  )";
  using ast_matchers::cxxOperatorCallExpr;
  using ast_matchers::match;
  using ast_matchers::selectFirst;
  using ast_matchers::traverse;
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto *CallExpr = selectFirst<CXXOperatorCallExpr>(
            "call_expr",
            match(cxxOperatorCallExpr().bind("call_expr"), ASTCtx));

        EXPECT_NE(&Env.getResultObjectLocation(*CallExpr), nullptr);
      });
}

TEST(TransferTest, StaticCast) {
  std::string Code = R"(
    void target(int Foo) {
      int Bar = static_cast<int>(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooVal = Env.getValue(*FooDecl);
        const auto *BarVal = Env.getValue(*BarDecl);
        EXPECT_TRUE(isa<IntegerValue>(FooVal));
        EXPECT_TRUE(isa<IntegerValue>(BarVal));
        EXPECT_EQ(FooVal, BarVal);
      });
}

TEST(TransferTest, IntegralCast) {
  std::string Code = R"(
    void target(int Foo) {
      long Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooVal = Env.getValue(*FooDecl);
        const auto *BarVal = Env.getValue(*BarDecl);
        EXPECT_TRUE(isa<IntegerValue>(FooVal));
        EXPECT_TRUE(isa<IntegerValue>(BarVal));
        EXPECT_EQ(FooVal, BarVal);
      });
}

TEST(TransferTest, IntegraltoBooleanCast) {
  std::string Code = R"(
    void target(int Foo) {
      bool Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooVal = Env.getValue(*FooDecl);
        const auto *BarVal = Env.getValue(*BarDecl);
        EXPECT_TRUE(isa<IntegerValue>(FooVal));
        EXPECT_TRUE(isa<BoolValue>(BarVal));
      });
}

TEST(TransferTest, IntegralToBooleanCastFromBool) {
  std::string Code = R"(
    void target(bool Foo) {
      int Zab = Foo;
      bool Bar = Zab;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooVal = Env.getValue(*FooDecl);
        const auto *BarVal = Env.getValue(*BarDecl);
        EXPECT_TRUE(isa<BoolValue>(FooVal));
        EXPECT_TRUE(isa<BoolValue>(BarVal));
        EXPECT_EQ(FooVal, BarVal);
      });
}

TEST(TransferTest, NullToPointerCast) {
  std::string Code = R"(
    using my_nullptr_t = decltype(nullptr);
    struct Baz {};
    void target() {
      int *FooX = nullptr;
      int *FooY = nullptr;
      bool **Bar = nullptr;
      Baz *Baz = nullptr;
      my_nullptr_t Null = 0;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooXDecl = findValueDecl(ASTCtx, "FooX");
        ASSERT_THAT(FooXDecl, NotNull());

        const ValueDecl *FooYDecl = findValueDecl(ASTCtx, "FooY");
        ASSERT_THAT(FooYDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const ValueDecl *NullDecl = findValueDecl(ASTCtx, "Null");
        ASSERT_THAT(NullDecl, NotNull());

        const auto *FooXVal = cast<PointerValue>(Env.getValue(*FooXDecl));
        const auto *FooYVal = cast<PointerValue>(Env.getValue(*FooYDecl));
        const auto *BarVal = cast<PointerValue>(Env.getValue(*BarDecl));
        const auto *BazVal = cast<PointerValue>(Env.getValue(*BazDecl));
        const auto *NullVal = cast<PointerValue>(Env.getValue(*NullDecl));

        EXPECT_EQ(FooXVal, FooYVal);
        EXPECT_NE(FooXVal, BarVal);
        EXPECT_NE(FooXVal, BazVal);
        EXPECT_NE(BarVal, BazVal);

        const StorageLocation &FooPointeeLoc = FooXVal->getPointeeLoc();
        EXPECT_TRUE(isa<ScalarStorageLocation>(FooPointeeLoc));
        EXPECT_THAT(Env.getValue(FooPointeeLoc), IsNull());

        const StorageLocation &BarPointeeLoc = BarVal->getPointeeLoc();
        EXPECT_TRUE(isa<ScalarStorageLocation>(BarPointeeLoc));
        EXPECT_THAT(Env.getValue(BarPointeeLoc), IsNull());

        const StorageLocation &BazPointeeLoc = BazVal->getPointeeLoc();
        EXPECT_TRUE(isa<RecordStorageLocation>(BazPointeeLoc));
        EXPECT_THAT(Env.getValue(BazPointeeLoc), IsNull());

        const StorageLocation &NullPointeeLoc = NullVal->getPointeeLoc();
        EXPECT_TRUE(isa<ScalarStorageLocation>(NullPointeeLoc));
        EXPECT_THAT(Env.getValue(NullPointeeLoc), IsNull());
      });
}

TEST(TransferTest, PointerToMemberVariable) {
  std::string Code = R"(
    struct S {
      int i;
    };
    void target() {
      int S::*MemberPointer = &S::i;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *MemberPointerDecl =
            findValueDecl(ASTCtx, "MemberPointer");
        ASSERT_THAT(MemberPointerDecl, NotNull());
        ASSERT_THAT(Env.getValue(*MemberPointerDecl), IsNull());
      });
}

TEST(TransferTest, PointerToMemberFunction) {
  std::string Code = R"(
    struct S {
      void Method();
    };
    void target() {
      void (S::*MemberPointer)() = &S::Method;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *MemberPointerDecl =
            findValueDecl(ASTCtx, "MemberPointer");
        ASSERT_THAT(MemberPointerDecl, NotNull());
        ASSERT_THAT(Env.getValue(*MemberPointerDecl), IsNull());
      });
}

TEST(TransferTest, NullToMemberPointerCast) {
  std::string Code = R"(
    struct Foo {};
    void target() {
      int Foo::*MemberPointer = nullptr;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *MemberPointerDecl =
            findValueDecl(ASTCtx, "MemberPointer");
        ASSERT_THAT(MemberPointerDecl, NotNull());
        ASSERT_THAT(Env.getValue(*MemberPointerDecl), IsNull());
      });
}

TEST(TransferTest, AddrOfValue) {
  std::string Code = R"(
    void target() {
      int Foo;
      int *Bar = &Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<ScalarStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *BarVal = cast<PointerValue>(Env.getValue(*BarDecl));
        EXPECT_EQ(&BarVal->getPointeeLoc(), FooLoc);
      });
}

TEST(TransferTest, AddrOfReference) {
  std::string Code = R"(
    void target(int *Foo) {
      int *Bar = &(*Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooVal = cast<PointerValue>(Env.getValue(*FooDecl));
        const auto *BarVal = cast<PointerValue>(Env.getValue(*BarDecl));
        EXPECT_EQ(&BarVal->getPointeeLoc(), &FooVal->getPointeeLoc());
      });
}

TEST(TransferTest, CannotAnalyzeFunctionTemplate) {
  std::string Code = R"(
    template <typename T>
    void target() {}
  )";
  ASSERT_THAT_ERROR(
      checkDataflowWithNoopAnalysis(Code),
      llvm::FailedWithMessage("Cannot analyze templated declarations"));
}

TEST(TransferTest, CannotAnalyzeMethodOfClassTemplate) {
  std::string Code = R"(
    template <typename T>
    struct A {
      void target() {}
    };
  )";
  ASSERT_THAT_ERROR(
      checkDataflowWithNoopAnalysis(Code),
      llvm::FailedWithMessage("Cannot analyze templated declarations"));
}

TEST(TransferTest, VarDeclInitAssignConditionalOperator) {
  std::string Code = R"(
    struct A {};

    void target(A Foo, A Bar, bool Cond) {
      A Baz = Cond ?  Foo : Bar;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooVal = cast<RecordValue>(Env.getValue(*FooDecl));
        const auto *BarVal = cast<RecordValue>(Env.getValue(*BarDecl));

        const auto *BazVal = dyn_cast<RecordValue>(Env.getValue(*BazDecl));
        ASSERT_THAT(BazVal, NotNull());

        EXPECT_NE(BazVal, FooVal);
        EXPECT_NE(BazVal, BarVal);
      });
}

TEST(TransferTest, VarDeclInDoWhile) {
  std::string Code = R"(
    void target(int *Foo) {
      do {
        int Bar = *Foo;
        // [[in_loop]]
      } while (false);
      (void)0;
      // [[after_loop]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &EnvInLoop =
            getEnvironmentAtAnnotation(Results, "in_loop");
        const Environment &EnvAfterLoop =
            getEnvironmentAtAnnotation(Results, "after_loop");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooVal =
            cast<PointerValue>(EnvAfterLoop.getValue(*FooDecl));
        const auto *FooPointeeVal =
            cast<IntegerValue>(EnvAfterLoop.getValue(FooVal->getPointeeLoc()));

        const auto *BarVal = cast<IntegerValue>(EnvInLoop.getValue(*BarDecl));
        EXPECT_EQ(BarVal, FooPointeeVal);

        ASSERT_THAT(EnvAfterLoop.getValue(*BarDecl), IsNull());
      });
}

TEST(TransferTest, UnreachableAfterWhileTrue) {
  std::string Code = R"(
    void target() {
      while (true) {}
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        // The node after the while-true is pruned because it is trivially
        // known to be unreachable.
        ASSERT_TRUE(Results.empty());
      });
}

TEST(TransferTest, AggregateInitialization) {
  std::string BracesCode = R"(
    struct A {
      int Foo;
    };

    struct B {
      int Bar;
      A Baz;
      int Qux;
    };

    void target(int BarArg, int FooArg, int QuxArg) {
      B Quux{BarArg, {FooArg}, QuxArg};
      B OtherB;
      /*[[p]]*/
    }
  )";
  std::string BraceElisionCode = R"(
    struct A {
      int Foo;
    };

    struct B {
      int Bar;
      A Baz;
      int Qux;
    };

    void target(int BarArg, int FooArg, int QuxArg) {
      B Quux = {BarArg, FooArg, QuxArg};
      B OtherB;
      /*[[p]]*/
    }
  )";
  for (const std::string &Code : {BracesCode, BraceElisionCode}) {
    runDataflow(
        Code,
        [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
           ASTContext &ASTCtx) {
          ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

          const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
          ASSERT_THAT(FooDecl, NotNull());

          const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
          ASSERT_THAT(BarDecl, NotNull());

          const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
          ASSERT_THAT(BazDecl, NotNull());

          const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
          ASSERT_THAT(QuxDecl, NotNull());

          const ValueDecl *FooArgDecl = findValueDecl(ASTCtx, "FooArg");
          ASSERT_THAT(FooArgDecl, NotNull());

          const ValueDecl *BarArgDecl = findValueDecl(ASTCtx, "BarArg");
          ASSERT_THAT(BarArgDecl, NotNull());

          const ValueDecl *QuxArgDecl = findValueDecl(ASTCtx, "QuxArg");
          ASSERT_THAT(QuxArgDecl, NotNull());

          const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
          ASSERT_THAT(QuuxDecl, NotNull());

          const auto *FooArgVal = cast<IntegerValue>(Env.getValue(*FooArgDecl));
          const auto *BarArgVal = cast<IntegerValue>(Env.getValue(*BarArgDecl));
          const auto *QuxArgVal = cast<IntegerValue>(Env.getValue(*QuxArgDecl));

          const auto &QuuxLoc =
              *cast<RecordStorageLocation>(Env.getStorageLocation(*QuuxDecl));
          const auto &BazLoc =
              *cast<RecordStorageLocation>(QuuxLoc.getChild(*BazDecl));

          EXPECT_EQ(getFieldValue(&QuuxLoc, *BarDecl, Env), BarArgVal);
          EXPECT_EQ(getFieldValue(&BazLoc, *FooDecl, Env), FooArgVal);
          EXPECT_EQ(getFieldValue(&QuuxLoc, *QuxDecl, Env), QuxArgVal);

          // Check that fields initialized in an initializer list are always
          // modeled in other instances of the same type.
          const auto &OtherBLoc =
              getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "OtherB");
          EXPECT_THAT(OtherBLoc.getChild(*BarDecl), NotNull());
          EXPECT_THAT(OtherBLoc.getChild(*BazDecl), NotNull());
          EXPECT_THAT(OtherBLoc.getChild(*QuxDecl), NotNull());
        });
  }
}

TEST(TransferTest, AggregateInitializationReferenceField) {
  std::string Code = R"(
    struct S {
      int &RefField;
    };

    void target(int i) {
      S s = { i };
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *RefFieldDecl = findValueDecl(ASTCtx, "RefField");

        auto &ILoc = getLocForDecl<StorageLocation>(ASTCtx, Env, "i");
        auto &SLoc = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "s");

        EXPECT_EQ(SLoc.getChild(*RefFieldDecl), &ILoc);
      });
}

TEST(TransferTest, AggregateInitialization_NotExplicitlyInitializedField) {
  std::string Code = R"(
    struct S {
      int i1;
      int i2;
    };

    void target(int i) {
      S s = { i };
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *I1FieldDecl = findValueDecl(ASTCtx, "i1");
        const ValueDecl *I2FieldDecl = findValueDecl(ASTCtx, "i2");

        auto &SLoc = getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "s");

        auto &IValue = getValueForDecl<IntegerValue>(ASTCtx, Env, "i");
        auto &I1Value =
            *cast<IntegerValue>(getFieldValue(&SLoc, *I1FieldDecl, Env));
        EXPECT_EQ(&I1Value, &IValue);
        auto &I2Value =
            *cast<IntegerValue>(getFieldValue(&SLoc, *I2FieldDecl, Env));
        EXPECT_NE(&I2Value, &IValue);
      });
}

TEST(TransferTest, AggregateInitializationFunctionPointer) {
  // This is a repro for an assertion failure.
  // nullptr takes on the type of a const function pointer, but its type was
  // asserted to be equal to the *unqualified* type of Field, which no longer
  // included the const.
  std::string Code = R"(
    struct S {
      void (*const Field)();
    };

    void target() {
      S s{nullptr};
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {});
}

TEST(TransferTest, AssignToUnionMember) {
  std::string Code = R"(
    union A {
      int Foo;
    };

    void target(int Bar) {
      A Baz;
      Baz.Foo = Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());
        ASSERT_TRUE(BazDecl->getType()->isUnionType());

        auto BazFields = BazDecl->getType()->getAsRecordDecl()->fields();
        FieldDecl *FooDecl = nullptr;
        for (FieldDecl *Field : BazFields) {
          if (Field->getNameAsString() == "Foo") {
            FooDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(FooDecl, NotNull());

        const auto *BazLoc = dyn_cast_or_null<RecordStorageLocation>(
            Env.getStorageLocation(*BazDecl));
        ASSERT_THAT(BazLoc, NotNull());
        ASSERT_THAT(Env.getValue(*BazLoc), NotNull());

        const auto *FooVal =
            cast<IntegerValue>(getFieldValue(BazLoc, *FooDecl, Env));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());
        const auto *BarLoc = Env.getStorageLocation(*BarDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        EXPECT_EQ(Env.getValue(*BarLoc), FooVal);
      });
}

TEST(TransferTest, AssignFromBoolLiteral) {
  std::string Code = R"(
    void target() {
      bool Foo = true;
      bool Bar = false;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooVal =
            dyn_cast_or_null<BoolValue>(Env.getValue(*FooDecl));
        ASSERT_THAT(FooVal, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarVal =
            dyn_cast_or_null<BoolValue>(Env.getValue(*BarDecl));
        ASSERT_THAT(BarVal, NotNull());

        EXPECT_EQ(FooVal, &Env.getBoolLiteralValue(true));
        EXPECT_EQ(BarVal, &Env.getBoolLiteralValue(false));
      });
}

TEST(TransferTest, AssignFromCompositeBoolExpression) {
  {
    std::string Code = R"(
    void target(bool Foo, bool Bar, bool Qux) {
      bool Baz = (Foo) && (Bar || Qux);
      // [[p]]
    }
  )";
    runDataflow(
        Code,
        [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
           ASTContext &ASTCtx) {
          ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

          const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
          ASSERT_THAT(FooDecl, NotNull());

          const auto *FooVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*FooDecl));
          ASSERT_THAT(FooVal, NotNull());

          const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
          ASSERT_THAT(BarDecl, NotNull());

          const auto *BarVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*BarDecl));
          ASSERT_THAT(BarVal, NotNull());

          const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
          ASSERT_THAT(QuxDecl, NotNull());

          const auto *QuxVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*QuxDecl));
          ASSERT_THAT(QuxVal, NotNull());

          const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
          ASSERT_THAT(BazDecl, NotNull());

          const auto *BazVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*BazDecl));
          ASSERT_THAT(BazVal, NotNull());
          auto &A = Env.arena();
          EXPECT_EQ(&BazVal->formula(),
                    &A.makeAnd(FooVal->formula(),
                               A.makeOr(BarVal->formula(), QuxVal->formula())));
        });
  }

  {
    std::string Code = R"(
    void target(bool Foo, bool Bar, bool Qux) {
      bool Baz = (Foo && Qux) || (Bar);
      // [[p]]
    }
  )";
    runDataflow(
        Code,
        [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
           ASTContext &ASTCtx) {
          ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

          const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
          ASSERT_THAT(FooDecl, NotNull());

          const auto *FooVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*FooDecl));
          ASSERT_THAT(FooVal, NotNull());

          const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
          ASSERT_THAT(BarDecl, NotNull());

          const auto *BarVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*BarDecl));
          ASSERT_THAT(BarVal, NotNull());

          const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
          ASSERT_THAT(QuxDecl, NotNull());

          const auto *QuxVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*QuxDecl));
          ASSERT_THAT(QuxVal, NotNull());

          const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
          ASSERT_THAT(BazDecl, NotNull());

          const auto *BazVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*BazDecl));
          ASSERT_THAT(BazVal, NotNull());
          auto &A = Env.arena();
          EXPECT_EQ(&BazVal->formula(),
                    &A.makeOr(A.makeAnd(FooVal->formula(), QuxVal->formula()),
                              BarVal->formula()));
        });
  }

  {
    std::string Code = R"(
      void target(bool A, bool B, bool C, bool D) {
        bool Foo = ((A && B) && C) && D;
        // [[p]]
      }
    )";
    runDataflow(
        Code,
        [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
           ASTContext &ASTCtx) {
          ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

          const ValueDecl *ADecl = findValueDecl(ASTCtx, "A");
          ASSERT_THAT(ADecl, NotNull());

          const auto *AVal = dyn_cast_or_null<BoolValue>(Env.getValue(*ADecl));
          ASSERT_THAT(AVal, NotNull());

          const ValueDecl *BDecl = findValueDecl(ASTCtx, "B");
          ASSERT_THAT(BDecl, NotNull());

          const auto *BVal = dyn_cast_or_null<BoolValue>(Env.getValue(*BDecl));
          ASSERT_THAT(BVal, NotNull());

          const ValueDecl *CDecl = findValueDecl(ASTCtx, "C");
          ASSERT_THAT(CDecl, NotNull());

          const auto *CVal = dyn_cast_or_null<BoolValue>(Env.getValue(*CDecl));
          ASSERT_THAT(CVal, NotNull());

          const ValueDecl *DDecl = findValueDecl(ASTCtx, "D");
          ASSERT_THAT(DDecl, NotNull());

          const auto *DVal = dyn_cast_or_null<BoolValue>(Env.getValue(*DDecl));
          ASSERT_THAT(DVal, NotNull());

          const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
          ASSERT_THAT(FooDecl, NotNull());

          const auto *FooVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*FooDecl));
          ASSERT_THAT(FooVal, NotNull());
          auto &A = Env.arena();
          EXPECT_EQ(
              &FooVal->formula(),
              &A.makeAnd(A.makeAnd(A.makeAnd(AVal->formula(), BVal->formula()),
                                   CVal->formula()),
                         DVal->formula()));
        });
  }
}

TEST(TransferTest, AssignFromBoolNegation) {
  std::string Code = R"(
    void target() {
      bool Foo = true;
      bool Bar = !(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooVal =
            dyn_cast_or_null<BoolValue>(Env.getValue(*FooDecl));
        ASSERT_THAT(FooVal, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarVal =
            dyn_cast_or_null<BoolValue>(Env.getValue(*BarDecl));
        ASSERT_THAT(BarVal, NotNull());
        auto &A = Env.arena();
        EXPECT_EQ(&BarVal->formula(), &A.makeNot(FooVal->formula()));
      });
}

TEST(TransferTest, BuiltinExpect) {
  std::string Code = R"(
    void target(long Foo) {
      long Bar = __builtin_expect(Foo, true);
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        EXPECT_EQ(Env.getValue(*FooDecl), Env.getValue(*BarDecl));
      });
}

// `__builtin_expect` takes and returns a `long` argument, so other types
// involve casts. This verifies that we identify the input and output in that
// case.
TEST(TransferTest, BuiltinExpectBoolArg) {
  std::string Code = R"(
    void target(bool Foo) {
      bool Bar = __builtin_expect(Foo, true);
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        EXPECT_EQ(Env.getValue(*FooDecl), Env.getValue(*BarDecl));
      });
}

TEST(TransferTest, BuiltinUnreachable) {
  std::string Code = R"(
    void target(bool Foo) {
      bool Bar = false;
      if (Foo)
        Bar = Foo;
      else
        __builtin_unreachable();
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        // `__builtin_unreachable` promises that the code is
        // unreachable, so the compiler treats the "then" branch as the
        // only possible predecessor of this statement.
        EXPECT_EQ(Env.getValue(*FooDecl), Env.getValue(*BarDecl));
      });
}

TEST(TransferTest, BuiltinTrap) {
  std::string Code = R"(
    void target(bool Foo) {
      bool Bar = false;
      if (Foo)
        Bar = Foo;
      else
        __builtin_trap();
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        // `__builtin_trap` ensures program termination, so only the
        // "then" branch is a predecessor of this statement.
        EXPECT_EQ(Env.getValue(*FooDecl), Env.getValue(*BarDecl));
      });
}

TEST(TransferTest, BuiltinDebugTrap) {
  std::string Code = R"(
    void target(bool Foo) {
      bool Bar = false;
      if (Foo)
        Bar = Foo;
      else
        __builtin_debugtrap();
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        // `__builtin_debugtrap` doesn't ensure program termination.
        EXPECT_NE(Env.getValue(*FooDecl), Env.getValue(*BarDecl));
      });
}

TEST(TransferTest, StaticIntSingleVarDecl) {
  std::string Code = R"(
    void target() {
      static int Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));
      });
}

TEST(TransferTest, StaticIntGroupVarDecl) {
  std::string Code = R"(
    void target() {
      static int Foo, Bar;
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const Value *BarVal = Env.getValue(*BarLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        EXPECT_NE(FooVal, BarVal);
      });
}

TEST(TransferTest, GlobalIntVarDecl) {
  std::string Code = R"(
    static int Foo;

    void target() {
      int Bar = Foo;
      int Baz = Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const Value *BarVal = cast<IntegerValue>(Env.getValue(*BarDecl));
        const Value *BazVal = cast<IntegerValue>(Env.getValue(*BazDecl));
        EXPECT_EQ(BarVal, BazVal);
      });
}

TEST(TransferTest, StaticMemberIntVarDecl) {
  std::string Code = R"(
    struct A {
      static int Foo;
    };

    void target(A a) {
      int Bar = a.Foo;
      int Baz = a.Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const Value *BarVal = cast<IntegerValue>(Env.getValue(*BarDecl));
        const Value *BazVal = cast<IntegerValue>(Env.getValue(*BazDecl));
        EXPECT_EQ(BarVal, BazVal);
      });
}

TEST(TransferTest, StaticMemberRefVarDecl) {
  std::string Code = R"(
    struct A {
      static int &Foo;
    };

    void target(A a) {
      int Bar = a.Foo;
      int Baz = a.Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const Value *BarVal = cast<IntegerValue>(Env.getValue(*BarDecl));
        const Value *BazVal = cast<IntegerValue>(Env.getValue(*BazDecl));
        EXPECT_EQ(BarVal, BazVal);
      });
}

TEST(TransferTest, AssignMemberBeforeCopy) {
  std::string Code = R"(
    struct A {
      int Foo;
    };

    void target() {
      A A1;
      A A2;
      int Bar;
      A1.Foo = Bar;
      A2 = A1;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *A1Decl = findValueDecl(ASTCtx, "A1");
        ASSERT_THAT(A1Decl, NotNull());

        const ValueDecl *A2Decl = findValueDecl(ASTCtx, "A2");
        ASSERT_THAT(A2Decl, NotNull());

        const auto *BarVal = cast<IntegerValue>(Env.getValue(*BarDecl));

        const auto &A2Loc =
            *cast<RecordStorageLocation>(Env.getStorageLocation(*A2Decl));
        EXPECT_EQ(getFieldValue(&A2Loc, *FooDecl, Env), BarVal);
      });
}

TEST(TransferTest, BooleanEquality) {
  std::string Code = R"(
    void target(bool Bar) {
      bool Foo = true;
      if (Bar == Foo) {
        (void)0;
        /*[[p-then]]*/
      } else {
        (void)0;
        /*[[p-else]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p-then", "p-else"));
        const Environment &EnvThen =
            getEnvironmentAtAnnotation(Results, "p-then");
        const Environment &EnvElse =
            getEnvironmentAtAnnotation(Results, "p-else");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarValThen = getFormula(*BarDecl, EnvThen);
        EXPECT_TRUE(EnvThen.proves(BarValThen));

        auto &BarValElse = getFormula(*BarDecl, EnvElse);
        EXPECT_TRUE(EnvElse.proves(EnvElse.arena().makeNot(BarValElse)));
      });
}

TEST(TransferTest, BooleanInequality) {
  std::string Code = R"(
    void target(bool Bar) {
      bool Foo = true;
      if (Bar != Foo) {
        (void)0;
        /*[[p-then]]*/
      } else {
        (void)0;
        /*[[p-else]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p-then", "p-else"));
        const Environment &EnvThen =
            getEnvironmentAtAnnotation(Results, "p-then");
        const Environment &EnvElse =
            getEnvironmentAtAnnotation(Results, "p-else");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarValThen = getFormula(*BarDecl, EnvThen);
        EXPECT_TRUE(EnvThen.proves(EnvThen.arena().makeNot(BarValThen)));

        auto &BarValElse = getFormula(*BarDecl, EnvElse);
        EXPECT_TRUE(EnvElse.proves(BarValElse));
      });
}

TEST(TransferTest, IntegerLiteralEquality) {
  std::string Code = R"(
    void target() {
      bool equal = (42 == 42);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto &Equal =
            getValueForDecl<BoolValue>(ASTCtx, Env, "equal").formula();
        EXPECT_TRUE(Env.proves(Equal));
      });
}

TEST(TransferTest, CorrelatedBranches) {
  std::string Code = R"(
    void target(bool B, bool C) {
      if (B) {
        return;
      }
      (void)0;
      /*[[p0]]*/
      if (C) {
        B = true;
        /*[[p1]]*/
      }
      if (B) {
        (void)0;
        /*[[p2]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p0", "p1", "p2"));

        const ValueDecl *CDecl = findValueDecl(ASTCtx, "C");
        ASSERT_THAT(CDecl, NotNull());

        {
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p0");
          const ValueDecl *BDecl = findValueDecl(ASTCtx, "B");
          ASSERT_THAT(BDecl, NotNull());
          auto &BVal = getFormula(*BDecl, Env);

          EXPECT_TRUE(Env.proves(Env.arena().makeNot(BVal)));
        }

        {
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p1");
          auto &CVal = getFormula(*CDecl, Env);
          EXPECT_TRUE(Env.proves(CVal));
        }

        {
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p2");
          auto &CVal = getFormula(*CDecl, Env);
          EXPECT_TRUE(Env.proves(CVal));
        }
      });
}

TEST(TransferTest, LoopWithAssignmentConverges) {
  std::string Code = R"(
    bool foo();

    void target() {
       do {
        bool Bar = foo();
        if (Bar) break;
        (void)Bar;
        /*[[p]]*/
      } while (true);
    }
  )";
  // The key property that we are verifying is implicit in `runDataflow` --
  // namely, that the analysis succeeds, rather than hitting the maximum number
  // of iterations.
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal = getFormula(*BarDecl, Env);
        EXPECT_TRUE(Env.proves(Env.arena().makeNot(BarVal)));
      });
}

TEST(TransferTest, LoopWithStagedAssignments) {
  std::string Code = R"(
    bool foo();

    void target() {
      bool Bar = false;
      bool Err = false;
      while (foo()) {
        if (Bar)
          Err = true;
        Bar = true;
        /*[[p]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());
        const ValueDecl *ErrDecl = findValueDecl(ASTCtx, "Err");
        ASSERT_THAT(ErrDecl, NotNull());

        auto &BarVal = getFormula(*BarDecl, Env);
        auto &ErrVal = getFormula(*ErrDecl, Env);
        EXPECT_TRUE(Env.proves(BarVal));
        // An unsound analysis, for example only evaluating the loop once, can
        // conclude that `Err` is false. So, we test that this conclusion is not
        // reached.
        EXPECT_FALSE(Env.proves(Env.arena().makeNot(ErrVal)));
      });
}

TEST(TransferTest, LoopWithReferenceAssignmentConverges) {
  std::string Code = R"(
    bool &foo();

    void target() {
       do {
        bool& Bar = foo();
        if (Bar) break;
        (void)Bar;
        /*[[p]]*/
      } while (true);
    }
  )";
  // The key property that we are verifying is that the analysis succeeds,
  // rather than hitting the maximum number of iterations.
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal = getFormula(*BarDecl, Env);
        EXPECT_TRUE(Env.proves(Env.arena().makeNot(BarVal)));
      });
}

TEST(TransferTest, LoopWithStructReferenceAssignmentConverges) {
  std::string Code = R"(
    struct Lookup {
      int x;
    };

    void target(Lookup val, bool b) {
      const Lookup* l = nullptr;
      while (b) {
        l = &val;
        /*[[p-inner]]*/
      }
      (void)0;
      /*[[p-outer]]*/
    }
  )";
  // The key property that we are verifying is implicit in `runDataflow` --
  // namely, that the analysis succeeds, rather than hitting the maximum number
  // of iterations.
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p-inner", "p-outer"));
        const Environment &InnerEnv =
            getEnvironmentAtAnnotation(Results, "p-inner");
        const Environment &OuterEnv =
            getEnvironmentAtAnnotation(Results, "p-outer");

        const ValueDecl *ValDecl = findValueDecl(ASTCtx, "val");
        ASSERT_THAT(ValDecl, NotNull());

        const ValueDecl *LDecl = findValueDecl(ASTCtx, "l");
        ASSERT_THAT(LDecl, NotNull());

        // Inner.
        auto *LVal = dyn_cast<PointerValue>(InnerEnv.getValue(*LDecl));
        ASSERT_THAT(LVal, NotNull());

        EXPECT_EQ(&LVal->getPointeeLoc(),
                  InnerEnv.getStorageLocation(*ValDecl));

        // Outer.
        LVal = dyn_cast<PointerValue>(OuterEnv.getValue(*LDecl));
        ASSERT_THAT(LVal, NotNull());

        // The loop body may not have been executed, so we should not conclude
        // that `l` points to `val`.
        EXPECT_NE(&LVal->getPointeeLoc(),
                  OuterEnv.getStorageLocation(*ValDecl));
      });
}

TEST(TransferTest, LoopDereferencingChangingPointerConverges) {
  std::string Code = R"cc(
    bool some_condition();

    void target(int i1, int i2) {
      int *p = &i1;
      while (true) {
        (void)*p;
        if (some_condition())
          p = &i1;
        else
          p = &i2;
      }
    }
  )cc";
  ASSERT_THAT_ERROR(checkDataflowWithNoopAnalysis(Code), llvm::Succeeded());
}

TEST(TransferTest, LoopDereferencingChangingRecordPointerConverges) {
  std::string Code = R"cc(
    struct Lookup {
      int x;
    };

    bool some_condition();

    void target(Lookup l1, Lookup l2) {
      Lookup *l = &l1;
      while (true) {
        (void)l->x;
        if (some_condition())
          l = &l1;
        else
          l = &l2;
      }
    }
  )cc";
  ASSERT_THAT_ERROR(checkDataflowWithNoopAnalysis(Code), llvm::Succeeded());
}

TEST(TransferTest, LoopWithShortCircuitedConditionConverges) {
  std::string Code = R"cc(
    bool foo();

    void target() {
      bool c = false;
      while (foo() || foo()) {
        c = true;
      }
    }
  )cc";
  ASSERT_THAT_ERROR(checkDataflowWithNoopAnalysis(Code), llvm::Succeeded());
}

TEST(TransferTest, LoopCanProveInvariantForBoolean) {
  // Check that we can prove `b` is always false in the loop.
  // This test exercises the logic in `widenDistinctValues()` that preserves
  // information if the boolean can be proved to be either true or false in both
  // the previous and current iteration.
  std::string Code = R"cc(
    int return_int();
    void target() {
      bool b = return_int() == 0;
      if (b) return;
      while (true) {
        b;
        // [[p]]
        b = return_int() == 0;
        if (b) return;
      }
    }
  )cc";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        auto &BVal = getValueForDecl<BoolValue>(ASTCtx, Env, "b");
        EXPECT_TRUE(Env.proves(Env.arena().makeNot(BVal.formula())));
      });
}

TEST(TransferTest, DoesNotCrashOnUnionThisExpr) {
  std::string Code = R"(
    union Union {
      int A;
      float B;
    };

    void foo() {
      Union A;
      Union B;
      A = B;
    }
  )";
  // This is a crash regression test when calling the transfer function on a
  // `CXXThisExpr` that refers to a union.
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
         ASTContext &) {},
      LangStandard::lang_cxx17, /*ApplyBuiltinTransfer=*/true, "operator=");
}

TEST(TransferTest, StructuredBindingAssignFromStructIntMembersToRefs) {
  std::string Code = R"(
    struct A {
      int Foo;
      int Bar;
    };

    void target() {
      int Qux;
      A Baz;
      Baz.Foo = Qux;
      auto &FooRef = Baz.Foo;
      auto &BarRef = Baz.Bar;
      auto &[BoundFooRef, BoundBarRef] = Baz;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooRefDecl = findValueDecl(ASTCtx, "FooRef");
        ASSERT_THAT(FooRefDecl, NotNull());

        const ValueDecl *BarRefDecl = findValueDecl(ASTCtx, "BarRef");
        ASSERT_THAT(BarRefDecl, NotNull());

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        const ValueDecl *BoundFooRefDecl = findValueDecl(ASTCtx, "BoundFooRef");
        ASSERT_THAT(BoundFooRefDecl, NotNull());

        const ValueDecl *BoundBarRefDecl = findValueDecl(ASTCtx, "BoundBarRef");
        ASSERT_THAT(BoundBarRefDecl, NotNull());

        const StorageLocation *FooRefLoc = Env.getStorageLocation(*FooRefDecl);
        ASSERT_THAT(FooRefLoc, NotNull());

        const StorageLocation *BarRefLoc = Env.getStorageLocation(*BarRefDecl);
        ASSERT_THAT(BarRefLoc, NotNull());

        const Value *QuxVal = Env.getValue(*QuxDecl);
        ASSERT_THAT(QuxVal, NotNull());

        const StorageLocation *BoundFooRefLoc =
            Env.getStorageLocation(*BoundFooRefDecl);
        EXPECT_EQ(BoundFooRefLoc, FooRefLoc);

        const StorageLocation *BoundBarRefLoc =
            Env.getStorageLocation(*BoundBarRefDecl);
        EXPECT_EQ(BoundBarRefLoc, BarRefLoc);

        EXPECT_EQ(Env.getValue(*BoundFooRefDecl), QuxVal);
      });
}

TEST(TransferTest, StructuredBindingAssignFromStructRefMembersToRefs) {
  std::string Code = R"(
    struct A {
      int &Foo;
      int &Bar;
    };

    void target(A Baz) {
      int Qux;
      Baz.Foo = Qux;
      auto &FooRef = Baz.Foo;
      auto &BarRef = Baz.Bar;
      auto &[BoundFooRef, BoundBarRef] = Baz;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooRefDecl = findValueDecl(ASTCtx, "FooRef");
        ASSERT_THAT(FooRefDecl, NotNull());

        const ValueDecl *BarRefDecl = findValueDecl(ASTCtx, "BarRef");
        ASSERT_THAT(BarRefDecl, NotNull());

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        const ValueDecl *BoundFooRefDecl = findValueDecl(ASTCtx, "BoundFooRef");
        ASSERT_THAT(BoundFooRefDecl, NotNull());

        const ValueDecl *BoundBarRefDecl = findValueDecl(ASTCtx, "BoundBarRef");
        ASSERT_THAT(BoundBarRefDecl, NotNull());

        const StorageLocation *FooRefLoc = Env.getStorageLocation(*FooRefDecl);
        ASSERT_THAT(FooRefLoc, NotNull());

        const StorageLocation *BarRefLoc = Env.getStorageLocation(*BarRefDecl);
        ASSERT_THAT(BarRefLoc, NotNull());

        const Value *QuxVal = Env.getValue(*QuxDecl);
        ASSERT_THAT(QuxVal, NotNull());

        const StorageLocation *BoundFooRefLoc =
            Env.getStorageLocation(*BoundFooRefDecl);
        EXPECT_EQ(BoundFooRefLoc, FooRefLoc);

        const StorageLocation *BoundBarRefLoc =
            Env.getStorageLocation(*BoundBarRefDecl);
        EXPECT_EQ(BoundBarRefLoc, BarRefLoc);

        EXPECT_EQ(Env.getValue(*BoundFooRefDecl), QuxVal);
      });
}

TEST(TransferTest, StructuredBindingAssignFromStructIntMembersToInts) {
  std::string Code = R"(
    struct A {
      int Foo;
      int Bar;
    };

    void target() {
      int Qux;
      A Baz;
      Baz.Foo = Qux;
      auto &FooRef = Baz.Foo;
      auto &BarRef = Baz.Bar;
      auto [BoundFoo, BoundBar] = Baz;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooRefDecl = findValueDecl(ASTCtx, "FooRef");
        ASSERT_THAT(FooRefDecl, NotNull());

        const ValueDecl *BarRefDecl = findValueDecl(ASTCtx, "BarRef");
        ASSERT_THAT(BarRefDecl, NotNull());

        const ValueDecl *BoundFooDecl = findValueDecl(ASTCtx, "BoundFoo");
        ASSERT_THAT(BoundFooDecl, NotNull());

        const ValueDecl *BoundBarDecl = findValueDecl(ASTCtx, "BoundBar");
        ASSERT_THAT(BoundBarDecl, NotNull());

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        const StorageLocation *FooRefLoc = Env.getStorageLocation(*FooRefDecl);
        ASSERT_THAT(FooRefLoc, NotNull());

        const StorageLocation *BarRefLoc = Env.getStorageLocation(*BarRefDecl);
        ASSERT_THAT(BarRefLoc, NotNull());

        const Value *QuxVal = Env.getValue(*QuxDecl);
        ASSERT_THAT(QuxVal, NotNull());

        const StorageLocation *BoundFooLoc =
            Env.getStorageLocation(*BoundFooDecl);
        EXPECT_NE(BoundFooLoc, FooRefLoc);

        const StorageLocation *BoundBarLoc =
            Env.getStorageLocation(*BoundBarDecl);
        EXPECT_NE(BoundBarLoc, BarRefLoc);

        EXPECT_EQ(Env.getValue(*BoundFooDecl), QuxVal);
      });
}

TEST(TransferTest, StructuredBindingAssignFromTupleLikeType) {
  std::string Code = R"(
    namespace std {
    using size_t = int;
    template <class> struct tuple_size;
    template <std::size_t, class> struct tuple_element;
    template <class...> class tuple;

    namespace {
    template <class T, T v>
    struct size_helper { static const T value = v; };
    } // namespace

    template <class... T>
    struct tuple_size<tuple<T...>> : size_helper<std::size_t, sizeof...(T)> {};

    template <std::size_t I, class... T>
    struct tuple_element<I, tuple<T...>> {
      using type =  __type_pack_element<I, T...>;
    };

    template <class...> class tuple {};

    template <std::size_t I, class... T>
    typename tuple_element<I, tuple<T...>>::type get(tuple<T...>);
    } // namespace std

    std::tuple<bool, int> makeTuple();

    void target(bool B) {
      auto [BoundFoo, BoundBar] = makeTuple();
      bool Baz;
      // Include if-then-else to test interaction of `BindingDecl` with join.
      if (B) {
        Baz = BoundFoo;
        (void)BoundBar;
        // [[p1]]
      } else {
        Baz = BoundFoo;
      }
      (void)0;
      // [[p2]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");

        const ValueDecl *BoundFooDecl = findValueDecl(ASTCtx, "BoundFoo");
        ASSERT_THAT(BoundFooDecl, NotNull());

        const ValueDecl *BoundBarDecl = findValueDecl(ASTCtx, "BoundBar");
        ASSERT_THAT(BoundBarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        // BindingDecls always map to references -- either lvalue or rvalue, so
        // we still need to skip here.
        const Value *BoundFooValue = Env1.getValue(*BoundFooDecl);
        ASSERT_THAT(BoundFooValue, NotNull());
        EXPECT_TRUE(isa<BoolValue>(BoundFooValue));

        const Value *BoundBarValue = Env1.getValue(*BoundBarDecl);
        ASSERT_THAT(BoundBarValue, NotNull());
        EXPECT_TRUE(isa<IntegerValue>(BoundBarValue));

        // Test that a `DeclRefExpr` to a `BindingDecl` works as expected.
        EXPECT_EQ(Env1.getValue(*BazDecl), BoundFooValue);

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        // Test that `BoundFooDecl` retains the value we expect, after the join.
        BoundFooValue = Env2.getValue(*BoundFooDecl);
        EXPECT_EQ(Env2.getValue(*BazDecl), BoundFooValue);
      });
}

TEST(TransferTest, StructuredBindingAssignRefFromTupleLikeType) {
  std::string Code = R"(
    namespace std {
    using size_t = int;
    template <class> struct tuple_size;
    template <std::size_t, class> struct tuple_element;
    template <class...> class tuple;

    namespace {
    template <class T, T v>
    struct size_helper { static const T value = v; };
    } // namespace

    template <class... T>
    struct tuple_size<tuple<T...>> : size_helper<std::size_t, sizeof...(T)> {};

    template <std::size_t I, class... T>
    struct tuple_element<I, tuple<T...>> {
      using type =  __type_pack_element<I, T...>;
    };

    template <class...> class tuple {};

    template <std::size_t I, class... T>
    typename tuple_element<I, tuple<T...>>::type get(tuple<T...>);
    } // namespace std

    std::tuple<bool, int> &getTuple();

    void target(bool B) {
      auto &[BoundFoo, BoundBar] = getTuple();
      bool Baz;
      // Include if-then-else to test interaction of `BindingDecl` with join.
      if (B) {
        Baz = BoundFoo;
        (void)BoundBar;
        // [[p1]]
      } else {
        Baz = BoundFoo;
      }
      (void)0;
      // [[p2]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");

        const ValueDecl *BoundFooDecl = findValueDecl(ASTCtx, "BoundFoo");
        ASSERT_THAT(BoundFooDecl, NotNull());

        const ValueDecl *BoundBarDecl = findValueDecl(ASTCtx, "BoundBar");
        ASSERT_THAT(BoundBarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const Value *BoundFooValue = Env1.getValue(*BoundFooDecl);
        ASSERT_THAT(BoundFooValue, NotNull());
        EXPECT_TRUE(isa<BoolValue>(BoundFooValue));

        const Value *BoundBarValue = Env1.getValue(*BoundBarDecl);
        ASSERT_THAT(BoundBarValue, NotNull());
        EXPECT_TRUE(isa<IntegerValue>(BoundBarValue));

        // Test that a `DeclRefExpr` to a `BindingDecl` (with reference type)
        // works as expected. We don't test aliasing properties of the
        // reference, because we don't model `std::get` and so have no way to
        // equate separate references into the tuple.
        EXPECT_EQ(Env1.getValue(*BazDecl), BoundFooValue);

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        // Test that `BoundFooDecl` retains the value we expect, after the join.
        BoundFooValue = Env2.getValue(*BoundFooDecl);
        EXPECT_EQ(Env2.getValue(*BazDecl), BoundFooValue);
      });
}

TEST(TransferTest, BinaryOperatorComma) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      int &Baz = (Foo, Bar);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        ASSERT_THAT(BarLoc, NotNull());

        const StorageLocation *BazLoc = Env.getStorageLocation(*BazDecl);
        EXPECT_EQ(BazLoc, BarLoc);
      });
}

TEST(TransferTest, IfStmtBranchExtendsFlowCondition) {
  std::string Code = R"(
    void target(bool Foo) {
      if (Foo) {
        (void)0;
        // [[if_then]]
      } else {
        (void)0;
        // [[if_else]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("if_then", "if_else"));
        const Environment &ThenEnv =
            getEnvironmentAtAnnotation(Results, "if_then");
        const Environment &ElseEnv =
            getEnvironmentAtAnnotation(Results, "if_else");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &ThenFooVal= getFormula(*FooDecl, ThenEnv);
        EXPECT_TRUE(ThenEnv.proves(ThenFooVal));

        auto &ElseFooVal = getFormula(*FooDecl, ElseEnv);
        EXPECT_TRUE(ElseEnv.proves(ElseEnv.arena().makeNot(ElseFooVal)));
      });
}

TEST(TransferTest, WhileStmtBranchExtendsFlowCondition) {
  std::string Code = R"(
    void target(bool Foo) {
      while (Foo) {
        (void)0;
        // [[loop_body]]
      }
      (void)0;
      // [[after_loop]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(),
                    UnorderedElementsAre("loop_body", "after_loop"));
        const Environment &LoopBodyEnv =
            getEnvironmentAtAnnotation(Results, "loop_body");
        const Environment &AfterLoopEnv =
            getEnvironmentAtAnnotation(Results, "after_loop");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &LoopBodyFooVal = getFormula(*FooDecl, LoopBodyEnv);
        EXPECT_TRUE(LoopBodyEnv.proves(LoopBodyFooVal));

        auto &AfterLoopFooVal = getFormula(*FooDecl, AfterLoopEnv);
        EXPECT_TRUE(
            AfterLoopEnv.proves(AfterLoopEnv.arena().makeNot(AfterLoopFooVal)));
      });
}

TEST(TransferTest, DoWhileStmtBranchExtendsFlowCondition) {
  std::string Code = R"(
    void target(bool Foo) {
      bool Bar = true;
      do {
        (void)0;
        // [[loop_body]]
        Bar = false;
      } while (Foo);
      (void)0;
      // [[after_loop]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(),
                    UnorderedElementsAre("loop_body", "after_loop"));
        const Environment &LoopBodyEnv =
            getEnvironmentAtAnnotation(Results, "loop_body");
        const Environment &AfterLoopEnv =
            getEnvironmentAtAnnotation(Results, "after_loop");
        auto &A = AfterLoopEnv.arena();

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &LoopBodyFooVal= getFormula(*FooDecl, LoopBodyEnv);
        auto &LoopBodyBarVal = getFormula(*BarDecl, LoopBodyEnv);
        EXPECT_TRUE(
            LoopBodyEnv.proves(A.makeOr(LoopBodyBarVal, LoopBodyFooVal)));

        auto &AfterLoopFooVal = getFormula(*FooDecl, AfterLoopEnv);
        auto &AfterLoopBarVal = getFormula(*BarDecl, AfterLoopEnv);
        EXPECT_TRUE(AfterLoopEnv.proves(A.makeNot(AfterLoopFooVal)));
        EXPECT_TRUE(AfterLoopEnv.proves(A.makeNot(AfterLoopBarVal)));
      });
}

TEST(TransferTest, ForStmtBranchExtendsFlowCondition) {
  std::string Code = R"(
    void target(bool Foo) {
      for (; Foo;) {
        (void)0;
        // [[loop_body]]
      }
      (void)0;
      // [[after_loop]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(),
                    UnorderedElementsAre("loop_body", "after_loop"));
        const Environment &LoopBodyEnv =
            getEnvironmentAtAnnotation(Results, "loop_body");
        const Environment &AfterLoopEnv =
            getEnvironmentAtAnnotation(Results, "after_loop");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &LoopBodyFooVal= getFormula(*FooDecl, LoopBodyEnv);
        EXPECT_TRUE(LoopBodyEnv.proves(LoopBodyFooVal));

        auto &AfterLoopFooVal = getFormula(*FooDecl, AfterLoopEnv);
        EXPECT_TRUE(
            AfterLoopEnv.proves(AfterLoopEnv.arena().makeNot(AfterLoopFooVal)));
      });
}

TEST(TransferTest, ForStmtBranchWithoutConditionDoesNotExtendFlowCondition) {
  std::string Code = R"(
    void target(bool Foo) {
      for (;;) {
        (void)0;
        // [[loop_body]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("loop_body"));
        const Environment &LoopBodyEnv =
            getEnvironmentAtAnnotation(Results, "loop_body");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &LoopBodyFooVal= getFormula(*FooDecl, LoopBodyEnv);
        EXPECT_FALSE(LoopBodyEnv.proves(LoopBodyFooVal));
      });
}

TEST(TransferTest, ContextSensitiveOptionDisabled) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool(bool &Var) { Var = true; }

    void target() {
      bool Foo = GiveBool();
      SetBool(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_FALSE(Env.proves(FooVal));
        EXPECT_FALSE(Env.proves(Env.arena().makeNot(FooVal)));
      },
      {BuiltinOptions{/*.ContextSensitiveOpts=*/std::nullopt}});
}

TEST(TransferTest, ContextSensitiveReturnReference) {
  std::string Code = R"(
    class S {};
    S& target(bool b, S &s) {
      return s;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *SDecl = findValueDecl(ASTCtx, "s");
        ASSERT_THAT(SDecl, NotNull());

        auto *SLoc = Env.getStorageLocation(*SDecl);
        ASSERT_THAT(SLoc, NotNull());

        ASSERT_THAT(Env.getReturnStorageLocation(), Eq(SLoc));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

// This test is a regression test, based on a real crash.
TEST(TransferTest, ContextSensitiveReturnReferenceWithConditionalOperator) {
  std::string Code = R"(
    class S {};
    S& target(bool b, S &s) {
      return b ? s : s;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *SDecl = findValueDecl(ASTCtx, "s");
        ASSERT_THAT(SDecl, NotNull());

        auto *SLoc = Env.getStorageLocation(*SDecl);
        ASSERT_THAT(SLoc, NotNull());
        EXPECT_THAT(Env.getValue(*SLoc), NotNull());

        auto *Loc = Env.getReturnStorageLocation();
        ASSERT_THAT(Loc, NotNull());
        EXPECT_THAT(Env.getValue(*Loc), NotNull());

        // TODO: We would really like to make this stronger assertion, but that
        // doesn't work because we don't propagate values correctly through
        // the conditional operator yet.
        // ASSERT_THAT(Loc, Eq(SLoc));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveReturnOneOfTwoReferences) {
  std::string Code = R"(
    class S {};
    S &callee(bool b, S &s1_parm, S &s2_parm) {
      if (b)
        return s1_parm;
      else
        return s2_parm;
    }
    void target(bool b) {
      S s1;
      S s2;
      S &return_s1 = s1;
      S &return_s2 = s2;
      S &return_dont_know = callee(b, s1, s2);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *S1 = findValueDecl(ASTCtx, "s1");
        ASSERT_THAT(S1, NotNull());
        const ValueDecl *S2 = findValueDecl(ASTCtx, "s2");
        ASSERT_THAT(S2, NotNull());
        const ValueDecl *ReturnS1 = findValueDecl(ASTCtx, "return_s1");
        ASSERT_THAT(ReturnS1, NotNull());
        const ValueDecl *ReturnS2 = findValueDecl(ASTCtx, "return_s2");
        ASSERT_THAT(ReturnS2, NotNull());
        const ValueDecl *ReturnDontKnow =
            findValueDecl(ASTCtx, "return_dont_know");
        ASSERT_THAT(ReturnDontKnow, NotNull());

        StorageLocation *S1Loc = Env.getStorageLocation(*S1);
        StorageLocation *S2Loc = Env.getStorageLocation(*S2);

        EXPECT_THAT(Env.getStorageLocation(*ReturnS1), Eq(S1Loc));
        EXPECT_THAT(Env.getStorageLocation(*ReturnS2), Eq(S2Loc));

        // In the case where we don't have a consistent storage location for
        // the return value, the framework creates a new storage location, which
        // should be different from the storage locations of `s1` and `s2`.
        EXPECT_THAT(Env.getStorageLocation(*ReturnDontKnow), Ne(S1Loc));
        EXPECT_THAT(Env.getStorageLocation(*ReturnDontKnow), Ne(S2Loc));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveDepthZero) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool(bool &Var) { Var = true; }

    void target() {
      bool Foo = GiveBool();
      SetBool(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_FALSE(Env.proves(FooVal));
        EXPECT_FALSE(Env.proves(Env.arena().makeNot(FooVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{/*.Depth=*/0}}});
}

TEST(TransferTest, ContextSensitiveSetTrue) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool(bool &Var) { Var = true; }

    void target() {
      bool Foo = GiveBool();
      SetBool(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveSetFalse) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool(bool &Var) { Var = false; }

    void target() {
      bool Foo = GiveBool();
      SetBool(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(Env.arena().makeNot(FooVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveSetBothTrueAndFalse) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool(bool &Var, bool Val) { Var = Val; }

    void target() {
      bool Foo = GiveBool();
      bool Bar = GiveBool();
      SetBool(Foo, true);
      SetBool(Bar, false);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        auto &A = Env.arena();

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
        EXPECT_FALSE(Env.proves(A.makeNot(FooVal)));

        auto &BarVal = getFormula(*BarDecl, Env);
        EXPECT_FALSE(Env.proves(BarVal));
        EXPECT_TRUE(Env.proves(A.makeNot(BarVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveSetTwoLayersDepthOne) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool1(bool &Var) { Var = true; }
    void SetBool2(bool &Var) { SetBool1(Var); }

    void target() {
      bool Foo = GiveBool();
      SetBool2(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_FALSE(Env.proves(FooVal));
        EXPECT_FALSE(Env.proves(Env.arena().makeNot(FooVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{/*.Depth=*/1}}});
}

TEST(TransferTest, ContextSensitiveSetTwoLayersDepthTwo) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool1(bool &Var) { Var = true; }
    void SetBool2(bool &Var) { SetBool1(Var); }

    void target() {
      bool Foo = GiveBool();
      SetBool2(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{/*.Depth=*/2}}});
}

TEST(TransferTest, ContextSensitiveSetThreeLayersDepthTwo) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool1(bool &Var) { Var = true; }
    void SetBool2(bool &Var) { SetBool1(Var); }
    void SetBool3(bool &Var) { SetBool2(Var); }

    void target() {
      bool Foo = GiveBool();
      SetBool3(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_FALSE(Env.proves(FooVal));
        EXPECT_FALSE(Env.proves(Env.arena().makeNot(FooVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{/*.Depth=*/2}}});
}

TEST(TransferTest, ContextSensitiveSetThreeLayersDepthThree) {
  std::string Code = R"(
    bool GiveBool();
    void SetBool1(bool &Var) { Var = true; }
    void SetBool2(bool &Var) { SetBool1(Var); }
    void SetBool3(bool &Var) { SetBool2(Var); }

    void target() {
      bool Foo = GiveBool();
      SetBool3(Foo);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{/*.Depth=*/3}}});
}

TEST(TransferTest, ContextSensitiveMutualRecursion) {
  std::string Code = R"(
    bool Pong(bool X, bool Y);

    bool Ping(bool X, bool Y) {
      if (X) {
        return Y;
      } else {
        return Pong(!X, Y);
      }
    }

    bool Pong(bool X, bool Y) {
      if (Y) {
        return X;
      } else {
        return Ping(X, !Y);
      }
    }

    void target() {
      bool Foo = Ping(false, false);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        // The analysis doesn't crash...
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        // ... but it also can't prove anything here.
        EXPECT_FALSE(Env.proves(FooVal));
        EXPECT_FALSE(Env.proves(Env.arena().makeNot(FooVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{/*.Depth=*/4}}});
}

TEST(TransferTest, ContextSensitiveSetMultipleLines) {
  std::string Code = R"(
    void SetBools(bool &Var1, bool &Var2) {
      Var1 = true;
      Var2 = false;
    }

    void target() {
      bool Foo = false;
      bool Bar = true;
      SetBools(Foo, Bar);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
        EXPECT_FALSE(Env.proves(Env.arena().makeNot(FooVal)));

        auto &BarVal = getFormula(*BarDecl, Env);
        EXPECT_FALSE(Env.proves(BarVal));
        EXPECT_TRUE(Env.proves(Env.arena().makeNot(BarVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveSetMultipleBlocks) {
  std::string Code = R"(
    void IfCond(bool Cond, bool &Then, bool &Else) {
      if (Cond) {
        Then = true;
      } else {
        Else = true;
      }
    }

    void target() {
      bool Foo = false;
      bool Bar = false;
      bool Baz = false;
      IfCond(Foo, Bar, Baz);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        auto &BarVal = getFormula(*BarDecl, Env);
        EXPECT_FALSE(Env.proves(BarVal));
        EXPECT_TRUE(Env.proves(Env.arena().makeNot(BarVal)));

        auto &BazVal = getFormula(*BazDecl, Env);
        EXPECT_TRUE(Env.proves(BazVal));
        EXPECT_FALSE(Env.proves(Env.arena().makeNot(BazVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveReturnVoid) {
  std::string Code = R"(
    void Noop() { return; }

    void target() {
      Noop();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        // This just tests that the analysis doesn't crash.
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveReturnTrue) {
  std::string Code = R"(
    bool GiveBool() { return true; }

    void target() {
      bool Foo = GiveBool();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveReturnFalse) {
  std::string Code = R"(
    bool GiveBool() { return false; }

    void target() {
      bool Foo = GiveBool();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(Env.arena().makeNot(FooVal)));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveReturnArg) {
  std::string Code = R"(
    bool GiveBool();
    bool GiveBack(bool Arg) { return Arg; }

    void target() {
      bool Foo = GiveBool();
      bool Bar = GiveBack(Foo);
      bool Baz = Foo == Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        auto &BazVal = getFormula(*BazDecl, Env);
        EXPECT_TRUE(Env.proves(BazVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveReturnInt) {
  std::string Code = R"(
    int identity(int x) { return x; }

    void target() {
      int y = identity(42);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        // This just tests that the analysis doesn't crash.
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveReturnRecord) {
  std::string Code = R"(
    struct S {
      bool B;
    };

    S makeS(bool BVal) { return {BVal}; }

    void target() {
      S FalseS = makeS(false);
      S TrueS = makeS(true);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto &FalseSLoc =
            getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "FalseS");
        auto &TrueSLoc =
            getLocForDecl<RecordStorageLocation>(ASTCtx, Env, "TrueS");

        EXPECT_EQ(getFieldValue(&FalseSLoc, "B", ASTCtx, Env),
                  &Env.getBoolLiteralValue(false));
        EXPECT_EQ(getFieldValue(&TrueSLoc, "B", ASTCtx, Env),
                  &Env.getBoolLiteralValue(true));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveMethodLiteral) {
  std::string Code = R"(
    class MyClass {
    public:
      bool giveBool() { return true; }
    };

    void target() {
      MyClass MyObj;
      bool Foo = MyObj.giveBool();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveMethodGetter) {
  std::string Code = R"(
    class MyClass {
    public:
      bool getField() { return Field; }

      bool Field;
    };

    void target() {
      MyClass MyObj;
      MyObj.Field = true;
      bool Foo = MyObj.getField();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveMethodSetter) {
  std::string Code = R"(
    class MyClass {
    public:
      void setField(bool Val) { Field = Val; }

      bool Field;
    };

    void target() {
      MyClass MyObj;
      MyObj.setField(true);
      bool Foo = MyObj.Field;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveMethodGetterAndSetter) {
  std::string Code = R"(
    class MyClass {
    public:
      bool getField() { return Field; }
      void setField(bool Val) { Field = Val; }

    private:
      bool Field;
    };

    void target() {
      MyClass MyObj;
      MyObj.setField(true);
      bool Foo = MyObj.getField();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}


TEST(TransferTest, ContextSensitiveMethodTwoLayersVoid) {
  std::string Code = R"(
    class MyClass {
    public:
      void Inner() { MyField = true; }
      void Outer() { Inner(); }

      bool MyField;
    };

    void target() {
      MyClass MyObj;
      MyObj.Outer();
      bool Foo = MyObj.MyField;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        ;
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveMethodTwoLayersReturn) {
  std::string Code = R"(
    class MyClass {
    public:
      bool Inner() { return MyField; }
      bool Outer() { return Inner(); }

      bool MyField;
    };

    void target() {
      MyClass MyObj;
      MyObj.MyField = true;
      bool Foo = MyObj.Outer();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        ;
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveConstructorBody) {
  std::string Code = R"(
    class MyClass {
    public:
      MyClass() { MyField = true; }

      bool MyField;
    };

    void target() {
      MyClass MyObj;
      bool Foo = MyObj.MyField;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveConstructorInitializer) {
  std::string Code = R"(
    class MyClass {
    public:
      MyClass() : MyField(true) {}

      bool MyField;
    };

    void target() {
      MyClass MyObj;
      bool Foo = MyObj.MyField;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveConstructorDefault) {
  std::string Code = R"(
    class MyClass {
    public:
      MyClass() = default;

      bool MyField = true;
    };

    void target() {
      MyClass MyObj;
      bool Foo = MyObj.MyField;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto &FooVal = getFormula(*FooDecl, Env);
        EXPECT_TRUE(Env.proves(FooVal));
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, ContextSensitiveSelfReferentialClass) {
  // Test that the `this` pointer seen in the constructor has the same value
  // as the address of the variable the object is constructed into.
  std::string Code = R"(
    class MyClass {
    public:
      MyClass() : Self(this) {}
      MyClass *Self;
    };

    void target() {
      MyClass MyObj;
      MyClass *SelfPtr = MyObj.Self;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));

        const ValueDecl *MyObjDecl = findValueDecl(ASTCtx, "MyObj");
        ASSERT_THAT(MyObjDecl, NotNull());

        const ValueDecl *SelfDecl = findValueDecl(ASTCtx, "SelfPtr");
        ASSERT_THAT(SelfDecl, NotNull());

        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        auto &SelfVal = *cast<PointerValue>(Env.getValue(*SelfDecl));
        EXPECT_EQ(Env.getStorageLocation(*MyObjDecl), &SelfVal.getPointeeLoc());
      },
      {BuiltinOptions{ContextSensitiveOptions{}}});
}

TEST(TransferTest, UnnamedBitfieldInitializer) {
  std::string Code = R"(
    struct B {};
    struct A {
      unsigned a;
      unsigned : 4;
      unsigned c;
      B b;
    };
    void target() {
      A a = {};
      A test = a;
      (void)test.c;
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        // This doesn't need a body because this test was crashing the framework
        // before handling correctly Unnamed bitfields in `InitListExpr`.
      });
}

// Repro for a crash that used to occur with chained short-circuiting logical
// operators.
TEST(TransferTest, ChainedLogicalOps) {
  std::string Code = R"(
    bool target() {
      bool b = true || false || false || false;
      // [[p]]
      return b;
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        auto &B = getValueForDecl<BoolValue>(ASTCtx, Env, "b").formula();
        EXPECT_TRUE(Env.proves(B));
      });
}

// Repro for a crash that used to occur when we call a `noreturn` function
// within one of the operands of a `&&` or `||` operator.
TEST(TransferTest, NoReturnFunctionInsideShortCircuitedBooleanOp) {
  std::string Code = R"(
    __attribute__((noreturn)) int doesnt_return();
    bool some_condition();
    void target(bool b1, bool b2) {
      // Neither of these should crash. In addition, if we don't terminate the
      // program, we know that the operators need to trigger the short-circuit
      // logic, so `NoreturnOnRhsOfAnd` will be false and `NoreturnOnRhsOfOr`
      // will be true.
      bool NoreturnOnRhsOfAnd = b1 && doesnt_return() > 0;
      bool NoreturnOnRhsOfOr = b2 || doesnt_return() > 0;

      // Calling a `noreturn` function on the LHS of an `&&` or `||` makes the
      // entire expression unreachable. So we know that in both of the following
      // cases, if `target()` terminates, the `else` branch was taken.
      bool NoreturnOnLhsMakesAndUnreachable = false;
      if (some_condition())
         doesnt_return() > 0 && some_condition();
      else
         NoreturnOnLhsMakesAndUnreachable = true;

      bool NoreturnOnLhsMakesOrUnreachable = false;
      if (some_condition())
         doesnt_return() > 0 || some_condition();
      else
         NoreturnOnLhsMakesOrUnreachable = true;

      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        auto &A = Env.arena();

        // Check that [[p]] is reachable with a non-false flow condition.
        EXPECT_FALSE(Env.proves(A.makeLiteral(false)));

        auto &B1 = getValueForDecl<BoolValue>(ASTCtx, Env, "b1").formula();
        EXPECT_TRUE(Env.proves(A.makeNot(B1)));

        auto &NoreturnOnRhsOfAnd =
            getValueForDecl<BoolValue>(ASTCtx, Env, "NoreturnOnRhsOfAnd").formula();
        EXPECT_TRUE(Env.proves(A.makeNot(NoreturnOnRhsOfAnd)));

        auto &B2 = getValueForDecl<BoolValue>(ASTCtx, Env, "b2").formula();
        EXPECT_TRUE(Env.proves(B2));

        auto &NoreturnOnRhsOfOr =
            getValueForDecl<BoolValue>(ASTCtx, Env, "NoreturnOnRhsOfOr")
                .formula();
        EXPECT_TRUE(Env.proves(NoreturnOnRhsOfOr));

        auto &NoreturnOnLhsMakesAndUnreachable = getValueForDecl<BoolValue>(
            ASTCtx, Env, "NoreturnOnLhsMakesAndUnreachable").formula();
        EXPECT_TRUE(Env.proves(NoreturnOnLhsMakesAndUnreachable));

        auto &NoreturnOnLhsMakesOrUnreachable = getValueForDecl<BoolValue>(
            ASTCtx, Env, "NoreturnOnLhsMakesOrUnreachable").formula();
        EXPECT_TRUE(Env.proves(NoreturnOnLhsMakesOrUnreachable));
      });
}

TEST(TransferTest, NewExpressions) {
  std::string Code = R"(
    void target() {
      int *p = new int(42);
      // [[after_new]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env =
            getEnvironmentAtAnnotation(Results, "after_new");

        auto &P = getValueForDecl<PointerValue>(ASTCtx, Env, "p");

        EXPECT_THAT(Env.getValue(P.getPointeeLoc()), NotNull());
      });
}

TEST(TransferTest, NewExpressions_Structs) {
  std::string Code = R"(
    struct Inner {
      int InnerField;
    };

    struct Outer {
      Inner OuterField;
    };

    void target() {
      Outer *p = new Outer;
      // Access the fields to make sure the analysis actually generates children
      // for them in the `RecordStorageLocation` and `RecordValue`.
      p->OuterField.InnerField;
      // [[after_new]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env =
            getEnvironmentAtAnnotation(Results, "after_new");

        const ValueDecl *OuterField = findValueDecl(ASTCtx, "OuterField");
        const ValueDecl *InnerField = findValueDecl(ASTCtx, "InnerField");

        auto &P = getValueForDecl<PointerValue>(ASTCtx, Env, "p");

        auto &OuterLoc = cast<RecordStorageLocation>(P.getPointeeLoc());
        auto &OuterFieldLoc =
            *cast<RecordStorageLocation>(OuterLoc.getChild(*OuterField));
        auto &InnerFieldLoc = *OuterFieldLoc.getChild(*InnerField);

        // Values for the struct and all fields exist after the new.
        EXPECT_THAT(Env.getValue(OuterLoc), NotNull());
        EXPECT_THAT(Env.getValue(OuterFieldLoc), NotNull());
        EXPECT_THAT(Env.getValue(InnerFieldLoc), NotNull());
      });
}

TEST(TransferTest, FunctionToPointerDecayHasValue) {
  std::string Code = R"(
    struct A { static void static_member_func(); };
    void target() {
      // To check that we're treating function-to-pointer decay correctly,
      // create two pointers, then verify they refer to the same storage
      // location.
      // We need to do the test this way because even if an initializer (in this
      // case, the function-to-pointer decay) does not create a value, we still
      // create a value for the variable.
      void (*non_member_p1)() = target;
      void (*non_member_p2)() = target;

      // Do the same thing but for a static member function.
      void (*member_p1)() = A::static_member_func;
      void (*member_p2)() = A::static_member_func;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto &NonMemberP1 =
            getValueForDecl<PointerValue>(ASTCtx, Env, "non_member_p1");
        auto &NonMemberP2 =
            getValueForDecl<PointerValue>(ASTCtx, Env, "non_member_p2");
        EXPECT_EQ(&NonMemberP1.getPointeeLoc(), &NonMemberP2.getPointeeLoc());

        auto &MemberP1 =
            getValueForDecl<PointerValue>(ASTCtx, Env, "member_p1");
        auto &MemberP2 =
            getValueForDecl<PointerValue>(ASTCtx, Env, "member_p2");
        EXPECT_EQ(&MemberP1.getPointeeLoc(), &MemberP2.getPointeeLoc());
      });
}

// Check that a builtin function is not associated with a value. (It's only
// possible to call builtin functions directly, not take their address.)
TEST(TransferTest, BuiltinFunctionModeled) {
  std::string Code = R"(
    void target() {
      __builtin_expect(0, 0);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        using ast_matchers::selectFirst;
        using ast_matchers::match;
        using ast_matchers::traverse;
        using ast_matchers::implicitCastExpr;
        using ast_matchers::hasCastKind;

        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto *ImplicitCast = selectFirst<ImplicitCastExpr>(
            "implicit_cast",
            match(traverse(TK_AsIs,
                           implicitCastExpr(hasCastKind(CK_BuiltinFnToFnPtr))
                               .bind("implicit_cast")),
                  ASTCtx));

        ASSERT_THAT(ImplicitCast, NotNull());
        EXPECT_THAT(Env.getValue(*ImplicitCast), IsNull());
      });
}

// Check that a callee of a member operator call is modeled as a `PointerValue`.
// Member operator calls are unusual in that their callee is a pointer that
// stems from a `FunctionToPointerDecay`. In calls to non-operator non-static
// member functions, the callee is a `MemberExpr` (which does not have pointer
// type).
// We want to make sure that we produce a pointer value for the callee in this
// specific scenario and that its storage location is durable (for convergence).
TEST(TransferTest, MemberOperatorCallModelsPointerForCallee) {
  std::string Code = R"(
    struct S {
      bool operator!=(S s);
    };
    void target() {
      S s;
      (void)(s != s);
      (void)(s != s);
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        using ast_matchers::selectFirst;
        using ast_matchers::match;
        using ast_matchers::traverse;
        using ast_matchers::cxxOperatorCallExpr;

        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto Matches = match(
            traverse(TK_AsIs, cxxOperatorCallExpr().bind("call")), ASTCtx);

        ASSERT_EQ(Matches.size(), 2UL);

        auto *Call1 = Matches[0].getNodeAs<CXXOperatorCallExpr>("call");
        auto *Call2 = Matches[1].getNodeAs<CXXOperatorCallExpr>("call");

        ASSERT_THAT(Call1, NotNull());
        ASSERT_THAT(Call2, NotNull());

        EXPECT_EQ(cast<ImplicitCastExpr>(Call1->getCallee())->getCastKind(),
                  CK_FunctionToPointerDecay);
        EXPECT_EQ(cast<ImplicitCastExpr>(Call2->getCallee())->getCastKind(),
                  CK_FunctionToPointerDecay);

        auto *Ptr1 = cast<PointerValue>(Env.getValue(*Call1->getCallee()));
        auto *Ptr2 = cast<PointerValue>(Env.getValue(*Call2->getCallee()));

        ASSERT_EQ(&Ptr1->getPointeeLoc(), &Ptr2->getPointeeLoc());
      });
}

// Check that fields of anonymous records are modeled.
TEST(TransferTest, AnonymousStruct) {
  std::string Code = R"(
    struct S {
      struct {
        bool b;
      };
    };
    void target() {
      S s;
      s.b = true;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        const ValueDecl *SDecl = findValueDecl(ASTCtx, "s");
        const ValueDecl *BDecl = findValueDecl(ASTCtx, "b");
        const IndirectFieldDecl *IndirectField =
            findIndirectFieldDecl(ASTCtx, "b");

        auto *S = cast<RecordStorageLocation>(Env.getStorageLocation(*SDecl));
        auto &AnonStruct = *cast<RecordStorageLocation>(
            S->getChild(*cast<ValueDecl>(IndirectField->chain().front())));

        auto *B = cast<BoolValue>(getFieldValue(&AnonStruct, *BDecl, Env));
        ASSERT_TRUE(Env.proves(B->formula()));
      });
}

TEST(TransferTest, AnonymousStructWithInitializer) {
  std::string Code = R"(
    struct target {
      target() {
        (void)0;
        // [[p]]
      }
      struct {
        bool b = true;
      };
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        const ValueDecl *BDecl = findValueDecl(ASTCtx, "b");
        const IndirectFieldDecl *IndirectField =
            findIndirectFieldDecl(ASTCtx, "b");

        auto *ThisLoc =
            cast<RecordStorageLocation>(Env.getThisPointeeStorageLocation());
        auto &AnonStruct = *cast<RecordStorageLocation>(ThisLoc->getChild(
            *cast<ValueDecl>(IndirectField->chain().front())));

        auto *B = cast<BoolValue>(getFieldValue(&AnonStruct, *BDecl, Env));
        ASSERT_TRUE(Env.proves(B->formula()));
      });
}

TEST(TransferTest, AnonymousStructWithReferenceField) {
  std::string Code = R"(
    int global_i = 0;
    struct target {
      target() {
        (void)0;
        // [[p]]
      }
      struct {
        int &i = global_i;
      };
    };
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        const ValueDecl *GlobalIDecl = findValueDecl(ASTCtx, "global_i");
        const ValueDecl *IDecl = findValueDecl(ASTCtx, "i");
        const IndirectFieldDecl *IndirectField =
            findIndirectFieldDecl(ASTCtx, "i");

        auto *ThisLoc =
            cast<RecordStorageLocation>(Env.getThisPointeeStorageLocation());
        auto &AnonStruct = *cast<RecordStorageLocation>(ThisLoc->getChild(
            *cast<ValueDecl>(IndirectField->chain().front())));

        ASSERT_EQ(AnonStruct.getChild(*IDecl),
                  Env.getStorageLocation(*GlobalIDecl));
      });
}

TEST(TransferTest, EvaluateBlockWithUnreachablePreds) {
  // This is a crash repro.
  // `false` block may not have been processed when we try to evaluate the `||`
  // after visiting `true`, because it is not necessary (and therefore the edge
  // is marked unreachable). Trying to get the analysis state via
  // `getEnvironment` for the subexpression still should not crash.
  std::string Code = R"(
    int target(int i) {
      if ((i < 0 && true) || false) {
        return 0;
      }
      return 0;
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {});
}

TEST(TransferTest, LambdaCaptureByCopy) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      [Foo]() {
        (void)0;
        // [[p]]
      }();
    }
  )";
  runDataflowOnLambda(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        EXPECT_THAT(BarLoc, IsNull());
      });
}

TEST(TransferTest, LambdaCaptureByReference) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      [&Foo]() {
        (void)0;
        // [[p]]
      }();
    }
  )";
  runDataflowOnLambda(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        EXPECT_THAT(BarLoc, IsNull());
      });
}

TEST(TransferTest, LambdaCaptureWithInitializer) {
  std::string Code = R"(
    void target(int Bar) {
      [Foo=Bar]() {
        (void)0;
        // [[p]]
      }();
    }
  )";
  runDataflowOnLambda(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        EXPECT_THAT(BarLoc, IsNull());
      });
}

TEST(TransferTest, LambdaCaptureByCopyImplicit) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      [=]() {
        Foo;
        // [[p]]
      }();
    }
  )";
  runDataflowOnLambda(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        // There is no storage location for `Bar` because it isn't used in the
        // body of the lambda.
        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        EXPECT_THAT(BarLoc, IsNull());
      });
}

TEST(TransferTest, LambdaCaptureByReferenceImplicit) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      [&]() {
        Foo;
        // [[p]]
      }();
    }
  )";
  runDataflowOnLambda(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = Env.getStorageLocation(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        // There is no storage location for `Bar` because it isn't used in the
        // body of the lambda.
        const StorageLocation *BarLoc = Env.getStorageLocation(*BarDecl);
        EXPECT_THAT(BarLoc, IsNull());
      });
}

TEST(TransferTest, LambdaCaptureThis) {
  std::string Code = R"(
    struct Bar {
      int Foo;

      void target() {
        [this]() {
          Foo;
          // [[p]]
        }();
      }
    };
  )";
  runDataflowOnLambda(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const RecordStorageLocation *ThisPointeeLoc =
            Env.getThisPointeeStorageLocation();
        ASSERT_THAT(ThisPointeeLoc, NotNull());

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc = ThisPointeeLoc->getChild(*FooDecl);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));
      });
}

TEST(TransferTest, DifferentReferenceLocInJoin) {
  // This test triggers a case where the storage location for a reference-type
  // variable is different for two states being joined. We used to believe this
  // could not happen and therefore had an assertion disallowing this; this test
  // exists to demonstrate that we can handle this condition without a failing
  // assertion. See also the discussion here:
  // https://discourse.llvm.org/t/70086/6
  std::string Code = R"(
    namespace std {
      template <class T> struct initializer_list {
        const T* begin();
        const T* end();
      };
    }

    void target(char* p, char* end) {
      while (p != end) {
        if (*p == ' ') {
          p++;
          continue;
        }

        auto && range = {1, 2};
        for (auto b = range.begin(), e = range.end(); b != e; ++b) {
        }
        (void)0;
        // [[p]]
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        // Joining environments with different storage locations for the same
        // declaration results in the declaration being removed from the joined
        // environment.
        const ValueDecl *VD = findValueDecl(ASTCtx, "range");
        ASSERT_EQ(Env.getStorageLocation(*VD), nullptr);
      });
}

// This test verifies correct modeling of a relational dependency that goes
// through unmodeled functions (the simple `cond()` in this case).
TEST(TransferTest, ConditionalRelation) {
  std::string Code = R"(
    bool cond();
    void target() {
       bool a = true;
       bool b = true;
       if (cond()) {
         a = false;
         if (cond()) {
           b = false;
         }
       }
       (void)0;
       // [[p]]
    }
 )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        auto &A = Env.arena();
        auto &VarA = getValueForDecl<BoolValue>(ASTCtx, Env, "a").formula();
        auto &VarB = getValueForDecl<BoolValue>(ASTCtx, Env, "b").formula();

        EXPECT_FALSE(Env.allows(A.makeAnd(VarA, A.makeNot(VarB))));
      });
}

} // namespace
