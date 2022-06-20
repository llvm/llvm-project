//===- unittests/Analysis/FlowSensitive/TransferTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoopAnalysis.h"
#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/LangStandard.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <utility>

namespace {

using namespace clang;
using namespace dataflow;
using namespace test;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Pair;
using ::testing::SizeIs;

class TransferTest : public ::testing::Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match,
                   LangStandard::Kind Std = LangStandard::lang_cxx17,
                   bool ApplyBuiltinTransfer = true,
                   llvm::StringRef TargetFun = "target") {
    ASSERT_THAT_ERROR(
        test::checkDataflow<NoopAnalysis>(
            Code, TargetFun,
            [ApplyBuiltinTransfer](ASTContext &C, Environment &) {
              return NoopAnalysis(C, ApplyBuiltinTransfer);
            },
            [&Match](
                llvm::ArrayRef<
                    std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                    Results,
                ASTContext &ASTCtx) { Match(Results, ASTCtx); },
            {"-fsyntax-only", "-fno-delayed-template-parsing",
             "-std=" +
                 std::string(
                     LangStandard::getLangStandardForKind(Std).getName())}),
        llvm::Succeeded());
  }
};

TEST_F(TransferTest, IntVarDeclNotTrackedWhenTransferDisabled) {
  std::string Code = R"(
    void target() {
      int Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](llvm::ArrayRef<
             std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
             Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        EXPECT_EQ(Env.getStorageLocation(*FooDecl, SkipPast::None), nullptr);
      },
      LangStandard::lang_cxx17,
      /*ApplyBuiltinTransfer=*/false);
}

TEST_F(TransferTest, BoolVarDecl) {
  std::string Code = R"(
    void target() {
      bool Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const StorageLocation *FooLoc =
                    Env.getStorageLocation(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

                const Value *FooVal = Env.getValue(*FooLoc);
                EXPECT_TRUE(isa_and_nonnull<BoolValue>(FooVal));
              });
}

TEST_F(TransferTest, IntVarDecl) {
  std::string Code = R"(
    void target() {
      int Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const StorageLocation *FooLoc =
                    Env.getStorageLocation(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

                const Value *FooVal = Env.getValue(*FooLoc);
                EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));
              });
}

TEST_F(TransferTest, StructVarDecl) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target() {
      A Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, StructVarDeclWithInit) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    A Gen();

    void target() {
      A Foo = Gen();
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ClassVarDecl) {
  std::string Code = R"(
    class A {
      int Bar;
    };

    void target() {
      A Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ReferenceVarDecl) {
  std::string Code = R"(
    struct A {};

    A &getA();

    void target() {
      A &Foo = getA();
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const ReferenceValue *FooVal =
            cast<ReferenceValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooReferentLoc = FooVal->getReferentLoc();
        EXPECT_TRUE(isa<AggregateStorageLocation>(&FooReferentLoc));

        const Value *FooReferentVal = Env.getValue(FooReferentLoc);
        EXPECT_TRUE(isa_and_nonnull<StructValue>(FooReferentVal));
      });
}

TEST_F(TransferTest, SelfReferentialReferenceVarDecl) {
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
      // [[p]]
    }
  )";
  runDataflow(Code, [](llvm::ArrayRef<std::pair<
                           std::string, DataflowAnalysisState<NoopLattice>>>
                           Results,
                       ASTContext &ASTCtx) {
    ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
    const Environment &Env = Results[0].second.Env;

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

    const auto *FooLoc = cast<ScalarStorageLocation>(
        Env.getStorageLocation(*FooDecl, SkipPast::None));
    const auto *FooVal = cast<ReferenceValue>(Env.getValue(*FooLoc));
    const auto *FooReferentVal =
        cast<StructValue>(Env.getValue(FooVal->getReferentLoc()));

    const auto *BarVal =
        cast<ReferenceValue>(FooReferentVal->getChild(*BarDecl));
    const auto *BarReferentVal =
        cast<StructValue>(Env.getValue(BarVal->getReferentLoc()));

    const auto *FooRefVal =
        cast<ReferenceValue>(BarReferentVal->getChild(*FooRefDecl));
    const StorageLocation &FooReferentLoc = FooRefVal->getReferentLoc();
    EXPECT_THAT(Env.getValue(FooReferentLoc), IsNull());

    const auto *FooPtrVal =
        cast<PointerValue>(BarReferentVal->getChild(*FooPtrDecl));
    const StorageLocation &FooPtrPointeeLoc = FooPtrVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

    const auto *BazRefVal =
        cast<ReferenceValue>(BarReferentVal->getChild(*BazRefDecl));
    const StorageLocation &BazReferentLoc = BazRefVal->getReferentLoc();
    EXPECT_THAT(Env.getValue(BazReferentLoc), NotNull());

    const auto *BazPtrVal =
        cast<PointerValue>(BarReferentVal->getChild(*BazPtrDecl));
    const StorageLocation &BazPtrPointeeLoc = BazPtrVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(BazPtrPointeeLoc), NotNull());
  });
}

TEST_F(TransferTest, PointerVarDecl) {
  std::string Code = R"(
    struct A {};

    A *getA();

    void target() {
      A *Foo = getA();
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const PointerValue *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        EXPECT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        EXPECT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
      });
}

TEST_F(TransferTest, SelfReferentialPointerVarDecl) {
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
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto *FooLoc = cast<ScalarStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const auto *FooPointeeVal =
            cast<StructValue>(Env.getValue(FooVal->getPointeeLoc()));

        const auto *BarVal =
            cast<PointerValue>(FooPointeeVal->getChild(*BarDecl));
        const auto *BarPointeeVal =
            cast<StructValue>(Env.getValue(BarVal->getPointeeLoc()));

        const auto *FooRefVal =
            cast<ReferenceValue>(BarPointeeVal->getChild(*FooRefDecl));
        const StorageLocation &FooReferentLoc = FooRefVal->getReferentLoc();
        EXPECT_THAT(Env.getValue(FooReferentLoc), IsNull());

        const auto *FooPtrVal =
            cast<PointerValue>(BarPointeeVal->getChild(*FooPtrDecl));
        const StorageLocation &FooPtrPointeeLoc = FooPtrVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

        const auto *BazRefVal =
            cast<ReferenceValue>(BarPointeeVal->getChild(*BazRefDecl));
        const StorageLocation &BazReferentLoc = BazRefVal->getReferentLoc();
        EXPECT_THAT(Env.getValue(BazReferentLoc), NotNull());

        const auto *BazPtrVal =
            cast<PointerValue>(BarPointeeVal->getChild(*BazPtrDecl));
        const StorageLocation &BazPtrPointeeLoc = BazPtrVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(BazPtrPointeeLoc), NotNull());
      });
}

TEST_F(TransferTest, MultipleVarsDecl) {
  std::string Code = R"(
    void target() {
      int Foo, Bar;
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const StorageLocation *FooLoc =
                    Env.getStorageLocation(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

                const StorageLocation *BarLoc =
                    Env.getStorageLocation(*BarDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

                const Value *FooVal = Env.getValue(*FooLoc);
                EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const Value *BarVal = Env.getValue(*BarLoc);
                EXPECT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));
              });
}

TEST_F(TransferTest, JoinVarDecl) {
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
  runDataflow(Code, [](llvm::ArrayRef<std::pair<
                           std::string, DataflowAnalysisState<NoopLattice>>>
                           Results,
                       ASTContext &ASTCtx) {
    ASSERT_THAT(Results, ElementsAre(Pair("p4", _), Pair("p3", _),
                                     Pair("p2", _), Pair("p1", _)));
    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
    ASSERT_THAT(BarDecl, NotNull());

    const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
    ASSERT_THAT(BazDecl, NotNull());

    const Environment &Env1 = Results[3].second.Env;
    const StorageLocation *FooLoc =
        Env1.getStorageLocation(*FooDecl, SkipPast::None);
    EXPECT_THAT(FooLoc, NotNull());
    EXPECT_THAT(Env1.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env1.getStorageLocation(*BazDecl, SkipPast::None), IsNull());

    const Environment &Env2 = Results[2].second.Env;
    EXPECT_EQ(Env2.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env2.getStorageLocation(*BarDecl, SkipPast::None), NotNull());
    EXPECT_THAT(Env2.getStorageLocation(*BazDecl, SkipPast::None), IsNull());

    const Environment &Env3 = Results[1].second.Env;
    EXPECT_EQ(Env3.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env3.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env3.getStorageLocation(*BazDecl, SkipPast::None), NotNull());

    const Environment &Env4 = Results[0].second.Env;
    EXPECT_EQ(Env4.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env4.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env4.getStorageLocation(*BazDecl, SkipPast::None), IsNull());
  });
}

TEST_F(TransferTest, BinaryOperatorAssign) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar;
      (Bar) = (Foo);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssign) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssignChained) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar;
      int Baz = (Bar = Foo);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
                EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssignPtrDeref) {
  std::string Code = R"(
    void target() {
      int Foo;
      int *Bar;
      *(Bar) = Foo;
      int Baz = *(Bar);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *BarVal =
                    cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
                EXPECT_EQ(Env.getValue(BarVal->getPointeeLoc()), FooVal);

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, AssignToAndFromReference) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p1", _), Pair("p2", _)));
        const Environment &Env1 = Results[0].second.Env;
        const Environment &Env2 = Results[1].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Value *FooVal = Env1.getValue(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const Value *BarVal = Env1.getValue(*BarDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env1.getValue(*BazDecl, SkipPast::Reference), FooVal);

        EXPECT_EQ(Env2.getValue(*BazDecl, SkipPast::Reference), BarVal);
        EXPECT_EQ(Env2.getValue(*FooDecl, SkipPast::None), BarVal);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());
        EXPECT_EQ(Env2.getValue(*QuxDecl, SkipPast::None), BarVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env2.getValue(*QuuxDecl, SkipPast::Reference), BarVal);
      });
}

TEST_F(TransferTest, MultipleParamDecls) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const StorageLocation *FooLoc =
                    Env.getStorageLocation(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

                const Value *FooVal = Env.getValue(*FooLoc);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const StorageLocation *BarLoc =
                    Env.getStorageLocation(*BarDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

                const Value *BarVal = Env.getValue(*BarLoc);
                EXPECT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));
              });
}

TEST_F(TransferTest, StructParamDecl) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target(A Foo) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ReferenceParamDecl) {
  std::string Code = R"(
    struct A {};

    void target(A &Foo) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const ReferenceValue *FooVal =
            dyn_cast<ReferenceValue>(Env.getValue(*FooLoc));
        ASSERT_THAT(FooVal, NotNull());

        const StorageLocation &FooReferentLoc = FooVal->getReferentLoc();
        EXPECT_TRUE(isa<AggregateStorageLocation>(&FooReferentLoc));

        const Value *FooReferentVal = Env.getValue(FooReferentLoc);
        EXPECT_TRUE(isa_and_nonnull<StructValue>(FooReferentVal));
      });
}

TEST_F(TransferTest, PointerParamDecl) {
  std::string Code = R"(
    struct A {};

    void target(A *Foo) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const PointerValue *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        EXPECT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        EXPECT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
      });
}

TEST_F(TransferTest, StructMember) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarVal);
      });
}

TEST_F(TransferTest, DerivedBaseMemberClass) {
  std::string Code = R"(
    class A {
      int ADefault;
    protected:
      int AProtected;
    private:
      int APrivate;
    public:
      int APublic;
    };

    class B : public A {
      int BDefault;
    protected:
      int BProtected;
    private:
      int BPrivate;
    };

    void target() {
      B Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto &FooLoc = *cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto &FooVal = *cast<StructValue>(Env.getValue(FooLoc));

        // Note: we can't test presence of children in `FooLoc`, because
        // `getChild` requires its argument be present (or fails an assert). So,
        // we limit to testing presence in `FooVal` and coherence between the
        // two.

        // Base-class fields.
        EXPECT_THAT(FooVal.getChild(*ADefaultDecl), NotNull());
        EXPECT_THAT(FooVal.getChild(*APrivateDecl), NotNull());

        EXPECT_THAT(FooVal.getChild(*AProtectedDecl), NotNull());
        EXPECT_EQ(Env.getValue(FooLoc.getChild(*APublicDecl)),
                  FooVal.getChild(*APublicDecl));
        EXPECT_THAT(FooVal.getChild(*APublicDecl), NotNull());
        EXPECT_EQ(Env.getValue(FooLoc.getChild(*AProtectedDecl)),
                  FooVal.getChild(*AProtectedDecl));

        // Derived-class fields.
        EXPECT_THAT(FooVal.getChild(*BDefaultDecl), NotNull());
        EXPECT_EQ(Env.getValue(FooLoc.getChild(*BDefaultDecl)),
                  FooVal.getChild(*BDefaultDecl));
        EXPECT_THAT(FooVal.getChild(*BProtectedDecl), NotNull());
        EXPECT_EQ(Env.getValue(FooLoc.getChild(*BProtectedDecl)),
                  FooVal.getChild(*BProtectedDecl));
        EXPECT_THAT(FooVal.getChild(*BPrivateDecl), NotNull());
        EXPECT_EQ(Env.getValue(FooLoc.getChild(*BPrivateDecl)),
                  FooVal.getChild(*BPrivateDecl));
      });
}

static void derivedBaseMemberExpectations(
    llvm::ArrayRef<std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
        Results,
    ASTContext &ASTCtx) {
  ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
  const Environment &Env = Results[0].second.Env;

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

  const auto &FooLoc = *cast<AggregateStorageLocation>(
      Env.getStorageLocation(*FooDecl, SkipPast::None));
  const auto &FooVal = *cast<StructValue>(Env.getValue(FooLoc));
  EXPECT_THAT(FooVal.getChild(*BarDecl), NotNull());
  EXPECT_EQ(Env.getValue(FooLoc.getChild(*BarDecl)), FooVal.getChild(*BarDecl));
}

TEST_F(TransferTest, DerivedBaseMemberStructDefault) {
  std::string Code = R"(
    struct A {
      int Bar;
    };
    struct B : public A {
    };

    void target() {
      B Foo;
      // [[p]]
    }
  )";
  runDataflow(Code, derivedBaseMemberExpectations);
}

TEST_F(TransferTest, DerivedBaseMemberPrivateFriend) {
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

TEST_F(TransferTest, ClassMember) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarVal);
      });
}

TEST_F(TransferTest, BaseClassInitializer) {
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
      test::checkDataflow<NoopAnalysis>(
          Code, cxxConstructorDecl(ofClass(hasName("B"))),
          [](ASTContext &C, Environment &) {
            return NoopAnalysis(C, /*ApplyBuiltinTransfer=*/true);
          },
          [](llvm::ArrayRef<
                 std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                 Results,
             ASTContext &ASTCtx) {
            // Regression test to verify that base-class initializers do not
            // trigger an assertion. If we add support for such initializers in
            // the future, we can expand this test to check more specific
            // properties.
            EXPECT_THAT(Results, ElementsAre(Pair("p", _)));
          },
          {"-fsyntax-only", "-fno-delayed-template-parsing",
           "-std=" + std::string(LangStandard::getLangStandardForKind(
                                     LangStandard::lang_cxx17)
                                     .getName())}),
      llvm::Succeeded());
}

TEST_F(TransferTest, ReferenceMember) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<ReferenceValue>(FooVal->getChild(*BarDecl));
        const auto *BarReferentVal =
            cast<IntegerValue>(Env.getValue(BarVal->getReferentLoc()));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarReferentVal);
      });
}

TEST_F(TransferTest, StructThisMember) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const auto *ThisLoc = dyn_cast<AggregateStorageLocation>(
            Env.getThisPointeeStorageLocation());
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(&ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None), BarVal);

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
            cast<AggregateStorageLocation>(&ThisLoc->getChild(*QuxDecl));
        const auto *QuxVal = dyn_cast<StructValue>(Env.getValue(*QuxLoc));
        ASSERT_THAT(QuxVal, NotNull());

        const auto *BazLoc =
            cast<ScalarStorageLocation>(&QuxLoc->getChild(*BazDecl));
        const auto *BazVal = cast<IntegerValue>(QuxVal->getChild(*BazDecl));
        EXPECT_EQ(Env.getValue(*BazLoc), BazVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuuxDecl, SkipPast::None), BazVal);
      });
}

TEST_F(TransferTest, ClassThisMember) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const auto *ThisLoc =
            cast<AggregateStorageLocation>(Env.getThisPointeeStorageLocation());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(&ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None), BarVal);

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
            cast<AggregateStorageLocation>(&ThisLoc->getChild(*QuxDecl));
        const auto *QuxVal = dyn_cast<StructValue>(Env.getValue(*QuxLoc));
        ASSERT_THAT(QuxVal, NotNull());

        const auto *BazLoc =
            cast<ScalarStorageLocation>(&QuxLoc->getChild(*BazDecl));
        const auto *BazVal = cast<IntegerValue>(QuxVal->getChild(*BazDecl));
        EXPECT_EQ(Env.getValue(*BazLoc), BazVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuuxDecl, SkipPast::None), BazVal);
      });
}

TEST_F(TransferTest, StructThisInLambda) {
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
      [](llvm::ArrayRef<
             std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
             Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p1", _)));
        const Environment &Env = Results[0].second.Env;

        const auto *ThisLoc = dyn_cast<AggregateStorageLocation>(
            Env.getThisPointeeStorageLocation());
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(&ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None), BarVal);
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
      [](llvm::ArrayRef<
             std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
             Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p2", _)));
        const Environment &Env = Results[0].second.Env;

        const auto *ThisLoc = dyn_cast<AggregateStorageLocation>(
            Env.getThisPointeeStorageLocation());
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(&ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None), BarVal);
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
      [](llvm::ArrayRef<
             std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
             Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p3", _)));
        const Environment &Env = Results[0].second.Env;

        EXPECT_THAT(Env.getThisPointeeStorageLocation(), IsNull());
      },
      LangStandard::lang_cxx17, /*ApplyBuiltinTransfer=*/true, "operator()");
}

TEST_F(TransferTest, ConstructorInitializer) {
  std::string Code = R"(
    struct target {
      int Bar;

      target(int Foo) : Bar(Foo) {
        int Qux = Bar;
        // [[p]]
      }
    };
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const auto *ThisLoc = dyn_cast<AggregateStorageLocation>(
                    Env.getThisPointeeStorageLocation());
                ASSERT_THAT(ThisLoc, NotNull());

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const auto *FooVal =
                    cast<IntegerValue>(Env.getValue(*FooDecl, SkipPast::None));

                const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
                ASSERT_THAT(QuxDecl, NotNull());
                EXPECT_EQ(Env.getValue(*QuxDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, DefaultInitializer) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const auto *ThisLoc = dyn_cast<AggregateStorageLocation>(
                    Env.getThisPointeeStorageLocation());
                ASSERT_THAT(ThisLoc, NotNull());

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const auto *FooVal =
                    cast<IntegerValue>(Env.getValue(*FooDecl, SkipPast::None));

                const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
                ASSERT_THAT(QuxDecl, NotNull());
                EXPECT_EQ(Env.getValue(*QuxDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, DefaultInitializerReference) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const auto *ThisLoc = dyn_cast<AggregateStorageLocation>(
            Env.getThisPointeeStorageLocation());
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooVal =
            cast<ReferenceValue>(Env.getValue(*FooDecl, SkipPast::None));

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        const auto *QuxVal =
            cast<ReferenceValue>(Env.getValue(*QuxDecl, SkipPast::None));
        EXPECT_EQ(&QuxVal->getReferentLoc(), &FooVal->getReferentLoc());
      });
}

TEST_F(TransferTest, TemporaryObject) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target() {
      A Foo = A();
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ElidableConstructor) {
  // This test is effectively the same as TransferTest.TemporaryObject, but
  // the code is compiled as C++ 14.
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target() {
      A Foo = A();
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](llvm::ArrayRef<
             std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
             Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      },
      LangStandard::lang_cxx14);
}

TEST_F(TransferTest, AssignmentOperator) {
  std::string Code = R"(
    struct A {
      int Baz;
    };

    void target() {
      A Foo;
      A Bar;
      // [[p1]]
      Foo = Bar;
      // [[p2]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p1", _), Pair("p2", _)));
        const Environment &Env1 = Results[0].second.Env;
        const Environment &Env2 = Results[1].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooLoc1 = cast<AggregateStorageLocation>(
            Env1.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc1 = cast<AggregateStorageLocation>(
            Env1.getStorageLocation(*BarDecl, SkipPast::None));

        const auto *FooVal1 = cast<StructValue>(Env1.getValue(*FooLoc1));
        const auto *BarVal1 = cast<StructValue>(Env1.getValue(*BarLoc1));
        EXPECT_NE(FooVal1, BarVal1);

        const auto *FooBazVal1 =
            cast<IntegerValue>(Env1.getValue(FooLoc1->getChild(*BazDecl)));
        const auto *BarBazVal1 =
            cast<IntegerValue>(Env1.getValue(BarLoc1->getChild(*BazDecl)));
        EXPECT_NE(FooBazVal1, BarBazVal1);

        const auto *FooLoc2 = cast<AggregateStorageLocation>(
            Env2.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc2 = cast<AggregateStorageLocation>(
            Env2.getStorageLocation(*BarDecl, SkipPast::None));

        const auto *FooVal2 = cast<StructValue>(Env2.getValue(*FooLoc2));
        const auto *BarVal2 = cast<StructValue>(Env2.getValue(*BarLoc2));
        EXPECT_EQ(FooVal2, BarVal2);

        const auto *FooBazVal2 =
            cast<IntegerValue>(Env2.getValue(FooLoc1->getChild(*BazDecl)));
        const auto *BarBazVal2 =
            cast<IntegerValue>(Env2.getValue(BarLoc1->getChild(*BazDecl)));
        EXPECT_EQ(FooBazVal2, BarBazVal2);
      });
}

TEST_F(TransferTest, CopyConstructor) {
  std::string Code = R"(
    struct A {
      int Baz;
    };

    void target() {
      A Foo;
      A Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*BarDecl, SkipPast::None));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<StructValue>(Env.getValue(*BarLoc));
        EXPECT_EQ(FooVal, BarVal);

        const auto *FooBazVal =
            cast<IntegerValue>(Env.getValue(FooLoc->getChild(*BazDecl)));
        const auto *BarBazVal =
            cast<IntegerValue>(Env.getValue(BarLoc->getChild(*BazDecl)));
        EXPECT_EQ(FooBazVal, BarBazVal);
      });
}

TEST_F(TransferTest, CopyConstructorWithParens) {
  std::string Code = R"(
    struct A {
      int Baz;
    };

    void target() {
      A Foo;
      A Bar((A(Foo)));
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*BarDecl, SkipPast::None));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<StructValue>(Env.getValue(*BarLoc));
        EXPECT_EQ(FooVal, BarVal);

        const auto *FooBazVal =
            cast<IntegerValue>(Env.getValue(FooLoc->getChild(*BazDecl)));
        const auto *BarBazVal =
            cast<IntegerValue>(Env.getValue(BarLoc->getChild(*BazDecl)));
        EXPECT_EQ(FooBazVal, BarBazVal);
      });
}

TEST_F(TransferTest, MoveConstructor) {
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
      // [[p1]]
      Foo = std::move(Bar);
      // [[p2]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p1", _), Pair("p2", _)));
        const Environment &Env1 = Results[0].second.Env;
        const Environment &Env2 = Results[1].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooLoc1 = cast<AggregateStorageLocation>(
            Env1.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc1 = cast<AggregateStorageLocation>(
            Env1.getStorageLocation(*BarDecl, SkipPast::None));

        const auto *FooVal1 = cast<StructValue>(Env1.getValue(*FooLoc1));
        const auto *BarVal1 = cast<StructValue>(Env1.getValue(*BarLoc1));
        EXPECT_NE(FooVal1, BarVal1);

        const auto *FooBazVal1 =
            cast<IntegerValue>(Env1.getValue(FooLoc1->getChild(*BazDecl)));
        const auto *BarBazVal1 =
            cast<IntegerValue>(Env1.getValue(BarLoc1->getChild(*BazDecl)));
        EXPECT_NE(FooBazVal1, BarBazVal1);

        const auto *FooLoc2 = cast<AggregateStorageLocation>(
            Env2.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal2 = cast<StructValue>(Env2.getValue(*FooLoc2));
        EXPECT_EQ(FooVal2, BarVal1);

        const auto *FooBazVal2 =
            cast<IntegerValue>(Env2.getValue(FooLoc1->getChild(*BazDecl)));
        EXPECT_EQ(FooBazVal2, BarBazVal1);
      });
}

TEST_F(TransferTest, BindTemporary) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                const auto &FooVal =
                    *cast<StructValue>(Env.getValue(*FooDecl, SkipPast::None));
                const auto *BarVal =
                    cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
                EXPECT_EQ(BarVal, FooVal.getChild(*BazDecl));
              });
}

TEST_F(TransferTest, StaticCast) {
  std::string Code = R"(
    void target(int Foo) {
      int Bar = static_cast<int>(Foo);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                const auto *BarVal = Env.getValue(*BarDecl, SkipPast::None);
                EXPECT_TRUE(isa<IntegerValue>(FooVal));
                EXPECT_TRUE(isa<IntegerValue>(BarVal));
                EXPECT_EQ(FooVal, BarVal);
              });
}

TEST_F(TransferTest, IntegralCast) {
  std::string Code = R"(
    void target(int Foo) {
      long Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                const auto *BarVal = Env.getValue(*BarDecl, SkipPast::None);
                EXPECT_TRUE(isa<IntegerValue>(FooVal));
                EXPECT_TRUE(isa<IntegerValue>(BarVal));
                EXPECT_EQ(FooVal, BarVal);
              });
}

TEST_F(TransferTest, IntegraltoBooleanCast) {
  std::string Code = R"(
    void target(int Foo) {
      bool Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                const auto *BarVal = Env.getValue(*BarDecl, SkipPast::None);
                EXPECT_TRUE(isa<IntegerValue>(FooVal));
                EXPECT_TRUE(isa<BoolValue>(BarVal));
              });
}

TEST_F(TransferTest, IntegralToBooleanCastFromBool) {
  std::string Code = R"(
    void target(bool Foo) {
      int Zab = Foo;
      bool Bar = Zab;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                const auto *BarVal = Env.getValue(*BarDecl, SkipPast::None);
                EXPECT_TRUE(isa<BoolValue>(FooVal));
                EXPECT_TRUE(isa<BoolValue>(BarVal));
                EXPECT_EQ(FooVal, BarVal);
              });
}

TEST_F(TransferTest, AddrOfValue) {
  std::string Code = R"(
    void target() {
      int Foo;
      int *Bar = &Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *FooLoc = cast<ScalarStorageLocation>(
                    Env.getStorageLocation(*FooDecl, SkipPast::None));
                const auto *BarVal =
                    cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
                EXPECT_EQ(&BarVal->getPointeeLoc(), FooLoc);
              });
}

TEST_F(TransferTest, AddrOfReference) {
  std::string Code = R"(
    void target(int *Foo) {
      int *Bar = &(*Foo);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *FooVal =
                    cast<PointerValue>(Env.getValue(*FooDecl, SkipPast::None));
                const auto *BarVal =
                    cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
                EXPECT_EQ(&BarVal->getPointeeLoc(), &FooVal->getPointeeLoc());
              });
}

TEST_F(TransferTest, DerefDependentPtr) {
  std::string Code = R"(
    template <typename T>
    void target(T *Foo) {
      T &Bar = *Foo;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooVal =
            cast<PointerValue>(Env.getValue(*FooDecl, SkipPast::None));
        const auto *BarVal =
            cast<ReferenceValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_EQ(&BarVal->getReferentLoc(), &FooVal->getPointeeLoc());
      });
}

TEST_F(TransferTest, VarDeclInitAssignConditionalOperator) {
  std::string Code = R"(
    struct A {};

    void target(A Foo, A Bar, bool Cond) {
      A Baz = Cond ?  Foo : Bar;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *FooVal =
            cast<StructValue>(Env.getValue(*FooDecl, SkipPast::None));
        const auto *BarVal =
            cast<StructValue>(Env.getValue(*BarDecl, SkipPast::None));

        const auto *BazVal =
            dyn_cast<StructValue>(Env.getValue(*BazDecl, SkipPast::None));
        ASSERT_THAT(BazVal, NotNull());

        EXPECT_NE(BazVal, FooVal);
        EXPECT_NE(BazVal, BarVal);
      });
}

TEST_F(TransferTest, VarDeclInDoWhile) {
  std::string Code = R"(
    void target(int *Foo) {
      do {
        int Bar = *Foo;
      } while (true);
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *FooVal =
                    cast<PointerValue>(Env.getValue(*FooDecl, SkipPast::None));
                const auto *FooPointeeVal =
                    cast<IntegerValue>(Env.getValue(FooVal->getPointeeLoc()));

                const auto *BarVal = dyn_cast_or_null<IntegerValue>(
                    Env.getValue(*BarDecl, SkipPast::None));
                ASSERT_THAT(BarVal, NotNull());

                EXPECT_EQ(BarVal, FooPointeeVal);
              });
}

TEST_F(TransferTest, AggregateInitialization) {
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
      /*[[p]]*/
    }
  )";
  std::string BraceEllisionCode = R"(
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
      /*[[p]]*/
    }
  )";
  for (const std::string &Code : {BracesCode, BraceEllisionCode}) {
    runDataflow(
        Code, [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
          ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
          const Environment &Env = Results[0].second.Env;

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

          const auto *FooArgVal =
              cast<IntegerValue>(Env.getValue(*FooArgDecl, SkipPast::None));
          const auto *BarArgVal =
              cast<IntegerValue>(Env.getValue(*BarArgDecl, SkipPast::None));
          const auto *QuxArgVal =
              cast<IntegerValue>(Env.getValue(*QuxArgDecl, SkipPast::None));

          const auto *QuuxVal =
              cast<StructValue>(Env.getValue(*QuuxDecl, SkipPast::None));
          ASSERT_THAT(QuuxVal, NotNull());

          const auto *BazVal = cast<StructValue>(QuuxVal->getChild(*BazDecl));
          ASSERT_THAT(BazVal, NotNull());

          EXPECT_EQ(QuuxVal->getChild(*BarDecl), BarArgVal);
          EXPECT_EQ(BazVal->getChild(*FooDecl), FooArgVal);
          EXPECT_EQ(QuuxVal->getChild(*QuxDecl), QuxArgVal);
        });
  }
}

TEST_F(TransferTest, AssignToUnionMember) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());
                ASSERT_TRUE(BazDecl->getType()->isUnionType());

                const auto *BazLoc = dyn_cast_or_null<AggregateStorageLocation>(
                    Env.getStorageLocation(*BazDecl, SkipPast::None));
                ASSERT_THAT(BazLoc, NotNull());

                // FIXME: Add support for union types.
                EXPECT_THAT(Env.getValue(*BazLoc), IsNull());
              });
}

TEST_F(TransferTest, AssignFromBoolLiteral) {
  std::string Code = R"(
    void target() {
      bool Foo = true;
      bool Bar = false;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const auto *FooVal = dyn_cast_or_null<AtomicBoolValue>(
                    Env.getValue(*FooDecl, SkipPast::None));
                ASSERT_THAT(FooVal, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *BarVal = dyn_cast_or_null<AtomicBoolValue>(
                    Env.getValue(*BarDecl, SkipPast::None));
                ASSERT_THAT(BarVal, NotNull());

                EXPECT_EQ(FooVal, &Env.getBoolLiteralValue(true));
                EXPECT_EQ(BarVal, &Env.getBoolLiteralValue(false));
              });
}

TEST_F(TransferTest, AssignFromCompositeBoolExpression) {
  {
    std::string Code = R"(
    void target(bool Foo, bool Bar, bool Qux) {
      bool Baz = (Foo) && (Bar || Qux);
      // [[p]]
    }
  )";
    runDataflow(
        Code, [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
          ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
          const Environment &Env = Results[0].second.Env;

          const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
          ASSERT_THAT(FooDecl, NotNull());

          const auto *FooVal = dyn_cast_or_null<BoolValue>(
              Env.getValue(*FooDecl, SkipPast::None));
          ASSERT_THAT(FooVal, NotNull());

          const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
          ASSERT_THAT(BarDecl, NotNull());

          const auto *BarVal = dyn_cast_or_null<BoolValue>(
              Env.getValue(*BarDecl, SkipPast::None));
          ASSERT_THAT(BarVal, NotNull());

          const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
          ASSERT_THAT(QuxDecl, NotNull());

          const auto *QuxVal = dyn_cast_or_null<BoolValue>(
              Env.getValue(*QuxDecl, SkipPast::None));
          ASSERT_THAT(QuxVal, NotNull());

          const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
          ASSERT_THAT(BazDecl, NotNull());

          const auto *BazVal = dyn_cast_or_null<ConjunctionValue>(
              Env.getValue(*BazDecl, SkipPast::None));
          ASSERT_THAT(BazVal, NotNull());
          EXPECT_EQ(&BazVal->getLeftSubValue(), FooVal);

          const auto *BazRightSubValVal =
              cast<DisjunctionValue>(&BazVal->getRightSubValue());
          EXPECT_EQ(&BazRightSubValVal->getLeftSubValue(), BarVal);
          EXPECT_EQ(&BazRightSubValVal->getRightSubValue(), QuxVal);
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
        Code, [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
          ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
          const Environment &Env = Results[0].second.Env;

          const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
          ASSERT_THAT(FooDecl, NotNull());

          const auto *FooVal = dyn_cast_or_null<BoolValue>(
              Env.getValue(*FooDecl, SkipPast::None));
          ASSERT_THAT(FooVal, NotNull());

          const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
          ASSERT_THAT(BarDecl, NotNull());

          const auto *BarVal = dyn_cast_or_null<BoolValue>(
              Env.getValue(*BarDecl, SkipPast::None));
          ASSERT_THAT(BarVal, NotNull());

          const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
          ASSERT_THAT(QuxDecl, NotNull());

          const auto *QuxVal = dyn_cast_or_null<BoolValue>(
              Env.getValue(*QuxDecl, SkipPast::None));
          ASSERT_THAT(QuxVal, NotNull());

          const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
          ASSERT_THAT(BazDecl, NotNull());

          const auto *BazVal = dyn_cast_or_null<DisjunctionValue>(
              Env.getValue(*BazDecl, SkipPast::None));
          ASSERT_THAT(BazVal, NotNull());

          const auto *BazLeftSubValVal =
              cast<ConjunctionValue>(&BazVal->getLeftSubValue());
          EXPECT_EQ(&BazLeftSubValVal->getLeftSubValue(), FooVal);
          EXPECT_EQ(&BazLeftSubValVal->getRightSubValue(), QuxVal);

          EXPECT_EQ(&BazVal->getRightSubValue(), BarVal);
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
        Code, [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
          ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
          const Environment &Env = Results[0].second.Env;

          const ValueDecl *ADecl = findValueDecl(ASTCtx, "A");
          ASSERT_THAT(ADecl, NotNull());

          const auto *AVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*ADecl, SkipPast::None));
          ASSERT_THAT(AVal, NotNull());

          const ValueDecl *BDecl = findValueDecl(ASTCtx, "B");
          ASSERT_THAT(BDecl, NotNull());

          const auto *BVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*BDecl, SkipPast::None));
          ASSERT_THAT(BVal, NotNull());

          const ValueDecl *CDecl = findValueDecl(ASTCtx, "C");
          ASSERT_THAT(CDecl, NotNull());

          const auto *CVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*CDecl, SkipPast::None));
          ASSERT_THAT(CVal, NotNull());

          const ValueDecl *DDecl = findValueDecl(ASTCtx, "D");
          ASSERT_THAT(DDecl, NotNull());

          const auto *DVal =
              dyn_cast_or_null<BoolValue>(Env.getValue(*DDecl, SkipPast::None));
          ASSERT_THAT(DVal, NotNull());

          const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
          ASSERT_THAT(FooDecl, NotNull());

          const auto *FooVal = dyn_cast_or_null<ConjunctionValue>(
              Env.getValue(*FooDecl, SkipPast::None));
          ASSERT_THAT(FooVal, NotNull());

          const auto &FooLeftSubVal =
              cast<ConjunctionValue>(FooVal->getLeftSubValue());
          const auto &FooLeftLeftSubVal =
              cast<ConjunctionValue>(FooLeftSubVal.getLeftSubValue());
          EXPECT_EQ(&FooLeftLeftSubVal.getLeftSubValue(), AVal);
          EXPECT_EQ(&FooLeftLeftSubVal.getRightSubValue(), BVal);
          EXPECT_EQ(&FooLeftSubVal.getRightSubValue(), CVal);
          EXPECT_EQ(&FooVal->getRightSubValue(), DVal);
        });
  }
}

TEST_F(TransferTest, AssignFromBoolNegation) {
  std::string Code = R"(
    void target() {
      bool Foo = true;
      bool Bar = !(Foo);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const auto *FooVal = dyn_cast_or_null<AtomicBoolValue>(
                    Env.getValue(*FooDecl, SkipPast::None));
                ASSERT_THAT(FooVal, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *BarVal = dyn_cast_or_null<NegationValue>(
                    Env.getValue(*BarDecl, SkipPast::None));
                ASSERT_THAT(BarVal, NotNull());

                EXPECT_EQ(&BarVal->getSubVal(), FooVal);
              });
}

TEST_F(TransferTest, BuiltinExpect) {
  std::string Code = R"(
    void target(long Foo) {
      long Bar = __builtin_expect(Foo, true);
      /*[[p]]*/
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const auto &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None),
                          Env.getValue(*BarDecl, SkipPast::None));
              });
}

// `__builtin_expect` takes and returns a `long` argument, so other types
// involve casts. This verifies that we identify the input and output in that
// case.
TEST_F(TransferTest, BuiltinExpectBoolArg) {
  std::string Code = R"(
    void target(bool Foo) {
      bool Bar = __builtin_expect(Foo, true);
      /*[[p]]*/
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const auto &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None),
                          Env.getValue(*BarDecl, SkipPast::None));
              });
}

TEST_F(TransferTest, BuiltinUnreachable) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const auto &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                // `__builtin_unreachable` promises that the code is
                // unreachable, so the compiler treats the "then" branch as the
                // only possible predecessor of this statement.
                EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None),
                          Env.getValue(*BarDecl, SkipPast::None));
              });
}

TEST_F(TransferTest, BuiltinTrap) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const auto &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                // `__builtin_trap` ensures program termination, so only the
                // "then" branch is a predecessor of this statement.
                EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None),
                          Env.getValue(*BarDecl, SkipPast::None));
              });
}

TEST_F(TransferTest, BuiltinDebugTrap) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const auto &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                // `__builtin_debugtrap` doesn't ensure program termination.
                EXPECT_NE(Env.getValue(*FooDecl, SkipPast::None),
                          Env.getValue(*BarDecl, SkipPast::None));
              });
}

TEST_F(TransferTest, StaticIntSingleVarDecl) {
  std::string Code = R"(
    void target() {
      static int Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const StorageLocation *FooLoc =
                    Env.getStorageLocation(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

                const Value *FooVal = Env.getValue(*FooLoc);
                EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));
              });
}

TEST_F(TransferTest, StaticIntGroupVarDecl) {
  std::string Code = R"(
    void target() {
      static int Foo, Bar;
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const StorageLocation *FooLoc =
                    Env.getStorageLocation(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

                const StorageLocation *BarLoc =
                    Env.getStorageLocation(*BarDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

                const Value *FooVal = Env.getValue(*FooLoc);
                EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const Value *BarVal = Env.getValue(*BarLoc);
                EXPECT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

                EXPECT_NE(FooVal, BarVal);
              });
}

TEST_F(TransferTest, GlobalIntVarDecl) {
  std::string Code = R"(
    static int Foo;

    void target() {
      int Bar = Foo;
      int Baz = Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                const Value *BarVal =
                    cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
                const Value *BazVal =
                    cast<IntegerValue>(Env.getValue(*BazDecl, SkipPast::None));
                EXPECT_EQ(BarVal, BazVal);
              });
}

TEST_F(TransferTest, StaticMemberIntVarDecl) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                const Value *BarVal =
                    cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
                const Value *BazVal =
                    cast<IntegerValue>(Env.getValue(*BazDecl, SkipPast::None));
                EXPECT_EQ(BarVal, BazVal);
              });
}

TEST_F(TransferTest, StaticMemberRefVarDecl) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                const Value *BarVal =
                    cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
                const Value *BazVal =
                    cast<IntegerValue>(Env.getValue(*BazDecl, SkipPast::None));
                EXPECT_EQ(BarVal, BazVal);
              });
}

TEST_F(TransferTest, AssignMemberBeforeCopy) {
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *A1Decl = findValueDecl(ASTCtx, "A1");
                ASSERT_THAT(A1Decl, NotNull());

                const ValueDecl *A2Decl = findValueDecl(ASTCtx, "A2");
                ASSERT_THAT(A2Decl, NotNull());

                const auto *BarVal =
                    cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));

                const auto *A2Val =
                    cast<StructValue>(Env.getValue(*A2Decl, SkipPast::None));
                EXPECT_EQ(A2Val->getChild(*FooDecl), BarVal);
              });
}

TEST_F(TransferTest, BooleanEquality) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p-else", _), Pair("p-then", _)));
        const Environment &EnvElse = Results[0].second.Env;
        const Environment &EnvThen = Results[1].second.Env;

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarValThen =
            *cast<BoolValue>(EnvThen.getValue(*BarDecl, SkipPast::None));
        EXPECT_TRUE(EnvThen.flowConditionImplies(BarValThen));

        auto &BarValElse =
            *cast<BoolValue>(EnvElse.getValue(*BarDecl, SkipPast::None));
        EXPECT_FALSE(EnvElse.flowConditionImplies(BarValElse));
      });
}

TEST_F(TransferTest, BooleanInequality) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p-else", _), Pair("p-then", _)));
        const Environment &EnvElse = Results[0].second.Env;
        const Environment &EnvThen = Results[1].second.Env;

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarValThen =
            *cast<BoolValue>(EnvThen.getValue(*BarDecl, SkipPast::None));
        EXPECT_FALSE(EnvThen.flowConditionImplies(BarValThen));

        auto &BarValElse =
            *cast<BoolValue>(EnvElse.getValue(*BarDecl, SkipPast::None));
        EXPECT_TRUE(EnvElse.flowConditionImplies(BarValElse));
      });
}

TEST_F(TransferTest, CorrelatedBranches) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, SizeIs(3));

        const ValueDecl *CDecl = findValueDecl(ASTCtx, "C");
        ASSERT_THAT(CDecl, NotNull());

        {
          ASSERT_THAT(Results[2], Pair("p0", _));
          const Environment &Env = Results[2].second.Env;
          const ValueDecl *BDecl = findValueDecl(ASTCtx, "B");
          ASSERT_THAT(BDecl, NotNull());
          auto &BVal = *cast<BoolValue>(Env.getValue(*BDecl, SkipPast::None));

          EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BVal)));
        }

        {
          ASSERT_THAT(Results[1], Pair("p1", _));
          const Environment &Env = Results[1].second.Env;
          auto &CVal = *cast<BoolValue>(Env.getValue(*CDecl, SkipPast::None));
          EXPECT_TRUE(Env.flowConditionImplies(CVal));
        }

        {
          ASSERT_THAT(Results[0], Pair("p2", _));
          const Environment &Env = Results[0].second.Env;
          auto &CVal = *cast<BoolValue>(Env.getValue(*CDecl, SkipPast::None));
          EXPECT_TRUE(Env.flowConditionImplies(CVal));
        }
      });
}

TEST_F(TransferTest, LoopWithAssignmentConverges) {
  std::string Code = R"(

    bool &foo();

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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal = *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BarVal)));
      });
}

TEST_F(TransferTest, LoopWithReferenceAssignmentConverges) {
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
  // The key property that we are verifying is implicit in `runDataflow` --
  // namely, that the analysis succeeds, rather than hitting the maximum number
  // of iterations.
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal =
            *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::Reference));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BarVal)));
      });
}

TEST_F(TransferTest, LoopWithStructReferenceAssignmentConverges) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results,
                    ElementsAre(Pair("p-outer", _), Pair("p-inner", _)));
        const Environment &OuterEnv = Results[0].second.Env;
        const Environment &InnerEnv = Results[1].second.Env;

        const ValueDecl *ValDecl = findValueDecl(ASTCtx, "val");
        ASSERT_THAT(ValDecl, NotNull());

        const ValueDecl *LDecl = findValueDecl(ASTCtx, "l");
        ASSERT_THAT(LDecl, NotNull());

        // Inner.
        auto *LVal =
            dyn_cast<PointerValue>(InnerEnv.getValue(*LDecl, SkipPast::None));
        ASSERT_THAT(LVal, NotNull());

        EXPECT_EQ(&LVal->getPointeeLoc(),
                  InnerEnv.getStorageLocation(*ValDecl, SkipPast::Reference));

        // Outer.
        LVal =
            dyn_cast<PointerValue>(OuterEnv.getValue(*LDecl, SkipPast::None));
        ASSERT_THAT(LVal, NotNull());

        // The loop body may not have been executed, so we should not conclude
        // that `l` points to `val`.
        EXPECT_NE(&LVal->getPointeeLoc(),
                  OuterEnv.getStorageLocation(*ValDecl, SkipPast::Reference));
      });
}

TEST_F(TransferTest, DoesNotCrashOnUnionThisExpr) {
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
      [](llvm::ArrayRef<
             std::pair<std::string, DataflowAnalysisState<NoopLattice>>>,
         ASTContext &) {},
      LangStandard::lang_cxx17, /*ApplyBuiltinTransfer=*/true, "operator=");
}

TEST_F(TransferTest, StructuredBindingAssignFromStructIntMembersToRefs) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const StorageLocation *FooRefLoc =
            Env.getStorageLocation(*FooRefDecl, SkipPast::Reference);
        ASSERT_THAT(FooRefLoc, NotNull());

        const StorageLocation *BarRefLoc =
            Env.getStorageLocation(*BarRefDecl, SkipPast::Reference);
        ASSERT_THAT(BarRefLoc, NotNull());

        const Value *QuxVal = Env.getValue(*QuxDecl, SkipPast::None);
        ASSERT_THAT(QuxVal, NotNull());

        const StorageLocation *BoundFooRefLoc =
            Env.getStorageLocation(*BoundFooRefDecl, SkipPast::Reference);
        EXPECT_EQ(BoundFooRefLoc, FooRefLoc);

        const StorageLocation *BoundBarRefLoc =
            Env.getStorageLocation(*BoundBarRefDecl, SkipPast::Reference);
        EXPECT_EQ(BoundBarRefLoc, BarRefLoc);

        EXPECT_EQ(Env.getValue(*BoundFooRefDecl, SkipPast::Reference), QuxVal);
      });
}

TEST_F(TransferTest, StructuredBindingAssignFromStructRefMembersToRefs) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const StorageLocation *FooRefLoc =
            Env.getStorageLocation(*FooRefDecl, SkipPast::Reference);
        ASSERT_THAT(FooRefLoc, NotNull());

        const StorageLocation *BarRefLoc =
            Env.getStorageLocation(*BarRefDecl, SkipPast::Reference);
        ASSERT_THAT(BarRefLoc, NotNull());

        const Value *QuxVal = Env.getValue(*QuxDecl, SkipPast::None);
        ASSERT_THAT(QuxVal, NotNull());

        const StorageLocation *BoundFooRefLoc =
            Env.getStorageLocation(*BoundFooRefDecl, SkipPast::Reference);
        EXPECT_EQ(BoundFooRefLoc, FooRefLoc);

        const StorageLocation *BoundBarRefLoc =
            Env.getStorageLocation(*BoundBarRefDecl, SkipPast::Reference);
        EXPECT_EQ(BoundBarRefLoc, BarRefLoc);

        EXPECT_EQ(Env.getValue(*BoundFooRefDecl, SkipPast::Reference), QuxVal);
      });
}

TEST_F(TransferTest, StructuredBindingAssignFromStructIntMembersToInts) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

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

        const StorageLocation *FooRefLoc =
            Env.getStorageLocation(*FooRefDecl, SkipPast::Reference);
        ASSERT_THAT(FooRefLoc, NotNull());

        const StorageLocation *BarRefLoc =
            Env.getStorageLocation(*BarRefDecl, SkipPast::Reference);
        ASSERT_THAT(BarRefLoc, NotNull());

        const Value *QuxVal = Env.getValue(*QuxDecl, SkipPast::None);
        ASSERT_THAT(QuxVal, NotNull());

        const StorageLocation *BoundFooLoc =
            Env.getStorageLocation(*BoundFooDecl, SkipPast::Reference);
        EXPECT_NE(BoundFooLoc, FooRefLoc);

        const StorageLocation *BoundBarLoc =
            Env.getStorageLocation(*BoundBarDecl, SkipPast::Reference);
        EXPECT_NE(BoundBarLoc, BarRefLoc);

        EXPECT_EQ(Env.getValue(*BoundFooDecl, SkipPast::Reference), QuxVal);
      });
}

TEST_F(TransferTest, BinaryOperatorComma) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      int &Baz = (Foo, Bar);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                const StorageLocation *BarLoc =
                    Env.getStorageLocation(*BarDecl, SkipPast::Reference);
                ASSERT_THAT(BarLoc, NotNull());

                const StorageLocation *BazLoc =
                    Env.getStorageLocation(*BazDecl, SkipPast::Reference);
                EXPECT_EQ(BazLoc, BarLoc);
              });
}

TEST_F(TransferTest, IfStmtBranchExtendsFlowCondition) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results,
                    ElementsAre(Pair("if_else", _), Pair("if_then", _)));
        const Environment &ThenEnv = Results[1].second.Env;
        const Environment &ElseEnv = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        BoolValue &ThenFooVal =
            *cast<BoolValue>(ThenEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(ThenEnv.flowConditionImplies(ThenFooVal));

        BoolValue &ElseFooVal =
            *cast<BoolValue>(ElseEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(ElseEnv.flowConditionImplies(ElseEnv.makeNot(ElseFooVal)));
      });
}

TEST_F(TransferTest, WhileStmtBranchExtendsFlowCondition) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results,
                    ElementsAre(Pair("after_loop", _), Pair("loop_body", _)));
        const Environment &LoopBodyEnv = Results[1].second.Env;
        const Environment &AfterLoopEnv = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        BoolValue &LoopBodyFooVal =
            *cast<BoolValue>(LoopBodyEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(LoopBodyEnv.flowConditionImplies(LoopBodyFooVal));

        BoolValue &AfterLoopFooVal =
            *cast<BoolValue>(AfterLoopEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(AfterLoopEnv.flowConditionImplies(
            AfterLoopEnv.makeNot(AfterLoopFooVal)));
      });
}

TEST_F(TransferTest, DoWhileStmtBranchExtendsFlowCondition) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results,
                    ElementsAre(Pair("after_loop", _), Pair("loop_body", _)));
        const Environment &LoopBodyEnv = Results[1].second.Env;
        const Environment &AfterLoopEnv = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        BoolValue &LoopBodyFooVal =
            *cast<BoolValue>(LoopBodyEnv.getValue(*FooDecl, SkipPast::None));
        BoolValue &LoopBodyBarVal =
            *cast<BoolValue>(LoopBodyEnv.getValue(*BarDecl, SkipPast::None));
        EXPECT_TRUE(LoopBodyEnv.flowConditionImplies(
            LoopBodyEnv.makeOr(LoopBodyBarVal, LoopBodyFooVal)));

        BoolValue &AfterLoopFooVal =
            *cast<BoolValue>(AfterLoopEnv.getValue(*FooDecl, SkipPast::None));
        BoolValue &AfterLoopBarVal =
            *cast<BoolValue>(AfterLoopEnv.getValue(*BarDecl, SkipPast::None));
        EXPECT_TRUE(AfterLoopEnv.flowConditionImplies(
            AfterLoopEnv.makeNot(AfterLoopFooVal)));
        EXPECT_TRUE(AfterLoopEnv.flowConditionImplies(
            AfterLoopEnv.makeNot(AfterLoopBarVal)));
      });
}

TEST_F(TransferTest, ForStmtBranchExtendsFlowCondition) {
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results,
                    ElementsAre(Pair("after_loop", _), Pair("loop_body", _)));
        const Environment &LoopBodyEnv = Results[1].second.Env;
        const Environment &AfterLoopEnv = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        BoolValue &LoopBodyFooVal =
            *cast<BoolValue>(LoopBodyEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(LoopBodyEnv.flowConditionImplies(LoopBodyFooVal));

        BoolValue &AfterLoopFooVal =
            *cast<BoolValue>(AfterLoopEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(AfterLoopEnv.flowConditionImplies(
            AfterLoopEnv.makeNot(AfterLoopFooVal)));
      });
}

} // namespace
