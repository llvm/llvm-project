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

class TransferTest : public ::testing::Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match,
                   LangStandard::Kind Std = LangStandard::lang_cxx17,
                   bool ApplyBuiltinTransfer = true) {
    ASSERT_THAT_ERROR(
        test::checkDataflow<NoopAnalysis>(
            Code, "target",
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

TEST_F(TransferTest, IntVarDecl) {
  std::string Code = R"(
    void target() {
      int Foo;
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
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));
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
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));
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
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        EXPECT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        EXPECT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
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
    const auto *FooPointeeVal =
        cast<StructValue>(Env.getValue(FooVal->getPointeeLoc()));

    const auto *BarVal =
        cast<ReferenceValue>(&FooPointeeVal->getChild(*BarDecl));
    const auto *BarPointeeVal =
        cast<StructValue>(Env.getValue(BarVal->getPointeeLoc()));

    const auto *FooRefVal =
        cast<ReferenceValue>(&BarPointeeVal->getChild(*FooRefDecl));
    const StorageLocation &FooRefPointeeLoc = FooRefVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(FooRefPointeeLoc), IsNull());

    const auto *FooPtrVal =
        cast<PointerValue>(&BarPointeeVal->getChild(*FooPtrDecl));
    const StorageLocation &FooPtrPointeeLoc = FooPtrVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

    const auto *BazRefVal =
        cast<ReferenceValue>(&BarPointeeVal->getChild(*BazRefDecl));
    const StorageLocation &BazRefPointeeLoc = BazRefVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(BazRefPointeeLoc), NotNull());

    const auto *BazPtrVal =
        cast<PointerValue>(&BarPointeeVal->getChild(*BazPtrDecl));
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
            cast<PointerValue>(&FooPointeeVal->getChild(*BarDecl));
        const auto *BarPointeeVal =
            cast<StructValue>(Env.getValue(BarVal->getPointeeLoc()));

        const auto *FooRefVal =
            cast<ReferenceValue>(&BarPointeeVal->getChild(*FooRefDecl));
        const StorageLocation &FooRefPointeeLoc = FooRefVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(FooRefPointeeLoc), IsNull());

        const auto *FooPtrVal =
            cast<PointerValue>(&BarPointeeVal->getChild(*FooPtrDecl));
        const StorageLocation &FooPtrPointeeLoc = FooPtrVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

        const auto *BazRefVal =
            cast<ReferenceValue>(&BarPointeeVal->getChild(*BazRefDecl));
        const StorageLocation &BazRefPointeeLoc = BazRefVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(BazRefPointeeLoc), NotNull());

        const auto *BazPtrVal =
            cast<PointerValue>(&BarPointeeVal->getChild(*BazPtrDecl));
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
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));
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

                const ReferenceValue *FooVal =
                    dyn_cast<ReferenceValue>(Env.getValue(*FooLoc));
                ASSERT_THAT(FooVal, NotNull());

                const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
                EXPECT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

                const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
                EXPECT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
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
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarVal);
      });
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
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarVal);
      });
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
        const auto *BarVal = cast<ReferenceValue>(&FooVal->getChild(*BarDecl));
        const auto *BarPointeeVal =
            cast<IntegerValue>(Env.getValue(BarVal->getPointeeLoc()));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarPointeeVal);
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
        const auto *BazVal = cast<IntegerValue>(&QuxVal->getChild(*BazDecl));
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
        const auto *BazVal = cast<IntegerValue>(&QuxVal->getChild(*BazDecl));
        EXPECT_EQ(Env.getValue(*BazLoc), BazVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuuxDecl, SkipPast::None), BazVal);
      });
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
        EXPECT_EQ(&QuxVal->getPointeeLoc(), &FooVal->getPointeeLoc());
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
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));
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
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));
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
                EXPECT_EQ(BarVal, &FooVal.getChild(*BazDecl));
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

                const auto *FooVal =
                    cast<IntegerValue>(Env.getValue(*FooDecl, SkipPast::None));
                const auto *BarVal =
                    cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
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
        EXPECT_EQ(&BarVal->getPointeeLoc(), &FooVal->getPointeeLoc());
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

          const auto *BazVal = cast<StructValue>(&QuuxVal->getChild(*BazDecl));
          ASSERT_THAT(BazVal, NotNull());

          EXPECT_EQ(&QuuxVal->getChild(*BarDecl), BarArgVal);
          EXPECT_EQ(&BazVal->getChild(*FooDecl), FooArgVal);
          EXPECT_EQ(&QuuxVal->getChild(*QuxDecl), QuxArgVal);
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

TEST_F(TransferTest, AssignFromBoolConjunction) {
  std::string Code = R"(
    void target() {
      bool Foo = true;
      bool Bar = true;
      bool Baz = (Foo) && (Bar);
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

        const auto *FooVal =
            dyn_cast_or_null<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        ASSERT_THAT(FooVal, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarVal =
            dyn_cast_or_null<BoolValue>(Env.getValue(*BarDecl, SkipPast::None));
        ASSERT_THAT(BarVal, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *BazVal = dyn_cast_or_null<ConjunctionValue>(
            Env.getValue(*BazDecl, SkipPast::None));
        ASSERT_THAT(BazVal, NotNull());

        EXPECT_EQ(&BazVal->getLeftSubValue(), FooVal);
        EXPECT_EQ(&BazVal->getRightSubValue(), BarVal);
      });
}

TEST_F(TransferTest, AssignFromBoolDisjunction) {
  std::string Code = R"(
    void target() {
      bool Foo = true;
      bool Bar = true;
      bool Baz = (Foo) || (Bar);
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

        const auto *FooVal =
            dyn_cast_or_null<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        ASSERT_THAT(FooVal, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarVal =
            dyn_cast_or_null<BoolValue>(Env.getValue(*BarDecl, SkipPast::None));
        ASSERT_THAT(BarVal, NotNull());

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        const auto *BazVal = dyn_cast_or_null<DisjunctionValue>(
            Env.getValue(*BazDecl, SkipPast::None));
        ASSERT_THAT(BazVal, NotNull());

        EXPECT_EQ(&BazVal->getLeftSubValue(), FooVal);
        EXPECT_EQ(&BazVal->getRightSubValue(), BarVal);
      });
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

} // namespace
