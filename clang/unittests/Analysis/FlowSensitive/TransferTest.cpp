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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <string>
#include <utility>

namespace {

using namespace clang;
using namespace dataflow;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Pair;

class TransferTest : public ::testing::Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match) {
    test::checkDataflow<NoopAnalysis>(
        Code, "target",
        [](ASTContext &C, Environment &) { return NoopAnalysis(C); },
        [&Match](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) { Match(Results, ASTCtx); },
        {"-fsyntax-only", "-std=c++17"});
  }
};

/// Returns the `ValueDecl` for the given identifier.
///
/// Requirements:
///
///  `Name` must be unique in `ASTCtx`.
static const ValueDecl *findValueDecl(ASTContext &ASTCtx,
                                      llvm::StringRef Name) {
  auto TargetNodes = ast_matchers::match(
      ast_matchers::valueDecl(ast_matchers::hasName(Name)).bind("v"), ASTCtx);
  assert(TargetNodes.size() == 1 && "Name must be unique");
  auto *const Result = ast_matchers::selectFirst<ValueDecl>("v", TargetNodes);
  assert(Result != nullptr);
  return Result;
}

TEST_F(TransferTest, IntVarDecl) {
  std::string Code = R"(
    void target() {
      int foo;
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

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));
      });
}

TEST_F(TransferTest, StructVarDecl) {
  std::string Code = R"(
    struct Foo {
      int Bar;
    };

    void target() {
      Foo foo;
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

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
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
        ASSERT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ClassVarDecl) {
  std::string Code = R"(
    class Foo {
      int Bar;
    };

    void target() {
      Foo foo;
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

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
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
        ASSERT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ReferenceVarDecl) {
  std::string Code = R"(
    struct Foo {};

    Foo& getFoo();

    void target() {
      Foo& foo = getFoo();
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

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const ReferenceValue *FooVal =
            cast<ReferenceValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        ASSERT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        ASSERT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
      });
}

TEST_F(TransferTest, SelfReferentialReferenceVarDecl) {
  std::string Code = R"(
    struct Foo;

    struct Baz {};

    struct Bar {
      Foo& FooRef;
      Foo* FooPtr;
      Baz& BazRef;
      Baz* BazPtr;
    };

    struct Foo {
      Bar& Bar;
    };

    Foo& getFoo();

    void target() {
      Foo& foo = getFoo();
      // [[p]]
    }
  )";
  runDataflow(Code, [](llvm::ArrayRef<std::pair<
                           std::string, DataflowAnalysisState<NoopLattice>>>
                           Results,
                       ASTContext &ASTCtx) {
    ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
    const Environment &Env = Results[0].second.Env;

    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
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
    ASSERT_THAT(Env.getValue(FooRefPointeeLoc), IsNull());

    const auto *FooPtrVal =
        cast<PointerValue>(&BarPointeeVal->getChild(*FooPtrDecl));
    const StorageLocation &FooPtrPointeeLoc = FooPtrVal->getPointeeLoc();
    ASSERT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

    const auto *BazRefVal =
        cast<ReferenceValue>(&BarPointeeVal->getChild(*BazRefDecl));
    const StorageLocation &BazRefPointeeLoc = BazRefVal->getPointeeLoc();
    ASSERT_THAT(Env.getValue(BazRefPointeeLoc), NotNull());

    const auto *BazPtrVal =
        cast<PointerValue>(&BarPointeeVal->getChild(*BazPtrDecl));
    const StorageLocation &BazPtrPointeeLoc = BazPtrVal->getPointeeLoc();
    ASSERT_THAT(Env.getValue(BazPtrPointeeLoc), NotNull());
  });
}

TEST_F(TransferTest, PointerVarDecl) {
  std::string Code = R"(
    struct Foo {};

    Foo* getFoo();

    void target() {
      Foo* foo = getFoo();
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

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const PointerValue *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        ASSERT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        ASSERT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
      });
}

TEST_F(TransferTest, SelfReferentialPointerVarDecl) {
  std::string Code = R"(
    struct Foo;

    struct Baz {};

    struct Bar {
      Foo& FooRef;
      Foo* FooPtr;
      Baz& BazRef;
      Baz* BazPtr;
    };

    struct Foo {
      Bar* Bar;
    };

    Foo* getFoo();

    void target() {
      Foo* foo = getFoo();
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

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
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
        ASSERT_THAT(Env.getValue(FooRefPointeeLoc), IsNull());

        const auto *FooPtrVal =
            cast<PointerValue>(&BarPointeeVal->getChild(*FooPtrDecl));
        const StorageLocation &FooPtrPointeeLoc = FooPtrVal->getPointeeLoc();
        ASSERT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

        const auto *BazRefVal =
            cast<ReferenceValue>(&BarPointeeVal->getChild(*BazRefDecl));
        const StorageLocation &BazRefPointeeLoc = BazRefVal->getPointeeLoc();
        ASSERT_THAT(Env.getValue(BazRefPointeeLoc), NotNull());

        const auto *BazPtrVal =
            cast<PointerValue>(&BarPointeeVal->getChild(*BazPtrDecl));
        const StorageLocation &BazPtrPointeeLoc = BazPtrVal->getPointeeLoc();
        ASSERT_THAT(Env.getValue(BazPtrPointeeLoc), NotNull());
      });
}

TEST_F(TransferTest, JoinVarDecl) {
  std::string Code = R"(
    void target(bool b) {
      int foo;
      // [[p1]]
      if (b) {
        int bar;
        // [[p2]]
      } else {
        int baz;
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
    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
    ASSERT_THAT(FooDecl, NotNull());

    const ValueDecl *BarDecl = findValueDecl(ASTCtx, "bar");
    ASSERT_THAT(BarDecl, NotNull());

    const ValueDecl *BazDecl = findValueDecl(ASTCtx, "baz");
    ASSERT_THAT(BazDecl, NotNull());

    const Environment &Env1 = Results[3].second.Env;
    const StorageLocation *FooLoc =
        Env1.getStorageLocation(*FooDecl, SkipPast::None);
    ASSERT_THAT(FooLoc, NotNull());
    ASSERT_THAT(Env1.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    ASSERT_THAT(Env1.getStorageLocation(*BazDecl, SkipPast::None), IsNull());

    const Environment &Env2 = Results[2].second.Env;
    ASSERT_EQ(Env2.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    ASSERT_THAT(Env2.getStorageLocation(*BarDecl, SkipPast::None), NotNull());
    ASSERT_THAT(Env2.getStorageLocation(*BazDecl, SkipPast::None), IsNull());

    const Environment &Env3 = Results[1].second.Env;
    ASSERT_EQ(Env3.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    ASSERT_THAT(Env3.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    ASSERT_THAT(Env3.getStorageLocation(*BazDecl, SkipPast::None), NotNull());

    const Environment &Env4 = Results[0].second.Env;
    ASSERT_EQ(Env4.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    ASSERT_THAT(Env4.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    ASSERT_THAT(Env4.getStorageLocation(*BazDecl, SkipPast::None), IsNull());
  });
}

TEST_F(TransferTest, BinaryOperatorAssign) {
  std::string Code = R"(
    void target() {
      int foo;
      int bar;
      (bar) = (foo);
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

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "bar");
                ASSERT_THAT(BarDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssign) {
  std::string Code = R"(
    void target() {
      int foo;
      int bar = foo;
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

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "bar");
                ASSERT_THAT(BarDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssignChained) {
  std::string Code = R"(
    void target() {
      int foo;
      int bar;
      int baz = (bar = foo);
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

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "baz");
                ASSERT_THAT(BazDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
                EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssignPtrDeref) {
  std::string Code = R"(
    void target() {
      int foo;
      int *bar;
      *(bar) = foo;
      int baz = *(bar);
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

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *BarVal =
                    cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
                EXPECT_EQ(Env.getValue(BarVal->getPointeeLoc()), FooVal);

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "baz");
                ASSERT_THAT(BazDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, AssignToAndFromReference) {
  std::string Code = R"(
    void target() {
      int foo;
      int bar;
      int& baz = foo;
      // [[p1]]
      baz = bar;
      int qux = baz;
      int& quux = baz;
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

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Value *FooVal = Env1.getValue(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "bar");
        ASSERT_THAT(BarDecl, NotNull());

        const Value *BarVal = Env1.getValue(*BarDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env1.getValue(*BazDecl, SkipPast::Reference), FooVal);

        EXPECT_EQ(Env2.getValue(*BazDecl, SkipPast::Reference), BarVal);
        EXPECT_EQ(Env2.getValue(*FooDecl, SkipPast::None), BarVal);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "qux");
        ASSERT_THAT(QuxDecl, NotNull());
        EXPECT_EQ(Env2.getValue(*QuxDecl, SkipPast::None), BarVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env2.getValue(*QuuxDecl, SkipPast::Reference), BarVal);
      });
}

} // namespace
