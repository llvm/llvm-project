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
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/LangStandard.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
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
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::UnorderedElementsAre;

using BuiltinOptions = DataflowAnalysisContext::Options;

template <typename Matcher>
void runDataflow(llvm::StringRef Code, Matcher Match,
                 DataflowAnalysisOptions Options,
                 LangStandard::Kind Std = LangStandard::lang_cxx17,
                 llvm::StringRef TargetFun = "target") {
  using ast_matchers::hasName;
  llvm::SmallVector<std::string, 3> ASTBuildArgs = {
      "-fsyntax-only", "-fno-delayed-template-parsing",
      "-std=" +
          std::string(LangStandard::getLangStandardForKind(Std).getName())};
  AnalysisInputs<NoopAnalysis> AI(
      Code, hasName(TargetFun),
      [UseBuiltinModel = Options.BuiltinOpts.has_value()](ASTContext &C,
                                                          Environment &Env) {
        return NoopAnalysis(
            C, DataflowAnalysisOptions{UseBuiltinModel
                                           ? Env.getAnalysisOptions()
                                           : std::optional<BuiltinOptions>()});
      });
  AI.ASTBuildArgs = ASTBuildArgs;
  if (Options.BuiltinOpts)
    AI.BuiltinOptions = *Options.BuiltinOpts;
  ASSERT_THAT_ERROR(
      checkDataflow<NoopAnalysis>(
          std::move(AI),
          /*VerifyResults=*/
          [&Match](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                       &Results,
                   const AnalysisOutputs &AO) { Match(Results, AO.ASTCtx); }),
      llvm::Succeeded());
}

template <typename Matcher>
void runDataflow(llvm::StringRef Code, Matcher Match,
                 LangStandard::Kind Std = LangStandard::lang_cxx17,
                 bool ApplyBuiltinTransfer = true,
                 llvm::StringRef TargetFun = "target") {
  runDataflow(Code, std::move(Match),
              {ApplyBuiltinTransfer ? BuiltinOptions{}
                                    : std::optional<BuiltinOptions>()},
              Std, TargetFun);
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

        EXPECT_EQ(Env.getStorageLocation(*FooDecl, SkipPast::None), nullptr);
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

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
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

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
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
        auto *FooValue = dyn_cast_or_null<PointerValue>(
            Env.getValue(*FooDecl, SkipPast::None));
        ASSERT_THAT(FooValue, NotNull());

        EXPECT_TRUE(isa<AggregateStorageLocation>(FooValue->getPointeeLoc()));
        auto *FooPointeeValue = Env.getValue(FooValue->getPointeeLoc());
        ASSERT_THAT(FooPointeeValue, NotNull());
        EXPECT_TRUE(isa<StructValue>(FooPointeeValue));
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
      A Foo = Bar.F1.F2.F3;
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
        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *UnmodeledDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Unmodeled") {
            UnmodeledDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(UnmodeledDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *UnmodeledLoc = &FooLoc->getChild(*UnmodeledDecl);
        ASSERT_TRUE(isa<ScalarStorageLocation>(UnmodeledLoc));
        ASSERT_THAT(Env.getValue(*UnmodeledLoc), IsNull());

        const ValueDecl *ZabDecl = findValueDecl(ASTCtx, "Zab");
        ASSERT_THAT(ZabDecl, NotNull());
        EXPECT_THAT(Env.getValue(*ZabDecl, SkipPast::None), NotNull());
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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
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

    const StorageLocation *FooLoc =
        Env1.getStorageLocation(*FooDecl, SkipPast::None);
    EXPECT_THAT(FooLoc, NotNull());
    EXPECT_THAT(Env1.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env1.getStorageLocation(*BazDecl, SkipPast::None), IsNull());

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    EXPECT_EQ(Env2.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env2.getStorageLocation(*BarDecl, SkipPast::None), NotNull());
    EXPECT_THAT(Env2.getStorageLocation(*BazDecl, SkipPast::None), IsNull());

    const Environment &Env3 = getEnvironmentAtAnnotation(Results, "p3");
    EXPECT_EQ(Env3.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env3.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env3.getStorageLocation(*BazDecl, SkipPast::None), NotNull());

    const Environment &Env4 = getEnvironmentAtAnnotation(Results, "p4");
    EXPECT_EQ(Env4.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env4.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env4.getStorageLocation(*BazDecl, SkipPast::None), IsNull());
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

        const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
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

        const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarVal);
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

  const auto &FooLoc = *cast<AggregateStorageLocation>(
      Env.getStorageLocation(*FooDecl, SkipPast::None));
  const auto &FooVal = *cast<StructValue>(Env.getValue(FooLoc));
  EXPECT_THAT(FooVal.getChild(*BarDecl), NotNull());
  EXPECT_EQ(Env.getValue(FooLoc.getChild(*BarDecl)), FooVal.getChild(*BarDecl));
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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarVal);
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
              [](ASTContext &C, Environment &) {
                return NoopAnalysis(C, /*ApplyBuiltinTransfer=*/true);
              })
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

        const auto *ThisLoc = dyn_cast<AggregateStorageLocation>(
            Env.getThisPointeeStorageLocation());
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooLoc =
            cast<ScalarStorageLocation>(&ThisLoc->getChild(*FooDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(&ThisLoc->getChild(*BarDecl));
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
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p2"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p2");

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

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST(TransferTest, ElidableConstructor) {
  // This test is effectively the same as TransferTest.TemporaryObject, but
  // the code is compiled as C++ 14.
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

TEST(TransferTest, AssignmentOperator) {
  std::string Code = R"(
    struct A {
      int Baz;
    };

    void target() {
      A Foo;
      A Bar;
      (void)Foo.Baz;
      // [[p1]]
      Foo = Bar;
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

TEST(TransferTest, CopyConstructor) {
  std::string Code = R"(
    struct A {
      int Baz;
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

        const auto &FooVal =
            *cast<StructValue>(Env.getValue(*FooDecl, SkipPast::None));
        const auto *BarVal =
            cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_EQ(BarVal, FooVal.getChild(*BazDecl));
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

        const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
        const auto *BarVal = Env.getValue(*BarDecl, SkipPast::None);
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

        const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
        const auto *BarVal = Env.getValue(*BarDecl, SkipPast::None);
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

        const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
        const auto *BarVal = Env.getValue(*BarDecl, SkipPast::None);
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

        const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
        const auto *BarVal = Env.getValue(*BarDecl, SkipPast::None);
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

        const auto *FooXVal =
            cast<PointerValue>(Env.getValue(*FooXDecl, SkipPast::None));
        const auto *FooYVal =
            cast<PointerValue>(Env.getValue(*FooYDecl, SkipPast::None));
        const auto *BarVal =
            cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
        const auto *BazVal =
            cast<PointerValue>(Env.getValue(*BazDecl, SkipPast::None));
        const auto *NullVal =
            cast<PointerValue>(Env.getValue(*NullDecl, SkipPast::None));

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
        EXPECT_TRUE(isa<AggregateStorageLocation>(BazPointeeLoc));
        EXPECT_THAT(Env.getValue(BazPointeeLoc), IsNull());

        const StorageLocation &NullPointeeLoc = NullVal->getPointeeLoc();
        EXPECT_TRUE(isa<ScalarStorageLocation>(NullPointeeLoc));
        EXPECT_THAT(Env.getValue(NullPointeeLoc), IsNull());
      });
}

TEST(TransferTest, NullToMemberPointerCast) {
  std::string Code = R"(
    struct Foo {};
    void target(Foo *Foo) {
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

        const auto *MemberPointerVal = cast<PointerValue>(
            Env.getValue(*MemberPointerDecl, SkipPast::None));

        const StorageLocation &MemberLoc = MemberPointerVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(MemberLoc), IsNull());
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

        const auto *FooLoc = cast<ScalarStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarVal =
            cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
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

        const auto *FooVal =
            cast<PointerValue>(Env.getValue(*FooDecl, SkipPast::None));
        const auto *BarVal =
            cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_EQ(&BarVal->getPointeeLoc(), &FooVal->getPointeeLoc());
      });
}

TEST(TransferTest, DerefDependentPtr) {
  std::string Code = R"(
    template <typename T>
    void target(T *Foo) {
      T &Bar = *Foo;
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

        const auto *FooVal =
            cast<PointerValue>(Env.getValue(*FooDecl, SkipPast::None));
        const auto *BarVal =
            cast<ReferenceValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_EQ(&BarVal->getReferentLoc(), &FooVal->getPointeeLoc());
      });
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

TEST(TransferTest, VarDeclInDoWhile) {
  std::string Code = R"(
    void target(int *Foo) {
      do {
        int Bar = *Foo;
      } while (true);
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

        const auto *BazLoc = dyn_cast_or_null<AggregateStorageLocation>(
            Env.getStorageLocation(*BazDecl, SkipPast::None));
        ASSERT_THAT(BazLoc, NotNull());
        ASSERT_THAT(Env.getValue(*BazLoc), NotNull());

        const auto *BazVal = cast<StructValue>(Env.getValue(*BazLoc));
        const auto *FooValFromBazVal = cast<IntegerValue>(BazVal->getChild(*FooDecl));
        const auto *FooValFromBazLoc = cast<IntegerValue>(Env.getValue(BazLoc->getChild(*FooDecl)));
        EXPECT_EQ(FooValFromBazLoc, FooValFromBazVal);

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());
        const auto *BarLoc = Env.getStorageLocation(*BarDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        EXPECT_EQ(Env.getValue(*BarLoc), FooValFromBazVal);
        EXPECT_EQ(Env.getValue(*BarLoc), FooValFromBazLoc);
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
        Code,
        [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
           ASTContext &ASTCtx) {
          ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

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
        Code,
        [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
           ASTContext &ASTCtx) {
          ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

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

        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None),
                  Env.getValue(*BarDecl, SkipPast::None));
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

        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None),
                  Env.getValue(*BarDecl, SkipPast::None));
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
        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None),
                  Env.getValue(*BarDecl, SkipPast::None));
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
        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None),
                  Env.getValue(*BarDecl, SkipPast::None));
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
        EXPECT_NE(Env.getValue(*FooDecl, SkipPast::None),
                  Env.getValue(*BarDecl, SkipPast::None));
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

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
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

        const Value *BarVal =
            cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
        const Value *BazVal =
            cast<IntegerValue>(Env.getValue(*BazDecl, SkipPast::None));
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

        const Value *BarVal =
            cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
        const Value *BazVal =
            cast<IntegerValue>(Env.getValue(*BazDecl, SkipPast::None));
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

        const Value *BarVal =
            cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));
        const Value *BazVal =
            cast<IntegerValue>(Env.getValue(*BazDecl, SkipPast::None));
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

        const auto *BarVal =
            cast<IntegerValue>(Env.getValue(*BarDecl, SkipPast::None));

        const auto *A2Val =
            cast<StructValue>(Env.getValue(*A2Decl, SkipPast::None));
        EXPECT_EQ(A2Val->getChild(*FooDecl), BarVal);
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

        auto &BarValThen =
            *cast<BoolValue>(EnvThen.getValue(*BarDecl, SkipPast::None));
        EXPECT_TRUE(EnvThen.flowConditionImplies(BarValThen));

        auto &BarValElse =
            *cast<BoolValue>(EnvElse.getValue(*BarDecl, SkipPast::None));
        EXPECT_FALSE(EnvElse.flowConditionImplies(BarValElse));
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

        auto &BarValThen =
            *cast<BoolValue>(EnvThen.getValue(*BarDecl, SkipPast::None));
        EXPECT_FALSE(EnvThen.flowConditionImplies(BarValThen));

        auto &BarValElse =
            *cast<BoolValue>(EnvElse.getValue(*BarDecl, SkipPast::None));
        EXPECT_TRUE(EnvElse.flowConditionImplies(BarValElse));
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
          auto &BVal = *cast<BoolValue>(Env.getValue(*BDecl, SkipPast::None));

          EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BVal)));
        }

        {
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p1");
          auto &CVal = *cast<BoolValue>(Env.getValue(*CDecl, SkipPast::None));
          EXPECT_TRUE(Env.flowConditionImplies(CVal));
        }

        {
          const Environment &Env = getEnvironmentAtAnnotation(Results, "p2");
          auto &CVal = *cast<BoolValue>(Env.getValue(*CDecl, SkipPast::None));
          EXPECT_TRUE(Env.flowConditionImplies(CVal));
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

        auto &BarVal = *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BarVal)));
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

        auto &BarVal = *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::None));
        auto &ErrVal = *cast<BoolValue>(Env.getValue(*ErrDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(BarVal));
        // An unsound analysis, for example only evaluating the loop once, can
        // conclude that `Err` is false. So, we test that this conclusion is not
        // reached.
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(ErrVal)));
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

        auto &BarVal =
            *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::Reference));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BarVal)));
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
        const Value *BoundFooValue =
            Env1.getValue(*BoundFooDecl, SkipPast::Reference);
        ASSERT_THAT(BoundFooValue, NotNull());
        EXPECT_TRUE(isa<BoolValue>(BoundFooValue));

        const Value *BoundBarValue =
            Env1.getValue(*BoundBarDecl, SkipPast::Reference);
        ASSERT_THAT(BoundBarValue, NotNull());
        EXPECT_TRUE(isa<IntegerValue>(BoundBarValue));

        // Test that a `DeclRefExpr` to a `BindingDecl` works as expected.
        EXPECT_EQ(Env1.getValue(*BazDecl, SkipPast::None), BoundFooValue);

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        // Test that `BoundFooDecl` retains the value we expect, after the join.
        BoundFooValue = Env2.getValue(*BoundFooDecl, SkipPast::Reference);
        EXPECT_EQ(Env2.getValue(*BazDecl, SkipPast::None), BoundFooValue);
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

        const Value *BoundFooValue =
            Env1.getValue(*BoundFooDecl, SkipPast::Reference);
        ASSERT_THAT(BoundFooValue, NotNull());
        EXPECT_TRUE(isa<BoolValue>(BoundFooValue));

        const Value *BoundBarValue =
            Env1.getValue(*BoundBarDecl, SkipPast::Reference);
        ASSERT_THAT(BoundBarValue, NotNull());
        EXPECT_TRUE(isa<IntegerValue>(BoundBarValue));

        // Test that a `DeclRefExpr` to a `BindingDecl` (with reference type)
        // works as expected. We don't test aliasing properties of the
        // reference, because we don't model `std::get` and so have no way to
        // equate separate references into the tuple.
        EXPECT_EQ(Env1.getValue(*BazDecl, SkipPast::None), BoundFooValue);

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        // Test that `BoundFooDecl` retains the value we expect, after the join.
        BoundFooValue = Env2.getValue(*BoundFooDecl, SkipPast::Reference);
        EXPECT_EQ(Env2.getValue(*BazDecl, SkipPast::None), BoundFooValue);
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

        const StorageLocation *BarLoc =
            Env.getStorageLocation(*BarDecl, SkipPast::Reference);
        ASSERT_THAT(BarLoc, NotNull());

        const StorageLocation *BazLoc =
            Env.getStorageLocation(*BazDecl, SkipPast::Reference);
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

        BoolValue &ThenFooVal =
            *cast<BoolValue>(ThenEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(ThenEnv.flowConditionImplies(ThenFooVal));

        BoolValue &ElseFooVal =
            *cast<BoolValue>(ElseEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(ElseEnv.flowConditionImplies(ElseEnv.makeNot(ElseFooVal)));
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

        BoolValue &LoopBodyFooVal =
            *cast<BoolValue>(LoopBodyEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(LoopBodyEnv.flowConditionImplies(LoopBodyFooVal));

        BoolValue &AfterLoopFooVal =
            *cast<BoolValue>(AfterLoopEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(AfterLoopEnv.flowConditionImplies(
            AfterLoopEnv.makeNot(AfterLoopFooVal)));
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

        BoolValue &LoopBodyFooVal =
            *cast<BoolValue>(LoopBodyEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(LoopBodyEnv.flowConditionImplies(LoopBodyFooVal));

        BoolValue &AfterLoopFooVal =
            *cast<BoolValue>(AfterLoopEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(AfterLoopEnv.flowConditionImplies(
            AfterLoopEnv.makeNot(AfterLoopFooVal)));
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

        BoolValue &LoopBodyFooVal =
            *cast<BoolValue>(LoopBodyEnv.getValue(*FooDecl, SkipPast::None));
        EXPECT_FALSE(LoopBodyEnv.flowConditionImplies(LoopBodyFooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_FALSE(Env.flowConditionImplies(FooVal));
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(FooVal)));
      },
      {BuiltinOptions{/*.ContextSensitiveOpts=*/std::nullopt}});
}

// This test is a regression test, based on a real crash.
TEST(TransferTest, ContextSensitiveReturnReferenceFromNonReferenceLvalue) {
  // This code exercises an unusual code path. If we return an lvalue directly,
  // the code will catch that it's an l-value based on the `Value`'s kind. If we
  // pass through a dummy function, the framework won't populate a value at
  // all. In contrast, this code results in a (fresh) value, but it is not
  // `ReferenceValue`. This test verifies that we catch this case as well.
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

        auto *Loc = Env.getReturnStorageLocation();
        ASSERT_THAT(Loc, NotNull());

        EXPECT_THAT(Env.getValue(*Loc), IsNull());
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_FALSE(Env.flowConditionImplies(FooVal));
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(FooVal)));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(FooVal)));
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

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(FooVal)));

        auto &BarVal = *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_FALSE(Env.flowConditionImplies(BarVal));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BarVal)));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_FALSE(Env.flowConditionImplies(FooVal));
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(FooVal)));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_FALSE(Env.flowConditionImplies(FooVal));
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(FooVal)));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        // ... but it also can't prove anything here.
        EXPECT_FALSE(Env.flowConditionImplies(FooVal));
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(FooVal)));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(FooVal)));

        auto &BarVal = *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_FALSE(Env.flowConditionImplies(BarVal));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BarVal)));
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

        auto &BarVal = *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::None));
        EXPECT_FALSE(Env.flowConditionImplies(BarVal));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(BarVal)));

        auto &BazVal = *cast<BoolValue>(Env.getValue(*BazDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(BazVal));
        EXPECT_FALSE(Env.flowConditionImplies(Env.makeNot(BazVal)));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(FooVal)));
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

        auto &BazVal = *cast<BoolValue>(Env.getValue(*BazDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(BazVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        auto &FooVal = *cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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

        // Check that [[p]] is reachable with a non-false flow condition.
        EXPECT_FALSE(Env.flowConditionImplies(Env.getBoolLiteralValue(false)));

        auto &B1 = getValueForDecl<BoolValue>(ASTCtx, Env, "b1");
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(B1)));

        auto &NoreturnOnRhsOfAnd =
            getValueForDecl<BoolValue>(ASTCtx, Env, "NoreturnOnRhsOfAnd");
        EXPECT_TRUE(Env.flowConditionImplies(Env.makeNot(NoreturnOnRhsOfAnd)));

        auto &B2 = getValueForDecl<BoolValue>(ASTCtx, Env, "b2");
        EXPECT_TRUE(Env.flowConditionImplies(B2));

        auto &NoreturnOnRhsOfOr =
            getValueForDecl<BoolValue>(ASTCtx, Env, "NoreturnOnRhsOfOr");
        EXPECT_TRUE(Env.flowConditionImplies(NoreturnOnRhsOfOr));

        auto &NoreturnOnLhsMakesAndUnreachable = getValueForDecl<BoolValue>(
            ASTCtx, Env, "NoreturnOnLhsMakesAndUnreachable");
        EXPECT_TRUE(Env.flowConditionImplies(NoreturnOnLhsMakesAndUnreachable));

        auto &NoreturnOnLhsMakesOrUnreachable = getValueForDecl<BoolValue>(
            ASTCtx, Env, "NoreturnOnLhsMakesOrUnreachable");
        EXPECT_TRUE(Env.flowConditionImplies(NoreturnOnLhsMakesOrUnreachable));
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
      // for them in the `AggregateStorageLoc` and `StructValue`.
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

        auto &OuterLoc = cast<AggregateStorageLocation>(P.getPointeeLoc());
        auto &OuterFieldLoc =
            cast<AggregateStorageLocation>(OuterLoc.getChild(*OuterField));
        auto &InnerFieldLoc = OuterFieldLoc.getChild(*InnerField);

        // Values for the struct and all fields exist after the new.
        EXPECT_THAT(Env.getValue(OuterLoc), NotNull());
        EXPECT_THAT(Env.getValue(OuterFieldLoc), NotNull());
        EXPECT_THAT(Env.getValue(InnerFieldLoc), NotNull());
      });
}

} // namespace
