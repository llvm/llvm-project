//===- OperatorNewDeletePointersExtractorTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindDecl.h"
#include "TestFixture.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Analyses/OperatorNewDelete/OperatorNewDeletePointers.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>

using namespace clang;
using namespace ssaf;

namespace {

/// Look up the \p SummaryT entity summary for the contributor named
/// \p ContributorName.
///
/// \tparam SummaryT        The concrete EntitySummary subtype to return.
/// \tparam ContributorT    The NamedDecl subtype to search for (defaults to
///                         FunctionDecl).
/// \tparam TUSummaryDataT  The type of the TUSummary data map, deduced from
///                         the \p TUSummaryData argument.
///
/// Returns null without emitting a failure when no summary was recorded for
/// the entity — that is a valid outcome in tests that assert absence.
/// Emits ADD_FAILURE and returns null when the decl or its EntityId cannot
/// be resolved, as those indicate test-infrastructure errors.
template <typename SummaryT, typename ContributorT = FunctionDecl,
          typename TUSummaryDataT>
const SummaryT *getEntitySummary(llvm::StringRef ContributorName,
                                 ASTContext &Ctx, TUSummaryExtractor &Extractor,
                                 const TUSummaryDataT &TUSummaryData) {
  const ContributorT *D = findDeclByName<ContributorT>(ContributorName, Ctx);

  if (!D) {
    ADD_FAILURE() << "failed to find decl '" << ContributorName << "'";
    return nullptr;
  }

  std::optional<EntityId> Id = Extractor.addEntity(D);

  if (!Id) {
    ADD_FAILURE() << "failed to get EntityId for '" << ContributorName << "'";
    return nullptr;
  }
  auto SumIt = TUSummaryData.find(SummaryT::summaryName());

  if (SumIt == TUSummaryData.end())
    return nullptr;

  auto EntIt = SumIt->second.find(*Id);

  if (EntIt == SumIt->second.end())
    return nullptr;
  return static_cast<const SummaryT *>(EntIt->second.get());
}

class OperatorNewDeletePointersExtractorTest : public ssaf::TestFixture {
protected:
  SSAFOptions Opts;
  TUSummary TUSum;
  TUSummaryBuilder Builder;
  std::unique_ptr<TUSummaryExtractor> Extractor;
  std::unique_ptr<ASTUnit> AST;

  OperatorNewDeletePointersExtractorTest()
      : TUSum(llvm::Triple("arm64-apple-macosx"),
              BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp")),
        Builder(TUSum, Opts), Extractor(nullptr) {}

  bool setUpTest(llvm::StringRef Code) {
    AST = tooling::buildASTFromCodeWithArgs(Code, {"-std=c++20"});
    if (!AST) {
      ADD_FAILURE() << "failed to build AST";
      return false;
    }
    for (auto &E : TUSummaryExtractorRegistry::entries()) {
      if (E.getName() == OperatorNewDeletePointersEntitySummary::Name) {
        Extractor = E.instantiate(Builder);
        break;
      }
    }
    if (!Extractor) {
      ADD_FAILURE() << "failed to find OperatorNewDeletePointersExtractor";
      return false;
    }
    Extractor->HandleTranslationUnit(AST->getASTContext());
    return true;
  }

  const OperatorNewDeletePointersEntitySummary *
  getEntitySummary(llvm::StringRef FnName) {
    return ::getEntitySummary<OperatorNewDeletePointersEntitySummary>(
        FnName, AST->getASTContext(), *Extractor, getData(TUSum));
  }

  std::optional<EntityId> getEntityId(llvm::StringRef Name) {
    if (const auto *D = findDeclByName(Name, AST->getASTContext()))
      return Extractor->addEntity(D);
    return std::nullopt;
  }

  std::optional<EntityId> getEntityIdForReturn(llvm::StringRef FnName) {
    if (const FunctionDecl *FD = findFnByName(FnName, AST->getASTContext()))
      return Extractor->addEntityForReturn(FD);
    return std::nullopt;
  }
};

//===----------------------------------------------------------------------===//
// Registration sanity
//===----------------------------------------------------------------------===//

TEST(OperatorNewDeletePointersExtractorRegistration, ExtractorRegistered) {
  EXPECT_TRUE(isTUSummaryExtractorRegistered(
      OperatorNewDeletePointersEntitySummary::Name));
}

//===----------------------------------------------------------------------===//
// Extractor cases
//===----------------------------------------------------------------------===//

TEST_F(OperatorNewDeletePointersExtractorTest, FreeOperatorDelete) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void operator delete(void *ptr) noexcept;
    void operator delete(void *ptr) noexcept { (void)ptr; }
  )cpp"));

  const auto *S = getEntitySummary("operator delete");

  ASSERT_TRUE(S);

  auto PtrId = getEntityId("ptr");

  ASSERT_TRUE(PtrId);

  EXPECT_EQ(*S, std::set{*PtrId});
}

TEST_F(OperatorNewDeletePointersExtractorTest, MemberOperatorDelete) {
  ASSERT_TRUE(setUpTest(R"cpp(
    class T {
    public:
      void operator delete(void *p) noexcept { (void)p; }
    };
  )cpp"));

  const auto *S = getEntitySummary("operator delete");

  ASSERT_TRUE(S);

  auto PId = getEntityId("p");
  ASSERT_TRUE(PId);

  EXPECT_EQ(*S, std::set{*PId});
}

TEST_F(OperatorNewDeletePointersExtractorTest, OperatorDeleteArray) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void operator delete[](void *p) noexcept;
    void operator delete[](void *p) noexcept { (void)p; }
  )cpp"));

  const auto *S = getEntitySummary("operator delete[]");

  ASSERT_TRUE(S);

  auto PId = getEntityId("p");

  ASSERT_TRUE(PId);
  EXPECT_EQ(*S, std::set{*PId});
}

TEST_F(OperatorNewDeletePointersExtractorTest, OperatorNew) {
  ASSERT_TRUE(setUpTest(R"cpp(
    typedef unsigned long size_t;
    void *operator new(size_t size);
    void *operator new(size_t size) { (void)size; return nullptr; }
  )cpp"));

  const auto *S = getEntitySummary("operator new");

  ASSERT_TRUE(S);

  auto RetId = getEntityIdForReturn("operator new");

  ASSERT_TRUE(RetId);
  EXPECT_EQ(*S, std::set{*RetId});
}

TEST_F(OperatorNewDeletePointersExtractorTest, PlacementNew) {
  ASSERT_TRUE(setUpTest(R"cpp(
    typedef unsigned long size_t;
    void *operator new(size_t size, void *placement) noexcept;
    void *operator new(size_t size, void *placement) noexcept {
      (void)size; return placement;
    }
  )cpp"));

  const auto *S = getEntitySummary("operator new");

  ASSERT_TRUE(S);

  auto PlacementId = getEntityId("placement");
  auto RetId = getEntityIdForReturn("operator new");

  ASSERT_TRUE(PlacementId);
  ASSERT_TRUE(RetId);
  EXPECT_EQ(*S, (std::set{*PlacementId, *RetId}));
}

TEST_F(OperatorNewDeletePointersExtractorTest, PlacementDelete) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void operator delete(void *ptr, void *placement) noexcept;
    void operator delete(void *ptr, void *placement) noexcept {
      (void)ptr; (void)placement;
    }
  )cpp"));

  const auto *S = getEntitySummary("operator delete");

  ASSERT_TRUE(S);

  auto PtrId = getEntityId("ptr");
  auto PlacementId = getEntityId("placement");

  ASSERT_TRUE(PtrId);
  ASSERT_TRUE(PlacementId);
  EXPECT_EQ(*S, (std::set{*PtrId, *PlacementId}));
}

TEST_F(OperatorNewDeletePointersExtractorTest, NoOperatorNewOrDeleteSummary) {
  ASSERT_TRUE(setUpTest(R"cpp(
    class T { int x; };
  )cpp"));

  auto &TUData = getData(TUSum);
  auto TUSummariesIter =
      TUData.find(OperatorNewDeletePointersEntitySummary::summaryName());

  ASSERT_EQ(TUSummariesIter, TUData.end());
}

} // namespace
