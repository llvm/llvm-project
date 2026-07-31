//===- VirtualMethodFamilyFormatTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatTest.h"
#include "ParsedAST.h"
#include "clang/AST/Decl.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace ssaf;

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
void PrintTo(const VirtualMethodSummary &S, std::ostream *OS) {
  std::string Str;
  llvm::raw_string_ostream(Str) << S;
  *OS << Str;
}
} // namespace clang::ssaf

namespace {

class VirtualMethodFamilyFormatTest : public JSONFormatTest {
protected:
  static constexpr EntityLinkage ExternalLinkage{EntityLinkageType::External};

  ParsedAST AST;

  std::unique_ptr<LUSummary> makeLUSummary() {
    return std::make_unique<LUSummary>(llvm::Triple("arm64-apple-macosx"),
                                       linkUnitNamespace());
  }

  /// Create an LU-scope EntityId for \p ND and mark it externally visible.
  EntityId addEntity(LUSummary &LU, const NamedDecl *ND) {
    return declare(LU, addEntity(getIdTable(LU), ND));
  }

  /// Create an LU-scope EntityId for the return slot of \p FD.
  EntityId addReturnEntity(LUSummary &LU, const FunctionDecl *FD) {
    return declare(LU, addReturnEntity(getIdTable(LU), FD));
  }

  /// Create an EntityId for \p ND directly in \p Ids. Used for the WPASuite
  /// tests, which carry a bare EntityIdTable rather than an LUSummary.
  EntityId addEntity(EntityIdTable &Ids, const NamedDecl *ND) {
    return getOrCreateId(Ids, ND ? getEntityName(ND) : std::nullopt, ND);
  }

  /// Create an EntityId for the return slot of \p FD directly in \p Ids.
  EntityId addReturnEntity(EntityIdTable &Ids, const FunctionDecl *FD) {
    return getOrCreateId(Ids, FD ? getEntityNameForReturn(FD) : std::nullopt,
                         FD);
  }

  /// Insert a single entity summary (by SummaryName, EntityId).
  template <typename SummaryT>
  void insertSummary(LUSummary &LU, EntityId Id, SummaryT Sum) {
    getData(LU)[SummaryT::summaryName()][Id] =
        std::make_unique<SummaryT>(std::move(Sum));
  }

  /// Looks up the \c SummaryT stored for \p Id, or nullptr (with a failure
  /// recorded) if there is none.
  template <typename SummaryT>
  const SummaryT *getSummary(const LUSummary &LU, EntityId Id) {
    const auto &Data = getData(LU);
    auto SumIt = Data.find(SummaryT::summaryName());
    if (SumIt == Data.end()) {
      ADD_FAILURE() << "no summaries of kind " << SummaryT::summaryName();
      return nullptr;
    }
    auto EIt = SumIt->second.find(Id);
    if (EIt == SumIt->second.end()) {
      ADD_FAILURE() << "no " << SummaryT::summaryName() << " for " << Id;
      return nullptr;
    }
    return static_cast<const SummaryT *>(EIt->second.get());
  }

  /// Number of \c SummaryT entries in \p LU.
  template <typename SummaryT> size_t summaryCount(const LUSummary &LU) {
    const auto &Data = getData(LU);
    auto It = Data.find(SummaryT::summaryName());
    return It == Data.end() ? 0 : It->second.size();
  }

  /// Round-trips an LUSummary through JSON write -> read and returns the
  /// resulting LUSummary on success.
  llvm::Expected<LUSummary> roundTripLU(const LUSummary &LU) {
    PathString InPath = makePath("vmf-lu.json");
    if (auto Err = JSONFormat().writeLUSummary(LU, InPath))
      return std::move(Err);
    return JSONFormat().readLUSummary(InPath);
  }

  /// Build a WPASuite with one entry of every given AnalysisResult type. The
  /// test fixture is a friend of WPASuite so it can construct one and reach
  /// into private members.
  WPASuite makeSuite(
      EntityIdTable IdTable,
      std::vector<std::pair<AnalysisName, std::unique_ptr<AnalysisResult>>>
          Entries) {
    WPASuite Suite = makeWPASuite();
    getIdTable(Suite) = std::move(IdTable);
    for (auto &[Name, R] : Entries)
      getData(Suite).emplace(Name, std::move(R));
    return Suite;
  }

  /// Round-trips a WPASuite through JSON write -> read.
  llvm::Expected<WPASuite> roundTripSuite(const WPASuite &Suite) {
    PathString Path = makePath("vmf-suite.json");
    if (auto Err = JSONFormat().writeWPASuite(Suite, Path))
      return std::move(Err);
    return JSONFormat().readWPASuite(Path);
  }

private:
  static NestedBuildNamespace linkUnitNamespace() {
    constexpr auto LinkUnitKind = BuildNamespaceKind::LinkUnit;
    return NestedBuildNamespace{BuildNamespace(LinkUnitKind, "TestLU")};
  }

  /// Entity names coming out of ASTEntityMapping have no namespace; qualify
  /// them with the link unit the way EntityLinker does, so the ids in these
  /// tests look like the ones a real LUSummary carries.
  EntityId getOrCreateId(EntityIdTable &Ids, std::optional<EntityName> Name,
                         const NamedDecl *ND) {
    if (Name)
      return Ids.getId(Name->makeQualified(linkUnitNamespace()));
    ADD_FAILURE() << "no entity name for "
                  << (ND ? ND->getQualifiedNameAsString() : "<null decl>");
    // EntityId has no default constructor; the failure above already fails
    // the test, so any id will do.
    return Ids.getId(EntityName("<missing>", "", linkUnitNamespace()));
  }

  static EntityId declare(LUSummary &LU, EntityId Id) {
    getLinkageTable(LU).insert({Id, ExternalLinkage});
    return Id;
  }
};

//===----------------------------------------------------------------------===//
// VirtualMethodSummary round-trips
//===----------------------------------------------------------------------===//

static VirtualMethodSummary
createSummary(llvm::ArrayRef<EntityId> Params, std::optional<EntityId> Return,
              llvm::ArrayRef<EntityId> Overridden = {}) {
  VirtualMethodSummary S;
  S.ParamEntities.assign(Params.begin(), Params.end());
  S.ReturnEntity = Return;
  S.OverriddenMethods.assign(Overridden.begin(), Overridden.end());
  return S;
}

TEST_F(VirtualMethodFamilyFormatTest, VirtualMethodSummaryEmpty) {
  ASSERT_TRUE(AST.parse(R"cpp(
    struct Foo {
      virtual void m();
    };
  )cpp"));

  auto LU = makeLUSummary();
  EntityId M = addEntity(*LU, AST.fn("Foo::m"));

  // ParamEntities and OverriddenMethods intentionally empty.
  const VirtualMethodSummary Expected =
      createSummary({}, addReturnEntity(*LU, AST.fn("Foo::m")));
  insertSummary(*LU, M, Expected);

  auto Round = roundTripLU(*LU);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  EXPECT_EQ(summaryCount<VirtualMethodSummary>(*Round), 1u);
  const auto *Out = getSummary<VirtualMethodSummary>(*Round, M);
  ASSERT_TRUE(Out);
  EXPECT_EQ(*Out, Expected);
}

TEST_F(VirtualMethodFamilyFormatTest, VirtualMethodSummarySingleParam) {
  ASSERT_TRUE(AST.parse(R"cpp(
    struct BarBase {
      virtual int *foo(int *p);
    };
    struct Bar : BarBase {
      int *foo(int *p) override;
    };
  )cpp"));

  auto LU = makeLUSummary();
  EntityId M = addEntity(*LU, AST.fn("Bar::foo"));
  EntityId Overridden = addEntity(*LU, AST.fn("BarBase::foo"));
  EntityId P = addEntity(*LU, AST.findParam("Bar::foo", 0));
  EntityId R = addReturnEntity(*LU, AST.fn("Bar::foo"));

  const VirtualMethodSummary Expected =
      createSummary({P}, R, /*Overridden=*/{Overridden});
  insertSummary(*LU, M, Expected);

  auto Round = roundTripLU(*LU);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  EXPECT_EQ(summaryCount<VirtualMethodSummary>(*Round), 1u);
  const auto *Out = getSummary<VirtualMethodSummary>(*Round, M);
  ASSERT_TRUE(Out);
  EXPECT_EQ(*Out, Expected);
}

TEST_F(VirtualMethodFamilyFormatTest, VirtualMethodSummaryMultiParam) {
  ASSERT_TRUE(AST.parse(R"cpp(
    struct Base {
      virtual char *foo(int *p1, char *p2);
    };
    struct Derived : Base {
      char *foo(int *p1, char *p2) override;
    };
  )cpp"));

  auto LU = makeLUSummary();
  EntityId M1 = addEntity(*LU, AST.fn("Base::foo"));
  EntityId M2 = addEntity(*LU, AST.fn("Derived::foo"));
  EntityId P1a = addEntity(*LU, AST.findParam("Base::foo", 0));
  EntityId P1b = addEntity(*LU, AST.findParam("Base::foo", 1));
  EntityId P2a = addEntity(*LU, AST.findParam("Derived::foo", 0));
  EntityId P2b = addEntity(*LU, AST.findParam("Derived::foo", 1));
  EntityId R1 = addReturnEntity(*LU, AST.fn("Base::foo"));
  EntityId R2 = addReturnEntity(*LU, AST.fn("Derived::foo"));

  const VirtualMethodSummary E1 = createSummary({P1a, P1b}, R1);
  // Derived::foo overrides Base::foo.
  const VirtualMethodSummary E2 =
      createSummary({P2a, P2b}, R2, /*Overridden=*/{M1});
  insertSummary(*LU, M1, E1);
  insertSummary(*LU, M2, E2);

  auto Round = roundTripLU(*LU);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  EXPECT_EQ(summaryCount<VirtualMethodSummary>(*Round), 2u);
  const auto *Out1 = getSummary<VirtualMethodSummary>(*Round, M1);
  ASSERT_TRUE(Out1);
  EXPECT_EQ(*Out1, E1);
  const auto *Out2 = getSummary<VirtualMethodSummary>(*Round, M2);
  ASSERT_TRUE(Out2);
  EXPECT_EQ(*Out2, E2);
}

//===----------------------------------------------------------------------===//
// VirtualMethodFamilyAnalysisResult round-trips (via WPASuite).
//===----------------------------------------------------------------------===//

class VirtualMethodFamilyAnalysisRoundTrip
    : public VirtualMethodFamilyFormatTest {
protected:
  EntityIdTable IdTable;
};

TEST_F(VirtualMethodFamilyAnalysisRoundTrip, EmptyResultRoundTrips) {
  auto R = std::make_unique<VirtualMethodFamilyAnalysisResult>();
  std::vector<std::pair<AnalysisName, std::unique_ptr<AnalysisResult>>> Entries;
  Entries.emplace_back(VirtualMethodFamilyAnalysisResult::analysisName(),
                       std::move(R));
  WPASuite Suite = makeSuite(std::move(IdTable), std::move(Entries));

  auto Round = roundTripSuite(Suite);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  auto Got = Round->get<VirtualMethodFamilyAnalysisResult>();
  ASSERT_THAT_EXPECTED(Got, llvm::Succeeded());
  EXPECT_TRUE(Got->RetAndParamData.empty());
}

static VirtualMethodFamilyAnalysisResult
createResult(llvm::ArrayRef<std::pair<EntityId, std::pair<EntityId, EntityId>>>
                 Entries) {
  VirtualMethodFamilyAnalysisResult Res;
  Res.RetAndParamData.reserve(Entries.size());
  for (const auto &[Id, Data] : Entries) {
    auto [FamilyId, OwnerMethodId] = Data;
    Res.RetAndParamData.insert({Id, {FamilyId, OwnerMethodId}});
  }
  return Res;
}

TEST_F(VirtualMethodFamilyAnalysisRoundTrip, SingleFamilyRoundTrips) {
  ASSERT_TRUE(AST.parse(R"cpp(
    struct Base {
      virtual void foo(int *p);
    };
    struct Derived : Base {
      void foo(int *p) override;
    };
  )cpp"));

  EntityId M1 = addEntity(IdTable, AST.fn("Base::foo"));
  EntityId M2 = addEntity(IdTable, AST.fn("Derived::foo"));
  EntityId P1 = addEntity(IdTable, AST.findParam("Base::foo", 0));
  EntityId P2 = addEntity(IdTable, AST.findParam("Derived::foo", 0));

  const VirtualMethodFamilyAnalysisResult Expected = createResult({
      {P1, {/*FamilyId=*/P1, /*OwnerMethodId=*/M1}},
      {P2, {/*FamilyId=*/P1, /*OwnerMethodId=*/M2}},
  });

  std::vector<std::pair<AnalysisName, std::unique_ptr<AnalysisResult>>> Entries;
  Entries.emplace_back(
      VirtualMethodFamilyAnalysisResult::analysisName(),
      std::make_unique<VirtualMethodFamilyAnalysisResult>(Expected));
  WPASuite Suite = makeSuite(std::move(IdTable), std::move(Entries));

  auto Round = roundTripSuite(Suite);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  auto Got = Round->get<VirtualMethodFamilyAnalysisResult>();
  ASSERT_THAT_EXPECTED(Got, llvm::Succeeded());
  EXPECT_EQ(*Got, Expected);
}

TEST_F(VirtualMethodFamilyAnalysisRoundTrip,
       MultiFamilyAndReturnSlotRoundTrips) {
  ASSERT_TRUE(AST.parse(R"cpp(
    struct A {
      virtual void m1(int *p);
      virtual int *m2();
    };
    struct B : A {
      void m1(int *p) override;
      int *m2() override;
    };
  )cpp"));

  EntityId M1 = addEntity(IdTable, AST.fn("A::m1"));
  EntityId M2 = addEntity(IdTable, AST.fn("B::m1"));
  EntityId M3 = addEntity(IdTable, AST.fn("A::m2"));
  EntityId M4 = addEntity(IdTable, AST.fn("B::m2"));
  EntityId P1 = addEntity(IdTable, AST.findParam("A::m1", 0));
  EntityId P2 = addEntity(IdTable, AST.findParam("B::m1", 0));
  EntityId R1 = addReturnEntity(IdTable, AST.fn("A::m2"));
  EntityId R2 = addReturnEntity(IdTable, AST.fn("B::m2"));

  const VirtualMethodFamilyAnalysisResult Expected = createResult({
      // Family #1: parameter slot
      {P1, {/*FamilyId=*/P1, /*OwnerMethodId=*/M1}},
      {P2, {/*FamilyId=*/P1, /*OwnerMethodId=*/M2}},
      // Family #2: return slot
      {R1, {/*FamilyId=*/R1, /*OwnerMethodId=*/M3}},
      {R2, {/*FamilyId=*/R1, /*OwnerMethodId=*/M4}},
  });

  std::vector<std::pair<AnalysisName, std::unique_ptr<AnalysisResult>>> Entries;
  Entries.emplace_back(
      VirtualMethodFamilyAnalysisResult::analysisName(),
      std::make_unique<VirtualMethodFamilyAnalysisResult>(Expected));
  WPASuite Suite = makeSuite(std::move(IdTable), std::move(Entries));

  auto Round = roundTripSuite(Suite);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  auto Got = Round->get<VirtualMethodFamilyAnalysisResult>();
  ASSERT_THAT_EXPECTED(Got, llvm::Succeeded());
  EXPECT_EQ(*Got, Expected);
}

} // namespace
