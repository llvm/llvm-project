//===- VirtualMethodFamilyAnalysisTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VirtualMethodFamilyTestSupport.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

using namespace clang;
using namespace ssaf;

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
void PrintTo(const VirtualMethodFamilyAnalysisResult &R, std::ostream *OS) {
  std::string Str;
  llvm::raw_string_ostream(Str) << R;
  *OS << Str;
}
} // namespace clang::ssaf

namespace {

class VirtualMethodFamilyAnalysisTest : public VirtualMethodFamilyTestBase {
protected:
  // Parses \p Code, runs the VirtualMethod extractor over it, and drives the
  // family analysis on the extracted summaries. Must be called exactly once
  // per test, before any of the lookup helpers below.
  void analyze(llvm::StringRef Code) {
    ASSERT_TRUE(runVirtualMethodExtractor(Code))
        << "failed to build the AST or instantiate the extractor";

    // Hand the extracted summaries to the analysis. The EntityIdTable is
    // copied rather than moved: the LUSummary is consumed by the driver, but
    // the ids in the result still have to be resolvable by name afterwards.
    constexpr auto LinkUnitKind = BuildNamespaceKind::LinkUnit;
    NestedBuildNamespace NS{BuildNamespace(LinkUnitKind, "TestLU")};
    llvm::Triple Target{"arm64-apple-macosx"};
    auto LU = std::make_unique<LUSummary>(Target, std::move(NS));
    getIdTable(*LU) = getIdTable(tuSummary());
    getLinkageTable(*LU) = getLinkageTable(tuSummary());
    getData(*LU) = std::move(getData(tuSummary()));

    AnalysisDriver Driver(std::move(LU));
    auto WPAOrErr = Driver.run<VirtualMethodFamilyAnalysisResult>();
    ASSERT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());
    WPA = std::move(*WPAOrErr);
    auto ROrErr = WPA.get<VirtualMethodFamilyAnalysisResult>();
    ASSERT_THAT_EXPECTED(ROrErr, llvm::Succeeded());
    R = &*ROrErr;
  }

  EntityId method(llvm::StringRef NameOrSignature) {
    return require(entityIdOf(AST.fn(NameOrSignature)), NameOrSignature);
  }

  EntityId param(llvm::StringRef NameOrSignature, unsigned Index = 0) {
    return require(entityIdOf(AST.findParam(NameOrSignature, Index)),
                   NameOrSignature);
  }

  EntityId ret(llvm::StringRef NameOrSignature) {
    return require(returnEntityIdOf(AST.fn(NameOrSignature)), NameOrSignature);
  }

  const VirtualMethodFamilyAnalysisResult &result() const { return *R; }

private:
  VirtualMethodFamilyAnalysisResult::Data at(EntityId ParamId) const {
    auto It = R->RetAndParamData.find(ParamId);
    if (It != R->RetAndParamData.end())
      return It->second;
    ADD_FAILURE() << "no data recorded for " << ParamId;
    // Let's just return some fallback value. The test fails anyway.
    return {ParamId, ParamId};
  }

  // EntityId has no default constructor, so report the miss and fall back to
  // an arbitrary id; the ADD_FAILURE() above makes the test fail regardless.
  EntityId require(std::optional<EntityId> Id, llvm::StringRef QualifiedName) {
    if (Id)
      return *Id;
    ADD_FAILURE() << "no entity extracted for '" << QualifiedName << "'";
    const auto &Entities = getEntities(getIdTable(tuSummary()));
    if (!Entities.empty())
      return Entities.begin()->second;
    return getIdTable(tuSummary()).getId(EntityName("<missing>", "", {}));
  }

  WPASuite WPA = makeWPASuite();
  const VirtualMethodFamilyAnalysisResult *R = nullptr;
};

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

TEST_F(VirtualMethodFamilyAnalysisTest, ChainOneFamily) {
  // Base <- Mid <- Der
  analyze(R"cpp(
    struct Base {
      virtual void foo(int *p);
    };
    struct Mid : Base {
      void foo(int *p) override;
    };
    struct Der : Mid {
      void foo(int *p) override;
    };
  )cpp");

  EntityId BaseFoo = method("Base::foo");
  EntityId MidFoo = method("Mid::foo");
  EntityId DerFoo = method("Der::foo");

  EntityId BaseFooP = param("Base::foo");
  EntityId MidFooP = param("Mid::foo");
  EntityId DerFooP = param("Der::foo");

  EntityId BaseFooR = ret("Base::foo");
  EntityId MidFooR = ret("Mid::foo");
  EntityId DerFooR = ret("Der::foo");

  EXPECT_EQ(result(),
            createResult({
                // Params
                {BaseFooP, {/*FamilyId=*/BaseFooP, /*OwnerMethodId=*/BaseFoo}},
                {MidFooP, {/*FamilyId=*/BaseFooP, /*OwnerMethodId=*/MidFoo}},
                {DerFooP, {/*FamilyId=*/BaseFooP, /*OwnerMethodId=*/DerFoo}},
                // Returns
                {BaseFooR, {/*FamilyId=*/BaseFooR, /*OwnerMethodId=*/BaseFoo}},
                {MidFooR, {/*FamilyId=*/BaseFooR, /*OwnerMethodId=*/MidFoo}},
                {DerFooR, {/*FamilyId=*/BaseFooR, /*OwnerMethodId=*/DerFoo}},
            }))
      << legend();
}

// Unrelated multiple inheritance: D::f overrides both {A::f, B::f}.
// The joining overrider bridges the two roots into a single family.

TEST_F(VirtualMethodFamilyAnalysisTest, UnrelatedMultipleInheritanceMerges) {
  // Base1 <--
  //          |-- Der
  // Base2 <--
  analyze(R"cpp(
    struct Base1 {
      virtual void foo(int *p);
    };
    struct Base2 {
      virtual void foo(int *p);
    };
    struct Der : Base1, Base2 {
      void foo(int *p) override;
    };
  )cpp");

  EntityId Base1Foo = method("Base1::foo");
  EntityId Base2Foo = method("Base2::foo");
  EntityId DerFoo = method("Der::foo");

  EntityId Base1FooP = param("Base1::foo");
  EntityId Base2FooP = param("Base2::foo");
  EntityId DerFooP = param("Der::foo");

  EntityId Base1FooR = ret("Base1::foo");
  EntityId Base2FooR = ret("Base2::foo");
  EntityId DerFooR = ret("Der::foo");

  EXPECT_EQ(
      result(),
      createResult({
          // Params
          {Base1FooP, {/*FamilyId=*/Base1FooP, /*OwnerMethodId=*/Base1Foo}},
          {Base2FooP, {/*FamilyId=*/Base1FooP, /*OwnerMethodId=*/Base2Foo}},
          {DerFooP, {/*FamilyId=*/Base1FooP, /*OwnerMethodId=*/DerFoo}},
          // Returns
          {Base1FooR, {/*FamilyId=*/Base1FooR, /*OwnerMethodId=*/Base1Foo}},
          {Base2FooR, {/*FamilyId=*/Base1FooR, /*OwnerMethodId=*/Base2Foo}},
          {DerFooR, {/*FamilyId=*/Base1FooR, /*OwnerMethodId=*/DerFoo}},
      }))
      << legend();
}

// Overloads have different vtable slots, thus they need to be treated separate.
TEST_F(VirtualMethodFamilyAnalysisTest, OverloadsNotMerged) {
  // Base <- Der
  analyze(R"cpp(
    struct Base {
      virtual void foo(int *p);  // <-- later gets overridden
      virtual void foo(char *p); // <-- unrelated overload
    };
    struct Der : Base {
      void foo(int *p) override;
    };
  )cpp");

  EntityId BaseFooInt = method("Base::foo(int *)");
  EntityId DerFoo = method("Der::foo");

  EntityId BaseFooIntP = param("Base::foo(int *)");
  EntityId DerFooP = param("Der::foo");

  EntityId BaseFooIntR = ret("Base::foo(int *)");
  EntityId DerFooR = ret("Der::foo");

  const auto Expected = createResult({
      // Params
      {BaseFooIntP, {/*FamilyId=*/BaseFooIntP, /*OwnerMethodId=*/BaseFooInt}},
      {DerFooP, {/*FamilyId=*/BaseFooIntP, /*OwnerMethodId=*/DerFoo}},
      // Returns
      {BaseFooIntR, {/*FamilyId=*/BaseFooIntR, /*OwnerMethodId=*/BaseFooInt}},
      {DerFooR, {/*FamilyId=*/BaseFooIntR, /*OwnerMethodId=*/DerFoo}},
  });
  // "Base::foo(char *)" is not mentioned because that is not overridden by
  // anyone.
  EXPECT_EQ(result(), Expected) << legend();
}

// Covariant returns have the same family.
TEST_F(VirtualMethodFamilyAnalysisTest, CovariantReturnUnifiesReturnSlots) {
  // Base <- Der
  analyze(R"cpp(
    struct Base {
      virtual Base *clone();
    };
    struct Der : Base {
      Der *clone() override; // <-- has covariant return type
    };
  )cpp");

  EntityId BaseClone = method("Base::clone");
  EntityId DerClone = method("Der::clone");

  EntityId BaseCloneR = ret("Base::clone");
  EntityId DerCloneR = ret("Der::clone");

  EXPECT_EQ(
      result(),
      createResult({
          {BaseCloneR, {/*FamilyId=*/BaseCloneR, /*OwnerMethodId=*/BaseClone}},
          {DerCloneR, {/*FamilyId=*/BaseCloneR, /*OwnerMethodId=*/DerClone}},
      }))
      << legend();
}

TEST_F(VirtualMethodFamilyAnalysisTest, DiamondOneFamily) {
  //    Base      //
  //   /    \     //
  // Left  Right  //
  //   \    /     //
  //    Dia       //
  analyze(R"cpp(
    struct Base {
      virtual void foo(int *p);
    };
    struct Left : Base {
      void foo(int *p) override;
    };
    struct Right : Base {
      void foo(int *p) override;
    };
    struct Dia : Left, Right {
      void foo(int *p) override;
    };
  )cpp");

  EntityId BaseFoo = method("Base::foo");
  EntityId LeftFoo = method("Left::foo");
  EntityId RightFoo = method("Right::foo");
  EntityId DiaFoo = method("Dia::foo");

  EntityId BaseFooP = param("Base::foo");
  EntityId LeftFooP = param("Left::foo");
  EntityId RightFooP = param("Right::foo");
  EntityId DiaFooP = param("Dia::foo");

  EntityId BaseFooR = ret("Base::foo");
  EntityId LeftFooR = ret("Left::foo");
  EntityId RightFooR = ret("Right::foo");
  EntityId DiaFooR = ret("Dia::foo");

  EXPECT_EQ(
      result(),
      createResult({
          // Params
          {BaseFooP, {/*FamilyId=*/BaseFooP, /*OwnerMethodId=*/BaseFoo}},
          {LeftFooP, {/*FamilyId=*/BaseFooP, /*OwnerMethodId=*/LeftFoo}},
          {RightFooP, {/*FamilyId=*/BaseFooP, /*OwnerMethodId=*/RightFoo}},
          {DiaFooP, {/*FamilyId=*/BaseFooP, /*OwnerMethodId=*/DiaFoo}},
          // Returns
          {BaseFooR, {/*FamilyId=*/BaseFooR, /*OwnerMethodId=*/BaseFoo}},
          {LeftFooR, {/*FamilyId=*/BaseFooR, /*OwnerMethodId=*/LeftFoo}},
          {RightFooR, {/*FamilyId=*/BaseFooR, /*OwnerMethodId=*/RightFoo}},
          {DiaFooR, {/*FamilyId=*/BaseFooR, /*OwnerMethodId=*/DiaFoo}},
      }))
      << legend();
}

TEST_F(VirtualMethodFamilyAnalysisTest, NoFamilies) {
  analyze(R"cpp(
    struct A {
      virtual void f(int *p);
    };
  )cpp");
  // No methods are overridden => empty map.
  EXPECT_EQ(result(), createResult({})) << legend();
}

} // namespace
