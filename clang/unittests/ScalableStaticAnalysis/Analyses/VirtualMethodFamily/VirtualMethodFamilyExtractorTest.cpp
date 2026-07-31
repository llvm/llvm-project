//===- VirtualMethodFamilyExtractorTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VirtualMethodFamilyTestSupport.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "gtest/gtest.h"

#include <set>

using namespace clang;
using namespace ssaf;

namespace {

using VirtualMethodFamilyExtractorTest = VirtualMethodFamilyTestBase;

TEST_F(VirtualMethodFamilyExtractorTest, Registers) {
  EXPECT_TRUE(isTUSummaryExtractorRegistered(VirtualMethodSummary::Name));
}

using VirtualMethodFamilyExtractorBasicFieldPopulationTest =
    VirtualMethodFamilyExtractorTest;

TEST_F(VirtualMethodFamilyExtractorBasicFieldPopulationTest, BaseVirtual) {
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    class Base {
    public:
      virtual void foo(int *p);
    };
  )cpp"));

  const auto *S = getMethodSummary(AST.fn("Base::foo"));
  ASSERT_TRUE(S);
  EXPECT_TRUE(S->ReturnEntity.has_value());
  EXPECT_EQ(S->ParamEntities.size(), 1u);
  // A root virtual method overrides nothing.
  EXPECT_TRUE(S->OverriddenMethods.empty());
}

TEST_F(VirtualMethodFamilyExtractorBasicFieldPopulationTest, PureVirtual) {
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    class Interface {
    public:
      virtual void foo(int *p) = 0;
    };
  )cpp"));

  // A pure-virtual method is still virtual, so a summary is produced for it.
  ASSERT_TRUE(getMethodSummary(AST.fn("Interface::foo")));
}

TEST_F(VirtualMethodFamilyExtractorBasicFieldPopulationTest,
       NonVirtualMethodSkipped) {
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    class C {
    public:
      virtual void v();
      void nv();
    };
  )cpp"));

  // Only the virtual method has a summary; non-virtual is skipped.
  EXPECT_EQ(methodSummaryCount(), 1u);
  EXPECT_TRUE(getMethodSummary(AST.fn("C::v")));
}

TEST_F(VirtualMethodFamilyExtractorBasicFieldPopulationTest,
       OverrideWithoutVirtualKeywordExtracted) {
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    class Base {
    public:
      virtual void foo(int *p);
    };
    class Derived : public Base {
    public:
      void foo(int *p) override;
    };
  )cpp"));

  EXPECT_EQ(methodSummaryCount(), 2u);
  EXPECT_TRUE(getMethodSummary(AST.fn("Base::foo")));
  EXPECT_TRUE(getMethodSummary(AST.fn("Derived::foo")));
}

using VirtualMethodFamilyExtractorOverriddenMethodTest =
    VirtualMethodFamilyExtractorTest;

TEST_F(VirtualMethodFamilyExtractorOverriddenMethodTest,
       EdgeBaseDerivedOverride) {
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    struct Base {
      virtual void f(int *p);
    };
    struct Derived : Base {
      void f(int *p) override;
    };
  )cpp"));
  const auto *D = getMethodSummary(AST.fn("Derived::f"));
  const auto *B = getMethodSummary(AST.fn("Base::f"));
  ASSERT_TRUE(D);
  ASSERT_TRUE(B);
  auto BId = entityIdOf(AST.fn("Base::f"));
  ASSERT_TRUE(BId.has_value());
  ASSERT_EQ(D->OverriddenMethods.size(), 1u);
  EXPECT_EQ(D->OverriddenMethods[0], *BId);
  EXPECT_TRUE(B->OverriddenMethods.empty());
}

TEST_F(VirtualMethodFamilyExtractorOverriddenMethodTest,
       EdgePureVirtualReabstractsOverride) {
  // Tricky: a pure-virtual method that OVERRIDES a concrete virtual. Its edge
  // set must be non-empty despite being pure.
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    struct A {
      virtual void f(int *p);
    };
    struct B : A {
      void f(int *p) = 0;
    };
  )cpp"));
  const auto *Bf = getMethodSummary(AST.fn("B::f"));
  ASSERT_TRUE(Bf);
  auto Af = entityIdOf(AST.fn("A::f"));
  ASSERT_TRUE(Af.has_value());
  ASSERT_EQ(Bf->OverriddenMethods.size(), 1u);
  EXPECT_EQ(Bf->OverriddenMethods[0], *Af);
}

TEST_F(VirtualMethodFamilyExtractorOverriddenMethodTest,
       EdgeMultipleInheritanceTwoEdges) {
  // Tricky: one override occupies two independent base slots.
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    struct A {
      virtual void f(int *p);
    };
    struct B {
      virtual void f(int *p);
    };
    struct D : A, B {
      void f(int *p) override;
    };
  )cpp"));
  const auto *Df = getMethodSummary(AST.fn("D::f"));
  ASSERT_TRUE(Df);
  auto Af = entityIdOf(AST.fn("A::f"));
  auto Bf = entityIdOf(AST.fn("B::f"));
  ASSERT_TRUE(Af.has_value() && Bf.has_value());
  ASSERT_EQ(Df->OverriddenMethods.size(), 2u);
  std::set<EntityId> Edges(Df->OverriddenMethods.begin(),
                           Df->OverriddenMethods.end());
  EXPECT_EQ(Edges.count(*Af), 1u);
  EXPECT_EQ(Edges.count(*Bf), 1u);
}

TEST_F(VirtualMethodFamilyExtractorOverriddenMethodTest,
       EdgeOverrideLinksMatchingOverloadOnly) {
  // Tricky: overloads must not be conflated. B::f(int*) overrides only the
  // f(int*) base overload, never f(char*).
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    struct A {
      virtual void f(int *p);
      virtual void f(char *p);
    };
    struct B : A {
      void f(int *p) override;
    };
  )cpp"));
  const auto *Bf = getMethodSummary(AST.fn("B::f"));
  ASSERT_TRUE(Bf);
  auto AfInt = entityIdOf(AST.fn("A::f(int *)"));
  auto AfChar = entityIdOf(AST.fn("A::f(char *)"));
  ASSERT_TRUE(AfInt.has_value() && AfChar.has_value());
  ASSERT_EQ(Bf->OverriddenMethods.size(), 1u);
  EXPECT_EQ(Bf->OverriddenMethods[0], *AfInt);
  EXPECT_NE(Bf->OverriddenMethods[0], *AfChar);
}

TEST_F(VirtualMethodFamilyExtractorOverriddenMethodTest,
       EdgeCovariantReturnOverride) {
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    struct Base {
      virtual Base *clone();
    };
    struct Deriv : Base {
      Deriv *clone() override;
    };
  )cpp"));
  const auto *Dc = getMethodSummary(AST.fn("Deriv::clone"));
  ASSERT_TRUE(Dc);
  auto Bc = entityIdOf(AST.fn("Base::clone"));
  ASSERT_TRUE(Bc.has_value());
  ASSERT_EQ(Dc->OverriddenMethods.size(), 1u);
  EXPECT_EQ(Dc->OverriddenMethods[0], *Bc);
  EXPECT_TRUE(Dc->ReturnEntity.has_value());
}

TEST_F(VirtualMethodFamilyExtractorOverriddenMethodTest,
       EdgeDependentBaseTemplatePatternNoCrash) {
  // Tricky: the primary template pattern has a dependent base; overridden_
  // methods is unresolved there. Must not crash; the instantiation carries the
  // edge.
  ASSERT_TRUE(runVirtualMethodExtractor(R"cpp(
    template <class T>
    struct Wrapper {
      virtual void f(int *p);
    };
    template <class T> struct DTypeParam : T {
      virtual void f(int *p);
    };
    template <class T> struct DSpec : Wrapper<T> {
      void f(int *p) override;
    };
    struct Concrete {
      virtual void f(int *p);
    };
    template struct DSpec<Concrete>;
  )cpp"));
  EXPECT_GT(methodSummaryCount(), 0u);
}

} // namespace
