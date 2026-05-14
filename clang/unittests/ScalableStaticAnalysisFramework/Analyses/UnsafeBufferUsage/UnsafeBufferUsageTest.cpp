//===- UnsafeBufferUsageTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "FindDecl.h"
#include "TestFixture.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <initializer_list>
#include <memory>
#include <optional>

using namespace clang;
using namespace ssaf;
using testing::UnorderedElementsAre;

namespace {
class UnsafeBufferUsageTest : public TestFixture {
protected:
  TUSummary TUSum;
  TUSummaryBuilder Builder;
  std::unique_ptr<TUSummaryExtractor> Extractor;
  std::unique_ptr<ASTUnit> AST;

  UnsafeBufferUsageTest()
      : TUSum(BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp")),
        Builder(TUSum) {}

  bool setUpTest(StringRef Code) {
    AST = tooling::buildASTFromCodeWithArgs(
        Code, {"-Wno-unused-value", "-Wno-int-to-pointer-cast"});

    Extractor =
        makeTUSummaryExtractor(UnsafeBufferUsageEntitySummary::Name, Builder);

    if (!Extractor) {
      ADD_FAILURE() << "failed to find UnsafeBufferUsageTUSummaryExtractor";
      return false;
    }
    Extractor->HandleTranslationUnit(AST->getASTContext());
    return true;
  }

  template <typename ContributorDecl = NamedDecl>
  const UnsafeBufferUsageEntitySummary *
  getEntitySummary(StringRef ContributorEntityName) {
    auto *ContributorDefn = findDeclByName<ContributorDecl>(
        ContributorEntityName, AST->getASTContext());

    if (!ContributorDefn) {
      ADD_FAILURE() << "failed to find Decl of \"" << ContributorEntityName
                    << "\"";
      return nullptr;
    }

    std::optional<EntityId> ContributorEntityId =
        Extractor->addEntity(ContributorDefn);
    if (!ContributorEntityId) {
      ADD_FAILURE() << "failed to get EntityName for contributor \""
                    << ContributorEntityName << "\"";
      return nullptr;
    }

    auto &TUSumData = getData(TUSum);
    auto EntitiesSumIter =
        TUSumData.find(UnsafeBufferUsageEntitySummary::summaryName());

    // If none entity summary was collected, it may not be an entry in
    // `TUSumData`:
    if (EntitiesSumIter == TUSumData.end())
      return nullptr;

    auto EntitySumIter = EntitiesSumIter->second.find(*ContributorEntityId);

    // If entity summary is empty, it may not exist:
    if (EntitySumIter == EntitiesSumIter->second.end())
      return nullptr;
    return static_cast<const UnsafeBufferUsageEntitySummary *>(
        EntitySumIter->second.get());
  }

  std::optional<EntityId> getEntityId(StringRef Name) {
    if (const auto *D = findDeclByName(Name, AST->getASTContext()))
      return Extractor->addEntity(D);
    return std::nullopt;
  }

  std::optional<EntityId> getEntityIdForReturn(StringRef FunName) {
    if (const auto *D = findFnByName(FunName, AST->getASTContext()))
      return Extractor->addEntityForReturn(D);
    return std::nullopt;
  }

  // Same as `std::pair<StringName, unsigned>` for a pair of entity declaration
  // name and a pointer level with an extra optional flag for whether the entity
  // represents a function return value:
  struct EPLPair {
    EPLPair(StringRef Name, unsigned Lv, bool isFunRet = false)
        : Name(Name), Lv(Lv), isFunRet(isFunRet) {}

    StringRef Name;
    unsigned Lv;
    bool isFunRet;
  };

  EntityPointerLevelSet makeSet(unsigned Line, ArrayRef<EPLPair> Pairs) {
    auto EPLs = llvm::map_range(
        Pairs, [this, Line](const EPLPair &Pair) -> EntityPointerLevel {
          std::optional<EntityId> Entity = Pair.isFunRet
                                               ? getEntityIdForReturn(Pair.Name)
                                               : getEntityId(Pair.Name);
          if (!Entity)
            ADD_FAILURE_AT(__FILE__, Line) << "Entity not found: " << Pair.Name;
          return buildEntityPointerLevel(*Entity, Pair.Lv);
        });
    return EntityPointerLevelSet{EPLs.begin(), EPLs.end()};
  }
};

//////////////////////////////////////////////////////////////
//                   Data Structure Tests                   //
//////////////////////////////////////////////////////////////

static llvm::iterator_range<EntityPointerLevelSet::iterator>
getSubsetOf(const EntityPointerLevelSet &Set, EntityId Entity) {
  return llvm::make_range(Set.equal_range(Entity));
}

TEST_F(UnsafeBufferUsageTest, EntityPointerLevelComparison) {
  EntityIdTable Table;
  EntityId E1 = Table.getId({"c:@F@foo", "", {}});
  EntityId E2 = Table.getId({"c:@F@bar", "", {}});

  auto P1 = buildEntityPointerLevel(E1, 2);
  auto P2 = buildEntityPointerLevel(E1, 2);
  auto P3 = buildEntityPointerLevel(E1, 1);
  auto P4 = buildEntityPointerLevel(E2, 2);

  EXPECT_EQ(P1, P2);
  EXPECT_NE(P1, P3);
  EXPECT_NE(P1, P4);
  EXPECT_NE(P3, P4);
  EXPECT_TRUE(P3 < P2);
  EXPECT_TRUE(P3 < P4);
  EXPECT_FALSE(P1 < P2);
  EXPECT_FALSE(P2 < P1);
}

TEST_F(UnsafeBufferUsageTest, UnsafeBufferUsageEntityPointerLevelSetTest) {
  EntityIdTable Table;
  EntityId E1 = Table.getId({"c:@F@foo", "", {}});
  EntityId E2 = Table.getId({"c:@F@bar", "", {}});
  EntityId E3 = Table.getId({"c:@F@baz", "", {}});

  auto P1 = buildEntityPointerLevel(E1, 1);
  auto P2 = buildEntityPointerLevel(E1, 2);
  auto P3 = buildEntityPointerLevel(E2, 1);
  auto P4 = buildEntityPointerLevel(E2, 2);
  auto P5 = buildEntityPointerLevel(E3, 1);

  EntityPointerLevelSet Set{P1, P2, P3, P4, P5};

  EXPECT_THAT(Set, UnorderedElementsAre(P1, P2, P3, P4, P5));
  EXPECT_THAT(getSubsetOf(Set, E1), UnorderedElementsAre(P1, P2));
  EXPECT_THAT(getSubsetOf(Set, E2), UnorderedElementsAre(P3, P4));
  EXPECT_THAT(getSubsetOf(Set, E3), UnorderedElementsAre(P5));
}

//////////////////////////////////////////////////////////////
//                   Extractor Tests                        //
//////////////////////////////////////////////////////////////

TEST_F(UnsafeBufferUsageTest, SimpleFunctionWithUnsafePointer) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p) {
      p[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, PointerArithmetic) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      *(p + 5);
      *(q - 3);
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, PointerIncrementDecrement) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p, int *q, int *r, int *s) {
      (++p)[5];
      (q++)[5];
      (--r)[5];
      (s--)[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum,
            makeSet(__LINE__, {{"p", 1U}, {"q", 1U}, {"r", 1U}, {"s", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, PointerAssignment) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      (p = q + 5)[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, CompoundAssignment) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      (p += 5)[5];
      (q -= 3)[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, MultiLevelPointer) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int **p, int **q, int **r) {
      (*p)[5];
      *(*q);
      *(q[5]);
      r[5][5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum,
            makeSet(__LINE__, {{"p", 2U}, {"q", 1U}, {"r", 1U}, {"r", 2U}}));
}

TEST_F(UnsafeBufferUsageTest, ConditionalOperator) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int **p, int **q, int cond) {
      (cond ? *p : *q)[5];
      cond ? p[5] : q[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum,
            makeSet(__LINE__, {{"p", 1U}, {"q", 1U}, {"p", 2U}, {"q", 2U}}));
}

TEST_F(UnsafeBufferUsageTest, CastExpression) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(void *p, int q) {
      ((int*)p)[5];
      ((int*)q)[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, CommaOperator) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p, int x) {
      (x++, p)[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, CommaOperator2) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int **p, int **q, int x) {
      (p[x] = 0, q[x] = 0)[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}, {"q", 2U}}));
}

TEST_F(UnsafeBufferUsageTest, ParenthesizedExpression) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p) {
      (((p)))[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, ArrayParameter) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int arr[], int arr2[][10]) {
      int n = 5;
      arr[100];
      arr2[5][n];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"arr", 1U}, {"arr2", 1U}, {"arr2", 2U}}));
}

TEST_F(UnsafeBufferUsageTest, FunctionCall) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int ** (*fp)();
    int ** foo() {
      fp = &foo;
      foo()[5];
      (*fp())[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  // No (foo, 2) because indirect calls are ignored.
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"foo", 1U, true}}));
}

TEST_F(UnsafeBufferUsageTest, StructMemberAccess) {
  ASSERT_TRUE(setUpTest(R"cpp(
    struct S {
      int *ptr;
      int (*ptr_to_arr)[10];
    };
    void foo(struct S obj) {
      int n = 5;
      obj.ptr[5];
      (*obj.ptr_to_arr)[n];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"ptr", 1U}, {"ptr_to_arr", 2U}}));
}

TEST_F(UnsafeBufferUsageTest, StringLiteralSubscript) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo() {
      "hello"[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  // String literals should not generate pointer kind variables
  EXPECT_EQ(Sum, nullptr);
}

TEST_F(UnsafeBufferUsageTest, OpaqueValueExpr) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p, int *q) {
       (p ?: q)[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, AddressOfOperator) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int x) {
      (&x)[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");
  // Address-of should not generate pointer kind variables for 'x':
  EXPECT_EQ(Sum, nullptr);
}

TEST_F(UnsafeBufferUsageTest, AddressOfThenDereference) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      (*(&p))[5];
      (&(*q))[5];
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1}, {"q", 1}}));
}

TEST_F(UnsafeBufferUsageTest, PointerToArrayOfPointers) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo() {
      int * arr[10];
      int * (*p)[10] = &arr;

      (*p)[5][5]; // '(*p)[5]' is unsafe
                  // '(*p)' is fine because 5 < 10
    }
  )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 3}}));
}

TEST_F(UnsafeBufferUsageTest, UnsafePointerInGlobalVariableInitializer) {
  ASSERT_TRUE(setUpTest(R"cpp(
      int *gp;
      int x = gp[5];
    )cpp"));
  const auto *Sum = getEntitySummary("x");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"gp", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, UnsafePointerInFieldInitializer) {
  ASSERT_TRUE(setUpTest(R"cpp(
      int *gp;
      struct Foo {
        int field = gp[5];
      };
    )cpp"));
  const auto *Sum = getEntitySummary("Foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"gp", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, UnsafePointerInFieldInitializer2) {
  ASSERT_TRUE(setUpTest(R"cpp(
      int *gp;
      union Foo {
        int field = gp[5];
        int x;
      };
    )cpp"));
  const auto *Sum = getEntitySummary("Foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"gp", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, InitializerList) {
  ASSERT_TRUE(setUpTest(R"cpp(
      int *gp;
      struct Foo {
        int field;
        int x;
      };
      Foo FooObj{gp[5], 0};
    )cpp"));
  const auto *Sum = getEntitySummary("FooObj");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"gp", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, UnsafePointerInCXXCtorInitializer) {
  ASSERT_TRUE(setUpTest(R"cpp(
      struct Foo {
        int member;
        Foo(int *p) : member(p[5]) {}
      };
    )cpp"));
  const auto *Sum = getEntitySummary<CXXConstructorDecl>("Foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, UnsafePointerInDefaultArg) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int * gp;
    void foo(int x = gp[5]);
    )cpp"));
  const auto *Sum = getEntitySummary("foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"gp", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, NestedDefinitions) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int * a = [](){
      struct Foo {
        void bar(int * ptr) { ptr[3] = 0; }
      };
      return nullptr;
    }();
    )cpp"));
  const auto *Sum = getEntitySummary("bar");

  EXPECT_NE(Sum, nullptr);
  // The closest contributor owns the fact:
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"ptr", 1U}}));

  Sum = getEntitySummary("Foo");

  EXPECT_EQ(Sum, nullptr);

  Sum = getEntitySummary("a");

  EXPECT_EQ(Sum, nullptr);
}

TEST_F(UnsafeBufferUsageTest, NestedDefinitions2) {
  bool SetupSuccess = setUpTest(R"cpp(
    int main(void) {
       struct Foo {
          void bar(int * ptr) { ptr[3] = 0; }
       };
    }
    )cpp");

  ASSERT_TRUE(SetupSuccess);

  const auto *Sum = getEntitySummary("bar");

  EXPECT_NE(Sum, nullptr);
  // The closest contributor owns the fact:
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"ptr", 1U}}));

  Sum = getEntitySummary("Foo");

  EXPECT_EQ(Sum, nullptr);

  Sum = getEntitySummary("main");

  EXPECT_EQ(Sum, nullptr);
}

} // namespace
