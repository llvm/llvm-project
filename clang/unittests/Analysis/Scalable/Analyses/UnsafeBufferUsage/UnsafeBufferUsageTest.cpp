//===------- unittests/Analysis/Scalable/UnsafeBufferUsageTest.cpp --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Analysis/Scalable/ASTEntityMapping.h"
#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsageBuilder.h"
#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsageExtractor.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ssaf;

namespace {

template <typename SomeDecl = NamedDecl>
const SomeDecl *findDeclByName(StringRef Name, ASTContext &Ctx) {
  class NamedDeclFinder : public DynamicRecursiveASTVisitor {
  public:
    StringRef SearchingName;
    const NamedDecl *FoundDecl = nullptr;

    NamedDeclFinder(StringRef SearchingName) : SearchingName(SearchingName) {}

    bool VisitDecl(Decl *D) override {
      if (const auto *ND = dyn_cast<NamedDecl>(D)) {
        if (ND->getNameAsString() == SearchingName) {
          FoundDecl = ND;
          return false;
        }
      }
      return true;
    }
  };

  NamedDeclFinder Finder(Name);

  Finder.TraverseDecl(Ctx.getTranslationUnitDecl());
  return dyn_cast_or_null<SomeDecl>(Finder.FoundDecl);
}

const FunctionDecl *findFnByName(StringRef Name, ASTContext &Ctx) {
  return findDeclByName<FunctionDecl>(Name, Ctx);
}

class UnsafeBufferUsageTest : public testing::Test {
protected:
  TUSummary TUSummary;
  UnsafeBufferUsageTUSummaryBuilder Builder;
  UnsafeBufferUsageTUSummaryExtractor Extractor;
  std::unique_ptr<ASTUnit> AST;

  UnsafeBufferUsageTest()
      : TUSummary(
            BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp")),
        Builder(TUSummary),
        Extractor(UnsafeBufferUsageTUSummaryExtractor(Builder)) {}

  std::unique_ptr<UnsafeBufferUsageEntitySummary>
  setUpTest(StringRef Code, StringRef ContributorName) {
    AST = tooling::buildASTFromCodeWithArgs(Code, {"-Wno-everything"});

    const auto *ContributorDefn =
        findDeclByName(ContributorName, AST->getASTContext());
    std::optional<EntityName> EN = getEntityName(ContributorDefn);

    if (!ContributorDefn || !EN)
      return nullptr;
    return Extractor.extractEntitySummary(
        Builder.addEntity(*EN), ContributorDefn, AST->getASTContext());
  }

  std::optional<EntityId> getEntityId(StringRef Name) {
    if (const auto *D = findDeclByName(Name, AST->getASTContext()))
      if (auto EntityName = getEntityName(D))
        return Builder.addEntity(*EntityName);
    return std::nullopt;
  }

  std::optional<EntityId> getEntityIdForReturn(StringRef FunName) {
    if (const auto *D = findFnByName(FunName, AST->getASTContext()))
      if (auto EntityName = getEntityNameForReturn(D))
        return Builder.addEntity(*EntityName);
    return std::nullopt;
  }
};

constexpr inline auto buildEntityPointerLevel =
    UnsafeBufferUsageTUSummaryBuilder::buildEntityPointerLevel;
constexpr inline auto buildUnsafeBufferUsageEntitySummary =
    UnsafeBufferUsageTUSummaryBuilder::buildUnsafeBufferUsageEntitySummary;

//////////////////////////////////////////////////////////////
//                   Data Structure Tests                   //
//////////////////////////////////////////////////////////////

#define EXPECT_CONTAINS(Set, Elt) EXPECT_NE((Set).find(Elt), (Set).end())
#define EXPECT_EXCLUDES(Set, Elt) EXPECT_EQ((Set).find(Elt), (Set).end())

TEST_F(UnsafeBufferUsageTest, EntityPointerLevelComparison) {
  EntityId E1 = Builder.addEntity({"c:@F@foo", "", {}});
  EntityId E2 = Builder.addEntity({"c:@F@bar", "", {}});

  auto P1 = Builder.buildEntityPointerLevel(E1, 2);
  auto P2 = Builder.buildEntityPointerLevel(E1, 2);
  auto P3 = Builder.buildEntityPointerLevel(E1, 1);
  auto P4 = Builder.buildEntityPointerLevel(E2, 2);

  EXPECT_EQ(P1, P2);
  EXPECT_NE(P1, P3);
  EXPECT_NE(P1, P4);
  EXPECT_NE(P3, P4);
  EXPECT_TRUE(P3 < P2);
  EXPECT_TRUE(P3 < P4);
  EXPECT_FALSE(P1 < P2);
  EXPECT_FALSE(P2 < P1);
}

TEST_F(UnsafeBufferUsageTest, UnsafeBufferUsageEntitySummaryTest) {
  EntityId E1 = Builder.addEntity({"c:@F@foo", "", {}});
  EntityId E2 = Builder.addEntity({"c:@F@bar", "", {}});
  EntityId E3 = Builder.addEntity({"c:@F@baz", "", {}});

  auto P1 = Builder.buildEntityPointerLevel(E1, 1);
  auto P2 = Builder.buildEntityPointerLevel(E1, 2);
  auto P3 = Builder.buildEntityPointerLevel(E2, 1);
  auto P4 = Builder.buildEntityPointerLevel(E2, 2);
  auto P5 = Builder.buildEntityPointerLevel(E3, 1);
  auto P6 = Builder.buildEntityPointerLevel(E3, 2);

  EntityPointerLevelSet Set{P1, P2, P3, P4, P5};
  auto ES = Builder.buildUnsafeBufferUsageEntitySummary(std::move(Set));
  ASSERT_TRUE(ES);

  EXPECT_CONTAINS(*ES, P1);
  EXPECT_CONTAINS(*ES, P2);
  EXPECT_CONTAINS(*ES, P3);
  EXPECT_CONTAINS(*ES, P4);
  EXPECT_CONTAINS(*ES, P5);
  EXPECT_EXCLUDES(*ES, P6);

  EntityPointerLevelSet Subset1{ES->getSubsetOf(E1).begin(),
                                ES->getSubsetOf(E1).end()};

  EXPECT_CONTAINS(Subset1, P1);
  EXPECT_CONTAINS(Subset1, P2);
  EXPECT_EQ(Subset1.size(), 2U);

  EntityPointerLevelSet Subset2{ES->getSubsetOf(E2).begin(),
                                ES->getSubsetOf(E2).end()};

  EXPECT_CONTAINS(Subset2, P3);
  EXPECT_CONTAINS(Subset2, P4);
  EXPECT_EQ(Subset2.size(), 2U);

  EntityPointerLevelSet Subset3{ES->getSubsetOf(E3).begin(),
                                ES->getSubsetOf(E3).end()};

  EXPECT_CONTAINS(Subset3, P5);
  EXPECT_EXCLUDES(Subset3, P6);
  EXPECT_EQ(Subset3.size(), 1U);
}

//////////////////////////////////////////////////////////////
//                   Extractor Tests                        //
//////////////////////////////////////////////////////////////

#define CHECK_ENTITY_POINTER_LEVEL(Name, PtrLv, Summary, TestFun, ForReturn)   \
  {                                                                            \
    std::optional<EntityId> Entity_##Name = getEntityId##ForReturn(#Name);     \
    EXPECT_NE(Entity_##Name, std::nullopt);                                    \
    TestFun((Summary),                                                         \
            Builder.buildEntityPointerLevel(*(Entity_##Name), PtrLv));         \
  }
#define CHECK_NO_ENTITY_POINTER_LEVEL(Name, PtrLv, Summary)                    \
  CHECK_ENTITY_POINTER_LEVEL(Name, PtrLv, *Summary, EXPECT_EXCLUDES, )
#define CHECK_HAS_ENTITY_POINTER_LEVEL(Name, PtrLv, Summary)                   \
  CHECK_ENTITY_POINTER_LEVEL(Name, PtrLv, *Summary, EXPECT_CONTAINS, )
#define CHECK_HAS_ENTITY_POINTER_LEVEL_FOR_RETURN(Name, PtrLv, Summary)        \
  CHECK_ENTITY_POINTER_LEVEL(Name, PtrLv, *Summary, EXPECT_CONTAINS, ForReturn)

TEST_F(UnsafeBufferUsageTest, SimpleFunctionWithUnsafePointer) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p) {
      p[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 1U);
}

TEST_F(UnsafeBufferUsageTest, PointerArithmetic) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int *q) {
      *(p + 5);
      *(q - 3);
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 2U);
}

TEST_F(UnsafeBufferUsageTest, PointerIncrementDecrement) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int *q, int *r, int *s) {
      (++p)[5];
      (q++)[5];
      (--r)[5];
      (s--)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(r, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(s, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 4U);
}

TEST_F(UnsafeBufferUsageTest, PointerAssignment) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int *q) {
      (p = q + 5)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 2U);
}

TEST_F(UnsafeBufferUsageTest, CompoundAssignment) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int *q) {
      (p += 5)[5];
      (q -= 3)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 2U);
}

TEST_F(UnsafeBufferUsageTest, MultiLevelPointer) {
  auto Sum = setUpTest(R"cpp(
    void foo(int **p, int **q, int **r) {
      (*p)[5];
      *(*q);
      *(q[5]);
      r[5][5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 2, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(r, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(r, 2, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(q, 2, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 4U);
}

TEST_F(UnsafeBufferUsageTest, ConditionalOperator) {
  auto Sum = setUpTest(R"cpp(
    void foo(int **p, int **q, int cond) {
      (cond ? *p : *q)[5];
      cond ? p[5] : q[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 2, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 2, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 4U);
}

TEST_F(UnsafeBufferUsageTest, CastExpression) {
  auto Sum = setUpTest(R"cpp(
    void foo(void *p, int q) {
      ((int*)p)[5];
      ((int*)q)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(q, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 1U);
}

TEST_F(UnsafeBufferUsageTest, CommaOperator) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int x) {
      (x++, p)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(x, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 1U);
}

TEST_F(UnsafeBufferUsageTest, ParenthesizedExpression) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p) {
      (((p)))[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 1U);
}

TEST_F(UnsafeBufferUsageTest, ArrayParameter) {
  auto Sum = setUpTest(R"cpp(
    void foo(int arr[], int arr2[][10]) {
      int n = 5;
      arr[100];
      arr2[5][n];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(arr, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(arr2, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(arr2, 2, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 3U);
}

TEST_F(UnsafeBufferUsageTest, FunctionCall) {
  auto Sum = setUpTest(R"cpp(
    int ** (*fp)();
    int ** foo() {
      fp = &foo;
      foo()[5];
      (*fp())[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL_FOR_RETURN(foo, 1, Sum);
  // No (foo, 2) becasue indirect calls are ignored.
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 1U);
}

TEST_F(UnsafeBufferUsageTest, StructMemberAccess) {
  auto Sum = setUpTest(R"cpp(
    struct S {
      int *ptr;
      int (*ptr_to_arr)[10];
    };
    void foo(struct S obj) {
      int n = 5;
      obj.ptr[5];
      (*obj.ptr_to_arr)[n];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(ptr, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(ptr_to_arr, 2, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(ptr, 2, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(ptr_to_arr, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 2U);
}

TEST_F(UnsafeBufferUsageTest, StringLiteralSubscript) {
  auto Sum = setUpTest(R"cpp(
    void foo() {
      "hello"[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  // String literals should not generate pointer kind variables
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), static_cast<size_t>(0));
}

TEST_F(UnsafeBufferUsageTest, OpaqueValueExpr) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int *q) {
       (p ?: q)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 1, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(p, 2, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(q, 2, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 2U);
}

TEST_F(UnsafeBufferUsageTest, AddressOfOperator) {
  auto Sum = setUpTest(R"cpp(
    void foo(int x) {
      (&x)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  // Address-of should not generate pointer kind variables for 'x'
  CHECK_NO_ENTITY_POINTER_LEVEL(x, 0, Sum);
  CHECK_NO_ENTITY_POINTER_LEVEL(x, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 0U);
}

TEST_F(UnsafeBufferUsageTest, AddressOfThenDereference) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int *q) {
      (*(&p))[5];
      (&(*q))[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  CHECK_HAS_ENTITY_POINTER_LEVEL(p, 1, Sum);
  CHECK_HAS_ENTITY_POINTER_LEVEL(q, 1, Sum);
  EXPECT_EQ(Sum->getNumUnsafeBuffers(), 2U);
}
} // namespace
