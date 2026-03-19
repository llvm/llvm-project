//===- UnsafeBufferUsageTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsageExtractor.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ssaf;
using testing::UnorderedElementsAre;

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

constexpr inline auto buildEntityPointerLevel =
    UnsafeBufferUsageTUSummaryExtractor::buildEntityPointerLevel;

class UnsafeBufferUsageTest : public testing::Test {
protected:
  TUSummary TUSum;
  TUSummaryBuilder Builder;
  UnsafeBufferUsageTUSummaryExtractor Extractor;
  std::unique_ptr<ASTUnit> AST;

  UnsafeBufferUsageTest()
      : TUSum(BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp")),
        Builder(TUSum), Extractor(Builder) {}

  std::unique_ptr<UnsafeBufferUsageEntitySummary>
  setUpTest(StringRef Code, StringRef ContributorName) {
    AST = tooling::buildASTFromCodeWithArgs(
        Code, {"-Wno-unused-value -Wno-int-to-pointer-cast"});

    const auto *ContributorDefn =
        findDeclByName(ContributorName, AST->getASTContext());
    std::optional<EntityName> EN = getEntityName(ContributorDefn);

    if (!ContributorDefn || !EN)
      return nullptr;

    llvm::Error Error = llvm::ErrorSuccess();
    auto Sum = Extractor.extractEntitySummary(ContributorDefn,
                                              AST->getASTContext(), Error);

    if (Error) {
      llvm::consumeError(std::move(Error));
      return nullptr;
    }
    return Sum;
  }

  std::optional<EntityId> getEntityId(StringRef Name) {
    if (const auto *D = findDeclByName(Name, AST->getASTContext()))
      if (auto EntityName = getEntityName(D))
        return Extractor.addEntity(*EntityName);
    return std::nullopt;
  }

  std::optional<EntityId> getEntityIdForReturn(StringRef FunName) {
    if (const auto *D = findFnByName(FunName, AST->getASTContext()))
      if (auto EntityName = getEntityNameForReturn(D))
        return Extractor.addEntity(*EntityName);
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
  EntityId E1 = Extractor.addEntity({"c:@F@foo", "", {}});
  EntityId E2 = Extractor.addEntity({"c:@F@bar", "", {}});

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
  EntityId E1 = Extractor.addEntity({"c:@F@foo", "", {}});
  EntityId E2 = Extractor.addEntity({"c:@F@bar", "", {}});
  EntityId E3 = Extractor.addEntity({"c:@F@baz", "", {}});

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
  auto Sum = setUpTest(R"cpp(
    void foo(int *p) {
      p[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
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
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}}));
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
  EXPECT_EQ(*Sum,
            makeSet(__LINE__, {{"p", 1U}, {"q", 1U}, {"r", 1U}, {"s", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, PointerAssignment) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int *q) {
      (p = q + 5)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}}));
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
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}}));
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
  EXPECT_EQ(*Sum,
            makeSet(__LINE__, {{"p", 2U}, {"q", 1U}, {"r", 1U}, {"r", 2U}}));
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
  EXPECT_EQ(*Sum,
            makeSet(__LINE__, {{"p", 1U}, {"q", 1U}, {"p", 2U}, {"q", 2U}}));
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
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, CommaOperator) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int x) {
      (x++, p)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
}

TEST_F(UnsafeBufferUsageTest, CommaOperator2) {
  auto Sum = setUpTest(R"cpp(
    void foo(int **p, int **q, int x) {
      (p[x] = 0, q[x] = 0)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}, {"q", 2U}}));
}

TEST_F(UnsafeBufferUsageTest, ParenthesizedExpression) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p) {
      (((p)))[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}}));
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
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"arr", 1U}, {"arr2", 1U}, {"arr2", 2U}}));
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
  // No (foo, 2) becasue indirect calls are ignored.
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"foo", 1U, true}}));
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
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"ptr", 1U}, {"ptr_to_arr", 2U}}));
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
  EXPECT_EQ(*Sum, makeSet(__LINE__, {}));
}

TEST_F(UnsafeBufferUsageTest, OpaqueValueExpr) {
  auto Sum = setUpTest(R"cpp(
    void foo(int *p, int *q) {
       (p ?: q)[5];
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1U}, {"q", 1U}}));
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
  EXPECT_EQ(*Sum, makeSet(__LINE__, {}));
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
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 1}, {"q", 1}}));
}

TEST_F(UnsafeBufferUsageTest, PointerToArrayOfPointers) {
  auto Sum = setUpTest(R"cpp(
    void foo() {
      int * arr[10];
      int * (*p)[10] = arr;

      (*p)[5][5]; // '(*p)[5]' is unsafe 
                  // '(*p)' is fine because 5 < 10
    }
  )cpp",
                       "foo");

  EXPECT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeSet(__LINE__, {{"p", 3}}));
}
} // namespace
