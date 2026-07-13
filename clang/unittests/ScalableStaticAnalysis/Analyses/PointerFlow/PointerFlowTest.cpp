//===- PointerFlowTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "TestFixture.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <type_traits>
#include <variant>

using namespace clang;
using namespace ssaf;

namespace {
// Use FindEntityByName to identify entities in unit tests.
// Unit tests are simple enough to meet the following assumptions:
// - Named declarations should have unique names, they can be found by comparing
//   names with strings;
// - Lambdas should initialize a variable named "X", they can be found using
//   "LambdaOfVar("X")";
// - CXX Ctors should have unique combination of names and number of parameters,
//   they can be found using "CXXCtorOfNumParms(name, numParms)".
struct LambdaOfVar {
  StringRef VarName;
};

struct CXXCtorOfNumParms {
  StringRef CXXCtorName;
  unsigned NumParms;
};

using FindEntityByName =
    std::variant<StringRef, CXXCtorOfNumParms, LambdaOfVar>;

template <typename... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <typename... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

StringRef toStringRef(const FindEntityByName &N) {
  return std::visit(
      Overloaded{
          [](StringRef S) -> StringRef { return S; },
          [](const CXXCtorOfNumParms &L) -> StringRef { return L.CXXCtorName; },
          [](const LambdaOfVar &L) -> StringRef { return L.VarName; },
      },
      N);
}

const NamedDecl *matchNamedDeclByFindEntityByName(const FindEntityByName &N,
                                                  const NamedDecl *D) {
  return std::visit(
      Overloaded{
          [&D](StringRef S) -> const NamedDecl * {
            if (D->getNameAsString() == S)
              return D;
            return nullptr;
          },
          [&D](const CXXCtorOfNumParms &L) -> const NamedDecl * {
            if (auto *CD = dyn_cast<CXXConstructorDecl>(D)) {
              if (CD->getNameAsString() == L.CXXCtorName &&
                  CD->getNumParams() == L.NumParms)
                return D;
            }
            return nullptr;
          },
          [&D](const LambdaOfVar &L) -> const NamedDecl * {
            if (const auto *VD = dyn_cast<VarDecl>(D); VD && VD->getInit()) {
              const Expr *Init = VD->getInit()->IgnoreUnlessSpelledInSource();
              if (isa<LambdaExpr>(Init) && VD->getNameAsString() == L.VarName)
                return cast<LambdaExpr>(Init)->getCallOperator();
            }
            return nullptr;
          },
      },
      N);
}

template <typename SomeDecl = NamedDecl,
          typename = std::enable_if_t<std::is_base_of_v<NamedDecl, SomeDecl>>>
const SomeDecl *findEntityByName(FindEntityByName Name, ASTContext &Ctx) {
  class NamedDeclFinder : public DynamicRecursiveASTVisitor {
  public:
    FindEntityByName SearchingName;
    const SomeDecl *FoundDecl = nullptr;

    NamedDeclFinder(FindEntityByName SearchingName)
        : SearchingName(SearchingName) {
      ShouldVisitTemplateInstantiations = true;
    }

    bool VisitDecl(Decl *D) override {
      if (D->isTemplated())
        return true;
      if (auto *ND = dyn_cast<NamedDecl>(D)) {
        FoundDecl = llvm::dyn_cast_or_null<SomeDecl>(
            matchNamedDeclByFindEntityByName(SearchingName, ND));
        if (FoundDecl)
          return false;
      }
      return true;
    }
  };

  NamedDeclFinder Finder(Name);

  Finder.TraverseDecl(Ctx.getTranslationUnitDecl());
  return dyn_cast_or_null<SomeDecl>(Finder.FoundDecl);
}

const FunctionDecl *findFnByName(FindEntityByName Name, ASTContext &Ctx) {
  return findEntityByName<FunctionDecl>(Name, Ctx);
}

// Same as `std::pair<StringName, unsigned>` for a pair of entity declaration
// name and a pointer level with an extra optional flag for whether the entity
// represents a function return value. This structure is used to explicitly
// spell out components of an EPL such as "{"p", 1}" or "{"foo_fn", 2, true}".
struct EPLPair {
  EPLPair(FindEntityByName Name, unsigned Lv, bool isFunRet = false)
      : Name(Name), Lv(Lv), isFunRet(isFunRet) {}

  FindEntityByName Name;
  unsigned Lv;
  bool isFunRet;
};

class PointerFlowTest : public TestFixture {
protected:
  SSAFOptions Opts;
  TUSummary TUSum;
  TUSummaryBuilder Builder;
  std::unique_ptr<TUSummaryExtractor> Extractor;
  std::unique_ptr<ASTUnit> AST;

  PointerFlowTest()
      : TUSum(llvm::Triple("arm64-apple-macosx"),
              BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp")),
        Builder(TUSum, Opts), Extractor(nullptr) {}

  template <typename ContributorDecl = NamedDecl,
            typename =
                std::enable_if_t<std::is_base_of_v<NamedDecl, ContributorDecl>>>
  bool setUpTest(StringRef Code, const SSAFOptions &Opts = {}) {
    this->Opts = Opts; // Override the options.
    AST = tooling::buildASTFromCodeWithArgs(
        Code, {"-Wno-unused-value", "-Wno-int-to-pointer-cast"});

    for (auto &E : clang::ssaf::TUSummaryExtractorRegistry::entries()) {
      if (E.getName() == PointerFlowEntitySummary::Name) {
        Extractor = E.instantiate(Builder);
        break;
      }
    }

    if (!Extractor) {
      ADD_FAILURE() << "failed to find PointerFlowTUSummaryExtractor";
      return false;
    }
    Extractor->HandleTranslationUnit(AST->getASTContext());
    return true;
  }

  template <typename ContributorDecl = NamedDecl>
  const PointerFlowEntitySummary *getEntitySummary(FindEntityByName Name) {
    const auto *ContributorDefn =
        findEntityByName<ContributorDecl>(Name, AST->getASTContext());

    if (!ContributorDefn) {
      ADD_FAILURE() << "failed to find Decl of \"" << toStringRef(Name) << "\"";
      return nullptr;
    }

    std::optional<EntityId> ContributorEntityId =
        Extractor->addEntity(ContributorDefn);

    if (!ContributorEntityId) {
      ADD_FAILURE() << "failed to get EntityName for contributor \""
                    << toStringRef(Name) << "\"";
      return nullptr;
    }

    auto &TUSumData = getData(TUSum);
    auto EntitiesSumIter =
        TUSumData.find(PointerFlowEntitySummary::summaryName());

    // If none entity summary was collected, it may not be an entry in
    // `TUSumData`:
    if (EntitiesSumIter == TUSumData.end())
      return nullptr;

    auto EntitySumIter = EntitiesSumIter->second.find(*ContributorEntityId);

    // If entity summary is empty, it may not exist:
    if (EntitySumIter == EntitiesSumIter->second.end())
      return nullptr;
    return static_cast<const PointerFlowEntitySummary *>(
        EntitySumIter->second.get());
  }

public:
  std::optional<EntityId> getEntityId(FindEntityByName Name) {
    if (const auto *D = findEntityByName(Name, AST->getASTContext())) {
      return Extractor->addEntity(D);
    }
    return std::nullopt;
  }

  std::optional<EntityId> getEntityIdForReturn(FindEntityByName FunName) {
    if (const auto *D = findFnByName(FunName, AST->getASTContext())) {
      return Extractor->addEntityForReturn(D);
    }
    return std::nullopt;
  }

  EdgeSet makeEdges(unsigned Line, ArrayRef<std::pair<EPLPair, EPLPair>> Edges);
};

// 'ToEPL(Test, Line)' is a lambda that converts a 'EPLPair' to a
// 'EntityPointerLevel':
static constexpr auto ToEPL =
    [](PointerFlowTest *Test,
       unsigned Line) -> std::function<EntityPointerLevel(const EPLPair &)> {
  return [Test, Line](const EPLPair &Pair) -> EntityPointerLevel {
    std::optional<EntityId> Entity = Pair.isFunRet
                                         ? Test->getEntityIdForReturn(Pair.Name)
                                         : Test->getEntityId(Pair.Name);
    if (!Entity) {
      ADD_FAILURE_AT(__FILE__, Line)
          << "Entity not found: " << toStringRef(Pair.Name);
    }
    return buildEntityPointerLevel(*Entity, Pair.Lv);
  };
};

EdgeSet
PointerFlowTest::makeEdges(unsigned Line,
                           ArrayRef<std::pair<EPLPair, EPLPair>> Edges) {
  EdgeSet Result;
  for (auto Edge : Edges)
    Result[ToEPL(this, Line)(Edge.first)].insert(
        ToEPL(this, Line)(Edge.second));
  return Result;
}

TEST_F(PointerFlowTest, IsExtractorRegisteredTest) {
  EXPECT_TRUE(isTUSummaryExtractorRegistered("PointerFlow"));
}

//////////////////////////////////////////////////////////////
//          Simple Assign Tests                             //
//////////////////////////////////////////////////////////////
TEST_F(PointerFlowTest, SimpleAssign) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      q = p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, AssignWithSubscriptLHS) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int **q, int *p, int x) {
      q[x] = p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 2U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, AssignWithPtrArithRHS) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      q = p + 5;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, AssignInSubscript) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      (q = p)[5];
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, MultipleAssign) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p, int *q, int *r) {
      q = p;
      r = q;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"q", 1U}, {"p", 1U}},
                                          {{"r", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, ChainedAssign) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p, int *q, int *r) {
      r = q = p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"q", 1U}, {"p", 1U}},
                                          {{"r", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, CastToRValue) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      q = static_cast<int *&&>(p);
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, AssignToMember) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S { int *field; };
    void foo(S s, int *p) {
      s.field = p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"field", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, AssignToMember2) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S { int *field; };
    void foo(S *s, int *p) {
      s->field = p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"field", 1U}, {"p", 1U}}}));
}

//////////////////////////////////////////////////////////////
//          Call Expr Tests.                                //
//////////////////////////////////////////////////////////////
TEST_F(PointerFlowTest, CallArg) {
  ASSERT_EQ(setUpTest(R"cpp(
    void bar(int *param);
    void foo(int *p) {
      bar(p);
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"param", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, CallMultiArgs) {
  ASSERT_EQ(setUpTest(R"cpp(
    void bar(int *param1, int y, int *param2);
    void foo(int *p, int x, int *q) {
      bar(p, x, q);
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"param1", 1U}, {"p", 1U}},
                                          {{"param2", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, CallAsCallArg) {
  ASSERT_EQ(setUpTest(R"cpp(

    int *bar(int * w);
    void foo(int * p) {
      foo(bar(p));
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"w", 1U}, {"p", 1U}},
                                       {{"p", 1U}, {"bar", 1U, true}}}));
}

TEST_F(PointerFlowTest, CXXOperatorCallMultiArgs) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S {
      int* operator()(int *a, int *b);
    };
    void foo(S obj, int *p, int *q) {
      foo(obj, obj(p, q), obj(p, q));
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                          {{"p", 1U}, {"operator()", 1U, true}},
                                          {{"q", 1U}, {"operator()", 1U, true}},
                                      }));
}

TEST_F(PointerFlowTest, CXXMemberCall) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S {
      int* method(int *a, int *b);
    };
    void foo(S obj, int *p, int *q) {
      foo(obj, obj.method(p, q), obj.method(p, q));
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"a", 1U}, {"p", 1U}},
                                       {{"b", 1U}, {"q", 1U}},
                                       {{"p", 1U}, {"method", 1U, true}},
                                       {{"q", 1U}, {"method", 1U, true}}}));
}

TEST_F(PointerFlowTest, VirtualMethodCall) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct Base {
      virtual void method(int *a);
    };
    void foo(Base &obj, int *p) {
      obj.method(p);
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"a", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, StaticMethodCall) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S {
      static void method(int *a, int *b);
    };
    void foo(int *p, int *q) {
      S::method(p, q);
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, DefaultArg) {
  ASSERT_EQ(setUpTest(R"cpp(
    int *g;
    void bar(int *a, int *b = g);
    void foo(int *p) {
      bar(p);
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__,
                            {{{"a", 1U}, {"p", 1U}}, {{"b", 1U}, {"g", 1U}}}));
}

// Counter-example for the concern that matchArgsWithParams could go OOB
// when fewer explicit args are provided than params (due to default args).
// In Clang's AST, CallExpr::getNumArgs() always includes CXXDefaultArgExpr
// nodes for defaulted parameters, so getNumArgs() >= getNumParams() holds.
TEST_F(PointerFlowTest, AllArgsDefaulted) {
  ASSERT_EQ(setUpTest(R"cpp(
    int *g1, *g2;
    void bar(int *a = g1, int *b = g2);
    void foo() {
      bar();
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"g1", 1U}},
                                          {{"b", 1U}, {"g2", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, DefaultArg2) {
  ASSERT_EQ(setUpTest(R"cpp(
    int *g;
    void bar(int *a, int *b = g);
    void foo(int *p) {
      bar(p, p);
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"p", 1U}},
                                      }));
}

//////////////////////////////////////////////////////////////
//          CXX Ctor Tests.                                 //
//////////////////////////////////////////////////////////////
TEST_F(PointerFlowTest, CXXCtorCallMultiArgs) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S {
      S(int *a, int *b) {}
    };
    void foo(int *p, int *q) {
      S s{p, q};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, CXXCtorCallMultiArgs2) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S {
      S(int *a, int x, int *b) {}
    };
    void foo(int *p, int x, int *q) {
      S s{p, x, q};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, CXXCtorCallAsCallArg) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct Wrapper {
      Wrapper(int *q) {}
    };
    void bar(Wrapper w);
    void foo(int *p) {
      bar(Wrapper{p});
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, DelegatingCXXCtorCall) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S {
      S(int *a, int *b) {}
      S(int *p) : S(p, p) {}
    };
  )cpp"),
            true);

  auto *Sum = getEntitySummary<CXXConstructorDecl>(CXXCtorOfNumParms{"S", 1});

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"p", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, CXXCtorBaseInit) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct Base {
      Base(int *a) {}
    };
    struct Derived : Base {
      Derived(int *p) : Base(p) {}
    };
  )cpp"),
            true);

  auto *Sum = getEntitySummary<CXXConstructorDecl>("Derived");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"a", 1U}, {"p", 1U}}}));
}

//////////////////////////////////////////////////////////////
//          Initializers Tests.                             //
//////////////////////////////////////////////////////////////
TEST_F(PointerFlowTest, LocalVarDeclInit) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p) {
      int *q = p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, LocalVarDeclInit2) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int (*arr)[10]) {
      int (*p)[10] = arr;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"p", 1U}, {"arr", 1U}}}));
}

TEST_F(PointerFlowTest, FieldInit) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p) {
      struct Bar {
        int *field = p;
      };
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("Bar");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"field", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, CXXCtorMemberInit) {
  StringRef Code = R"cpp(
    void foo(int *p) {
      struct Bar {
        int *member;
        Bar(int *q) : member(q) {}
      };
      Bar B{p};
    }
  )cpp";

  ASSERT_EQ(setUpTest(Code), true);
  auto *Sum = getEntitySummary<CXXConstructorDecl>("Bar");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"member", 1U}, {"q", 1U}}}));

  Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, GlobalVarInit) {
  ASSERT_EQ(setUpTest(R"cpp(
    int *q;
    int *g = q;
  )cpp"),
            true);

  auto *Sum = getEntitySummary<VarDecl>("g");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"g", 1U}, {"q", 1U}}}));
}

TEST_F(PointerFlowTest, StaticLocalInit) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p) {
      static int *s = p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"s", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, StaticMemberInit) {
  ASSERT_EQ(setUpTest(R"cpp(
    int *g;
    struct S { static int *member; };
    int *S::member = g;
  )cpp"),
            true);

  auto *Sum = getEntitySummary<VarDecl>("member");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"member", 1U}, {"g", 1U}}}));
}

//////////////////////////////////////////////////////////////
//              InitList Tests.                               //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowTest, ArrayInitList) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p, int *q) {
      int *arr[] = {p, q};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"arr", 2U}, {"p", 1U}},
                                          {{"arr", 2U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, StructInitList) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S { int *a; int *b; };
    void foo(int *p, int *q) {
      S s = {p, q};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                      }));
}

// A union initialized with a brace-enclosed initializer:
TEST_F(PointerFlowTest, UnionInitList) {
  ASSERT_EQ(setUpTest(R"cpp(
    union U { int *x; int y; };
    void foo(int *p) {
      U u = {p};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"x", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, NestedInitList) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct Inner { int * a; int * b; };
    struct S { Inner c; int * d; };
    void foo(int *p, int *q, int *r) {
      S s = {{p, q}, r};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                          {{"d", 1U}, {"r", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, NestedInitList2) {
  ASSERT_EQ(setUpTest(R"cpp(
    union Inner { int * a; int b; };
    struct S { Inner c; int * d; };
    void foo(int *p, int *q) {
      S s = {{p}, q};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"d", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, NestedInitList3) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct Inner { int * a; int * b; };
    union S { Inner c; int * d; };
    void foo(int *p, int *q) {
      S s = {{p, q}};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, NestedArrayInitList) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p, int *q, int *r, int *s) {
      int *arr[][2] = {{p, q}, {r, s}};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"arr", 3U}, {"p", 1U}},
                                          {{"arr", 3U}, {"q", 1U}},
                                          {{"arr", 3U}, {"r", 1U}},
                                          {{"arr", 3U}, {"s", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, MixedNestedArrayStructInitList) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct T { int *arr[2]; };
    void foo(int *p, int *q, int *r, int *s) {
      T t[2] = {{p, q}, {r, s}};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"arr", 2U}, {"p", 1U}},
                                          {{"arr", 2U}, {"q", 1U}},
                                          {{"arr", 2U}, {"r", 1U}},
                                          {{"arr", 2U}, {"s", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, ArrayOfStructInitList) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S { int *a; int *b; };
    void foo(int *p, int *q, int *r, int *s) {
      S arr[] = {{p, q}, {r, s}};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                          {{"a", 1U}, {"r", 1U}},
                                          {{"b", 1U}, {"s", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, ScalarPointerBraceInit) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int *q;
    void foo() {
      int *p{q};
    }
  )cpp"));

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"p", 1U}, {"q", 1U}}}));
}

TEST_F(PointerFlowTest, EmptyInitsScalarInt) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo() {
      int x{};
      int y = {};
      int *p{};
      int *q = {};
    }
  )cpp"));

  // No pointer-flow edge for 0-initialized scalar.
  ASSERT_FALSE(getEntitySummary("foo"));
}

TEST_F(PointerFlowTest, EmptyInitsUnion) {
  ASSERT_TRUE(setUpTest(R"cpp(
    union U { int x; int *p; };
    void foo() {
      U u{};
      U uu = {};
    }
  )cpp"));

  auto *Sum = getEntitySummary("foo");

  ASSERT_EQ(Sum, nullptr);
}

TEST_F(PointerFlowTest, EmptyInitsStruct) {
  ASSERT_TRUE(setUpTest(R"cpp(
    struct S { int x; int *p; };
    void foo() {
      S s{};
      S ss = {};
    }
  )cpp"));

  auto *Sum = getEntitySummary("foo");

  ASSERT_EQ(Sum, nullptr);
}

TEST_F(PointerFlowTest, EmptyInitsClass) {
  ASSERT_TRUE(setUpTest(R"cpp(
    class C { public: int x; int *p; };
    void foo() {
      C c{};
      C cc = {};
    }
  )cpp"));

  auto *Sum = getEntitySummary("foo");

  ASSERT_EQ(Sum, nullptr);
}

//////////////////////////////////////////////////////////////
//              Return Tests.                               //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowTest, ReturnEdge) {
  ASSERT_EQ(setUpTest(R"cpp(
    int *foo(int *p) {
      return p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"foo", 1U, true}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, MultipleReturnEdges) {
  ASSERT_EQ(setUpTest(R"cpp(
    int *foo(int *p, int *q, bool cond) {
      if (cond)
        return p;
      return q;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"foo", 1U, true}, {"p", 1U}},
                                          {{"foo", 1U, true}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, NoReturnEdgeForNonPointerReturnType) {
  ASSERT_EQ(setUpTest(R"cpp(
    int foo(int *p, int x) {
      return x;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  EXPECT_THAT(Sum, testing::AnyOf(testing::IsNull(),
                                  testing::Pointee(makeEdges(__LINE__, {}))));
}

TEST_F(PointerFlowTest, ReturnEdgeNotFromNestedFunction) {
  StringRef Code = R"cpp(
    int *foo(int *p) {
      struct Inner {
        int *bar(int *q) { return q; }
      };
      return p;
    }
  )cpp";

  ASSERT_EQ(setUpTest(Code), true);
  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"foo", 1U, true}, {"p", 1U}}}));

  Sum = getEntitySummary("bar");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"bar", 1U, true}, {"q", 1U}}}));
}

TEST_F(PointerFlowTest, ReturnEdgeInClassMethod) {
  ASSERT_EQ(setUpTest(R"cpp(
  void foo() {
    struct S {
      int *method(int *p, int *q) { return p; }
    };
  }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("method");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"method", 1U, true}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, NoEdgeFromIndirectCall) {
  ASSERT_EQ(setUpTest(R"cpp(
    void bar(int *param1);
    void baz(int *param2);

    void foo(int *p, void (*fp)(int *)) {
      fp(p);
    }

    int main() {
      int *q;
      foo(q, bar);
      foo(q, baz);
      return 0;
    }
  )cpp"),
            true);

  /* FIXME or TBD: Currently indirect calls produce no edge: */
  auto *Sum = getEntitySummary("foo");

  EXPECT_THAT(Sum, testing::AnyOf(testing::IsNull(),
                                  testing::Pointee(makeEdges(__LINE__, {}))));
}

//////////////////////////////////////////////////////////////
//          Lambda Tests.                                   //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowTest, ReturnInLambda) {
  StringRef Code = R"cpp(
    int* foo(int *p) {
      auto local = [](int *r) { return r; };
      return local(p);
    }
  )cpp";

  ASSERT_EQ(setUpTest(Code), true);
  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"r", 1U}, {"p", 1U}},
                                       {{"foo", 1U, true},
                                        {LambdaOfVar{"local"}, 1U, true}}}));

  Sum = getEntitySummary(LambdaOfVar{"local"});
  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__,
                            {{{LambdaOfVar{"local"}, 1U, true}, {"r", 1U}}}));
}

TEST_F(PointerFlowTest, NestedLambdaAssign) {
  StringRef Code = R"cpp(
    void foo() {
      auto outer_lambda = [](int *r, int *s) {
        s = r;
        auto inner_lambda = [](int *x, int *y) { y = x; };
      };
    }
  )cpp";

  ASSERT_EQ(setUpTest(Code), true);
  auto *Sum = getEntitySummary(LambdaOfVar{"outer_lambda"});
  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"s", 1U}, {"r", 1U}}}));

  Sum = getEntitySummary(LambdaOfVar{"inner_lambda"});
  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"y", 1U}, {"x", 1U}}}));
}

TEST_F(PointerFlowTest, ImplicitValueInit) {
  ASSERT_EQ(setUpTest(R"cpp(
    struct S { int *a; int *b; };
    void foo(int *p) {
      S s = {p};  // ImplicitValueInit inits 'b'
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"a", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, InitListExpr) {
  ASSERT_EQ(setUpTest(R"cpp(
    void foo(int *p) {
      int *q = {p};
      int *r = {};
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, CXXDefaultInitExpr) {
  ASSERT_EQ(setUpTest(R"cpp(
    int *g;
    struct S {
      int *field = g;
    };
    void foo(int *p) {
      S s;
      s.field = p;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary<RecordDecl>("S");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"field", 1U}, {"g", 1U}}}));
}

TEST_F(PointerFlowTest, CXXConstructExprArrayInit) {
  ASSERT_EQ(setUpTest(
                R"cpp(
    struct S {};
    void foo(S *q) {
      S arr[3]; // -VarDecl <...> arr 'S[3]' callinit
                //   `-CXXConstructExpr <...> 'S[3]' 'void () noexcept'
      q = arr;
    }
  )cpp"),
            true);

  auto *Sum = getEntitySummary("foo");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"arr", 1U}}}));
}

//////////////////////////////////////////////////////////////
// Template is ignored but instantiations are visited.      //
//////////////////////////////////////////////////////////////
TEST_F(PointerFlowTest, FunctionTemplate) {
  ASSERT_TRUE(setUpTest(R"cpp(
    template <typename T>
    T* f(T *p) {
      int *q = p;
      return q;
    }
  )cpp"));
  ASSERT_FALSE(findEntityByName("f", AST->getASTContext()));
}

TEST_F(PointerFlowTest, MethodInClassTemplate) {
  ASSERT_TRUE(setUpTest(R"cpp(
    template <typename T>
    struct Wrapper {
      T *ptr;
      void set(T *p) { ptr = p; }
    };
  )cpp"));
  ASSERT_FALSE(findEntityByName("set", AST->getASTContext()));
}

TEST_F(PointerFlowTest, FunctionTemplateInstantiation) {
  ASSERT_TRUE(setUpTest(R"cpp(
    template <typename T>
    T* f(T *p) {
      int *q = p;
      return q;
    }

    void test(int *p) {
      f(p);
    }
  )cpp"));

  auto *Sum = getEntitySummary<FunctionDecl>("f");

  ASSERT_TRUE(Sum);
  ASSERT_EQ(*Sum, makeEdges(__LINE__, {{{"q", 1U}, {"p", 1U}},
                                       {{"f", 1U, 1}, {"q", 1U}}}));
}

TEST_F(PointerFlowTest, MethodInClassTemplateInstantiation) {
  ASSERT_TRUE(setUpTest(R"cpp(
    template <typename T>
    struct Wrapper {
      T *ptr;
      void set(T *p) { ptr = p; }
    };

    void test(int *p) {
      Wrapper<int> W;

      W.set(p);
    }
  )cpp"));

  auto *Sum = getEntitySummary<FunctionDecl>("set");

  ASSERT_TRUE(Sum);
  ASSERT_EQ(*Sum, makeEdges(__LINE__, {{{"ptr", 1U}, {"p", 1U}}}));
}

//////////////////////////////////////////////////////////////
//          Robustness Tests (No Crash Tests)               //
//////////////////////////////////////////////////////////////

#ifndef NDEBUG
TEST_F(PointerFlowTest, StructuredBindingWithPointers) {
  StringRef Code = R"cpp(
    void foo() {
      int *a[2];
      auto [ptr1, ptr2] = a;
      ptr1[5];
      ptr2[3];
    }
  )cpp";

  // BindingDecl may not be fully supported, but should not crash.
  llvm::SaveAndRestore<bool> DebugFlag(llvm::DebugFlag, true);
  llvm::setCurrentDebugType("ssaf-analyses");
  testing::internal::CaptureStderr();

  ASSERT_TRUE(setUpTest(Code));
  // Verify the warning was logged
  ASSERT_TRUE(StringRef(testing::internal::GetCapturedStderr())
                  .contains("failed to create EntityId for Decomposition"));
}
#endif

TEST_F(PointerFlowTest, RHSResultsInNoEntityPointerLevel) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void f() {
      int *p = new int[10];
      const char *q = "hello";
    }
    struct S {
      int *p;
      void g() { p = (int *)this; }
    };
  )cpp"));
  ASSERT_FALSE(getEntitySummary<FunctionDecl>("f"));
  ASSERT_FALSE(getEntitySummary<CXXMethodDecl>("g"));
}

//////////////////////////////////////////////////////////////
//          Reference-to-pointer Tests                      //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowTest, ArgToRefParam) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void callee(int *&rp);
    void caller(int *p) {
      callee(p);
    }
  )cpp"));

  auto *Sum = getEntitySummary("caller");

  ASSERT_TRUE(Sum);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"rp", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, ArgToRefParamLevel2) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void callee(int **&rp);
    void caller(int **pp) {
      callee(pp);
    }
  )cpp"));

  auto *Sum = getEntitySummary("caller");

  ASSERT_TRUE(Sum);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"rp", 1U}, {"pp", 1U}}}));
}

TEST_F(PointerFlowTest, InitRefPtr) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void foo(int *p) {
      int *&rp = p;
      int * const & crp = p;
    }
  )cpp"));

  auto *Sum = getEntitySummary("foo");

  ASSERT_TRUE(Sum);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"rp", 1U}, {"p", 1U}},
                                       {{"crp", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, ReturnRefPtr) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int *& f();
    int *& foo() {
      return f();
    }
  )cpp"));

  auto *Sum = getEntitySummary("foo");

  ASSERT_TRUE(Sum);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"foo", 1U, true}, {"f", 1U, true}}}));
}

TEST_F(PointerFlowTest, RefBindVarInitFileScope) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int *gp;
    int *& gr = gp;
  )cpp"));

  auto *Sum = getEntitySummary<VarDecl>("gr");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"gr", 1U}, {"gp", 1U}}}));
}

TEST_F(PointerFlowTest, RefBindCallArgMultiple) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void h(int *& a, int *& b);
    void f(int *p, int *q) { h(p, q); }
  )cpp"));

  auto *Sum = getEntitySummary("f");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"a", 1U}, {"p", 1U}},
                                          {{"b", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, RefBindCallArgDefaulted) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int *global_default;
    void h(int *& x = global_default);
    void f() { h(); }
  )cpp"));

  auto *Sum = getEntitySummary("f");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"x", 1U}, {"global_default", 1U}}}));
}

TEST_F(PointerFlowTest, RefBindMemberInit) {
  ASSERT_TRUE(setUpTest(R"cpp(
    struct S { int *& m; S(int *& p) : m(p) {} };
  )cpp"));

  auto *Sum = getEntitySummary<CXXConstructorDecl>("S");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"m", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, RefBindReturnStmt) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int *& f(int *& p) { return p; }
  )cpp"));

  auto *Sum = getEntitySummary("f");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"f", 1U, true}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, RefBindMultiLevel) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int **gpp;
    int **& r = gpp;
  )cpp"));

  auto *Sum = getEntitySummary<VarDecl>("r");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"r", 1U}, {"gpp", 1U}}}));
}

TEST_F(PointerFlowTest, RefBindTernaryInit) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void f(bool cond, int *p, int *q) {
      int *& r = (cond ? p : q);
      (void)r;
    }
  )cpp"));

  auto *Sum = getEntitySummary("f");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {
                                          {{"r", 1U}, {"p", 1U}},
                                          {{"r", 1U}, {"q", 1U}},
                                      }));
}

TEST_F(PointerFlowTest, RefBindNonPointerReferenceIgnored) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void f() { int x = 0; int &r = x; (void)r; }
  )cpp"));

  EXPECT_EQ(getEntitySummary("f"), nullptr);
}

TEST_F(PointerFlowTest, RefBindArraySubscriptInInitializer) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void f(int *arr[], int i) { int *& r = arr[i]; (void)r; }
  )cpp"));

  auto *Sum = getEntitySummary("f");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"r", 1U}, {"arr", 2U}}}));
}

TEST_F(PointerFlowTest, RefBindExplicitCastInInitializer) {
  ASSERT_TRUE(setUpTest(R"cpp(
    void f(int *p) { int *& r = (int *&)p; (void)r; }
  )cpp"));

  auto *Sum = getEntitySummary("f");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"r", 1U}, {"p", 1U}}}));
}

TEST_F(PointerFlowTest, RefBindFunctionCallInitializer) {
  ASSERT_TRUE(setUpTest(R"cpp(
    int *& g();
    void f() { int *& r = g(); (void)r; }
  )cpp"));

  auto *Sum = getEntitySummary("f");

  ASSERT_NE(Sum, nullptr);
  EXPECT_EQ(*Sum, makeEdges(__LINE__, {{{"r", 1U}, {"g", 1U, true}}}));
}

//////////////////////////////////////////////////////////////
//          Local-entity inclusion option tests             //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowTest, LocalPointerSkippedByDefault) {
  StringRef Code = R"cpp(
    void foo(int *param) {
      int *local_ptr = param;
      local_ptr = param;
    }
  )cpp";

  ASSERT_TRUE(setUpTest(Code));

  // Without IncludeLocalEntities, the local pointer is not registered as a
  // contributor and therefore has no entity summary of its own.
  EXPECT_FALSE(getEntitySummary<VarDecl>("local_ptr"));
  // The enclosing function still has its summary describing the assignment
  // edge from `local_ptr` to `param` (via the initializer).
  EXPECT_TRUE(getEntitySummary("foo"));
}

TEST_F(PointerFlowTest, LocalPointerReportedWhenIncluded) {
  SSAFOptions Opts;
  Opts.IncludeLocalEntities = true;
  StringRef Code = R"cpp(
    void foo(int *param) {
      int *local_ptr = param;
      local_ptr = param;
    }
  )cpp";

  ASSERT_TRUE(setUpTest(Code, Opts));

  // With the option enabled, the local pointer becomes a contributor and gets
  // its own entity summary describing the same edge that was previously only
  // observable through `foo`.
  const auto *LocalSum = getEntitySummary<VarDecl>("local_ptr");
  ASSERT_TRUE(LocalSum);
  EXPECT_EQ(*LocalSum,
            makeEdges(__LINE__, {{{"local_ptr", 1U}, {"param", 1U}}}));

  EXPECT_TRUE(getEntitySummary("foo"));
}
} // namespace
