//===- PointerFlowPairsTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for `PointerFlowPair` and `PointerFlowPairMatcher`.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowPairs.h"
#include "FindDecl.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

using namespace clang;
using namespace ssaf;

namespace clang::ssaf {
// Defined in the analyses library (SSAFAnalysesCommon); forward-declared here
// so tests can drive the matcher over a contributor exactly as the extractor
// does.
void findMatchesIn(const NamedDecl *Contributor,
                   llvm::function_ref<void(const DynTypedNode &)> MatchAction);
} // namespace clang::ssaf

namespace {

std::string exprToString(const Expr *E, const ASTContext &Ctx) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  E->printPretty(OS, /*Helper=*/nullptr, Ctx.getPrintingPolicy());
  return S;
}

// Visitor for `PointerFlowPair::visitLHS` that pretty prints LHS
struct RenderLHS {
  const ASTContext &Ctx;

  std::string operator()(const ValueDecl *D, bool IsRet) const {
    return D->getNameAsString() + (IsRet ? "(ret)" : "");
  }
  std::string operator()(const Expr *E) const { return exprToString(E, Ctx); }
};

// Render a PointerFlowPair as "(<lhs>, <rhs>)", where both sides are
// rendered by `RenderLHS`/`exprToString`.
std::string pairToString(const PointerFlowPair &P, const ASTContext &Ctx) {
  return "(" + P.visitLHS(RenderLHS{Ctx}) + ", " + exprToString(P.RHS, Ctx) +
         ")";
}

// Finds a CXXConstructorDecl by name and parameter count, to disambiguate
// between overloaded constructors of the same class (e.g. a delegating
// constructor vs. its delegate).
const CXXConstructorDecl *
findCtorByNumParams(StringRef Name, unsigned NumParams, ASTContext &Ctx) {
  class CtorFinder : public DynamicRecursiveASTVisitor {
  public:
    StringRef Name;
    unsigned NumParams;
    const CXXConstructorDecl *Found = nullptr;

    CtorFinder(StringRef Name, unsigned NumParams)
        : Name(Name), NumParams(NumParams) {}

    bool VisitCXXConstructorDecl(CXXConstructorDecl *D) override {
      if (D->getNameAsString() == Name && D->getNumParams() == NumParams) {
        Found = D;
        return false;
      }
      return true;
    }
  };

  CtorFinder Finder(Name, NumParams);
  Finder.TraverseDecl(Ctx.getTranslationUnitDecl());
  return Finder.Found;
}

class PointerFlowPairsTest : public ::testing::Test {
protected:
  std::unique_ptr<ASTUnit> AST;

  bool buildAST(StringRef Code,
                std::vector<std::string> ExtraArgs = {"-Wno-unused-value"}) {
    AST = tooling::buildASTFromCodeWithArgs(Code, ExtraArgs);
    return AST != nullptr;
  }

  ASTContext &ctx() { return AST->getASTContext(); }

  // Drives `PointerFlowPairMatcher` over `Contrib` and returns every matched
  // pair rendered as "(<lhs>, <rhs>)", in traversal order.
  std::vector<std::string> getPairsFor(const NamedDecl *Contrib) {
    std::vector<std::string> Out;
    if (!Contrib) {
      ADD_FAILURE() << "null contributor";
      return Out;
    }
    PointerFlowPairMatcher Matcher(ctx());
    ssaf::findMatchesIn(Contrib, [&](const DynTypedNode &Node) {
      llvm::SmallVector<PointerFlowPair> Pairs;
      Matcher.matches(Node, Contrib, Pairs);
      for (const PointerFlowPair &P : Pairs)
        Out.push_back(pairToString(P, ctx()));
    });
    return Out;
  }

  template <typename ContributorDecl = NamedDecl>
  std::vector<std::string> getPairs(StringRef Name) {
    const auto *Contrib = findDeclByName<ContributorDecl>(Name, ctx());
    if (!Contrib) {
      ADD_FAILURE() << "failed to find Decl of \"" << Name.str() << "\"";
      return {};
    }
    return getPairsFor(Contrib);
  }
};

//////////////////////////////////////////////////////////////
//          Basic pair matching.                             //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowPairsTest, VarDeclInit) {
  ASSERT_TRUE(buildAST(R"cpp(
    void foo(int *p) {
      int *q = p;
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(q, p)"));
}

TEST_F(PointerFlowPairsTest, ReturnStmt) {
  ASSERT_TRUE(buildAST(R"cpp(
    int *foo(int *p) {
      return p;
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(foo(ret), p)"));
}

//////////////////////////////////////////////////////////////
//          No-match.                                       //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowPairsTest, NoPairForNonPointerAssign) {
  ASSERT_TRUE(buildAST(R"cpp(
    void foo(int a, int b) {
      a = b;
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::IsEmpty());
}

TEST_F(PointerFlowPairsTest, NoPairForUninitializedVar) {
  ASSERT_TRUE(buildAST(R"cpp(
    void foo() {
      int *p;
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::IsEmpty());
}

//////////////////////////////////////////////////////////////
//          Call / Ctor argument passing.                   //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowPairsTest, CallArgMatching) {
  ASSERT_TRUE(buildAST(R"cpp(
    void bar(int *param1, int y, int *param2);
    void foo(int *p, int x, int *q) {
      bar(p, x, q);
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"),
              testing::UnorderedElementsAre("(param1, p)", "(param2, q)"));
}

TEST_F(PointerFlowPairsTest, CXXOperatorCallSkipsImplicitObjectArgument) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S { int *operator()(int *a, int *b); };
    void foo(S obj, int *p, int *q) {
      obj(p, q);
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"),
              testing::UnorderedElementsAre("(a, p)", "(b, q)"));
}

TEST_F(PointerFlowPairsTest, CXXConstructExprArgMatching) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S { S(int *a, int *b) {} };
    void foo(int *p, int *q) {
      S s{p, q};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"),
              testing::UnorderedElementsAre("(a, p)", "(b, q)"));
}

TEST_F(PointerFlowPairsTest, MemberInitializer) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S {
      int *member;
      S(int *q) : member(q) {}
    };
  )cpp"));

  EXPECT_THAT(getPairs<CXXConstructorDecl>("S"),
              testing::ElementsAre("(member, q)"));
}

// The delegate target's arg-to-param pairs are found by the generic AST
// traversal visiting its underlying CXXConstructExpr directly (not by an
// explicit recursive call in `findUntypedPairsInDecl` -- an earlier version
// did that too, double-counting these pairs, since the traversal already
// visits every written constructor-initializer's init expr on its own).
TEST_F(PointerFlowPairsTest, DelegatingCtorMatchesDelegateInitExactlyOnce) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S {
      S(int *a, int *b) {}
      S(int *p) : S(p, p) {}
    };
  )cpp"));

  const auto *Delegator = findCtorByNumParams("S", 1, ctx());
  ASSERT_TRUE(Delegator);
  EXPECT_THAT(getPairsFor(Delegator),
              testing::UnorderedElementsAre("(a, p)", "(b, p)"));
}

// Same as above, for a base-initializer's underlying CXXConstructExpr.
TEST_F(PointerFlowPairsTest, BaseCtorInitializerMatchesBaseInitExactlyOnce) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct Base { Base(int *a) {} };
    struct Derived : Base { Derived(int *p) : Base(p) {} };
  )cpp"));

  EXPECT_THAT(getPairs<CXXConstructorDecl>("Derived"),
              testing::ElementsAre("(a, p)"));
}

//////////////////////////////////////////////////////////////
//          Initializer-list decomposition.                 //
//////////////////////////////////////////////////////////////

TEST_F(PointerFlowPairsTest, RecordInitListDecomposesPerField) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S { int *a; int *b; };
    void foo(int *p, int *q) {
      S s = {p, q};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"),
              testing::UnorderedElementsAre("(a, p)", "(b, q)"));
}

// Record decomposition applies the same way when the record-typed pair comes
// from call-argument matching rather than a VarDecl initializer.
TEST_F(PointerFlowPairsTest, CallArgRecordInitListDecomposesPerField) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S { int *a; int *b; };
    void bar(S s);
    void foo(int *p, int *q) {
      bar({p, q});
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"),
              testing::UnorderedElementsAre("(a, p)", "(b, q)"));
}

TEST_F(PointerFlowPairsTest, RecordWithBaseClassInitListIsDropped) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct Base { int *x; };
    struct Derived : Base { int *y; };
    void foo(int *p, int *q) {
      Derived d = {p, q};
    }
  )cpp",
                       {"-std=c++17", "-Wno-unused-value"}));

  EXPECT_THAT(getPairs("foo"), testing::IsEmpty());
}

TEST_F(PointerFlowPairsTest, UnionInitListPicksActiveField) {
  ASSERT_TRUE(buildAST(R"cpp(
    union U { int *x; int y; };
    void foo(int *p) {
      U u = {p};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(x, p)"));
}

TEST_F(PointerFlowPairsTest, UnionEmptyInitListProducesNoPair) {
  ASSERT_TRUE(buildAST(R"cpp(
    union U { int *x; int y; };
    void foo() {
      U u = {};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::IsEmpty());
}

TEST_F(PointerFlowPairsTest, ArrayOfPointersInitListIsKeptWhole) {
  ASSERT_TRUE(buildAST(R"cpp(
    void foo(int *p, int *q) {
      int *arr[] = {p, q};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(arr, {p, q})"));
}

TEST_F(PointerFlowPairsTest, ArrayOfScalarsInitListIsAlsoKeptWhole) {
  ASSERT_TRUE(buildAST(R"cpp(
    void foo(int x, int y) {
      int arr[] = {x, y};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(arr, {x, y})"));
}

TEST_F(PointerFlowPairsTest, CallArgArrayInitListBoundToReferenceIsKeptWhole) {
  ASSERT_TRUE(buildAST(R"cpp(
    void bar(int * const (&param)[2]);
    void foo(int *p, int *q) {
      bar({p, q});
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(param, {p, q})"));
}

TEST_F(PointerFlowPairsTest, RecordFieldOfArrayOfPointersKeepsInitListWhole) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S { int *arr[2]; };
    S foo(int *p, int *q) {
      return {p, q};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(arr, {p, q})"));
}

// Unlike an array of pointers, an array of records IS decomposed per-element,
// since each element is itself a record init-list.
TEST_F(PointerFlowPairsTest, ArrayOfRecordsInitListIsDecomposedPerElement) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S { int *a; int *b; };
    void foo(int *p, int *q, int *r, int *s) {
      S arr[] = {{p, q}, {r, s}};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::UnorderedElementsAre(
                                   "(a, p)", "(b, q)", "(a, r)", "(b, s)"));
}

// An unnamed bit-field consumes no slot in the semantic InitListExpr, so the
// field after it must still be paired with the right initializer.
TEST_F(PointerFlowPairsTest, StructInitListWithUnnamedBitFieldSkipsBitField) {
  ASSERT_TRUE(buildAST(R"cpp(
    struct S { int a; int : 4; int *p; };
    void foo(int a, int *q) {
      S s = {a, q};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(p, q)"));
}

TEST_F(PointerFlowPairsTest, EmptyInitListForScalarProducesNoPair) {
  ASSERT_TRUE(buildAST(R"cpp(
    void foo() {
      int *q = {};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::IsEmpty());
}

TEST_F(PointerFlowPairsTest, SingletonInitListForScalarRecursesToElement) {
  ASSERT_TRUE(buildAST(R"cpp(
    void foo(int *p) {
      int *q = {p};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(q, p)"));
}

TEST_F(PointerFlowPairsTest, AssignRHSInitListPeelsSingletonForScalarLHS) {
  ASSERT_TRUE(buildAST(R"cpp(
    void foo(int *p, int *q, int *r) {
      q = {p};
      r = {};
    }
  )cpp"));

  EXPECT_THAT(getPairs("foo"), testing::ElementsAre("(q, p)"));
}

} // namespace
