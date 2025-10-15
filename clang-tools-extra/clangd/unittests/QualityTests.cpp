//===-- QualityTests.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Evaluating scoring functions isn't a great fit for assert-based tests.
// For interesting cases, both exact scores and "X beats Y" are too brittle to
// make good hard assertions.
//
// Here we test the signal extraction and sanity-check that signals point in
// the right direction. This should be supplemented by quality metrics which
// we can compute from a corpus of queries and preferred rankings.
//
//===----------------------------------------------------------------------===//

#include "FileDistance.h"
#include "Quality.h"
#include "TestFS.h"
#include "TestTU.h"
#include "index/FileIndex.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/Support/Casting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace clangd {

// Force the unittest URI scheme to be linked,
static int LLVM_ATTRIBUTE_UNUSED UnittestSchemeAnchorDest =
    UnittestSchemeAnchorSource;

namespace {

TEST(QualityTests, SymbolQualitySignalExtraction) {
  auto Header = TestTU::withHeaderCode(R"cpp(
    int _X;

    [[deprecated]]
    int _f() { return _X; }

    #define DECL_NAME(x, y) x##_##y##_Decl
    #define DECL(x, y) class DECL_NAME(x, y) {};
    DECL(X, Y); // X_Y_Decl
  )cpp");

  auto Symbols = Header.headerSymbols();
  auto AST = Header.build();

  SymbolQualitySignals Quality;
  Quality.merge(findSymbol(Symbols, "X_Y_Decl"));
  EXPECT_TRUE(Quality.ImplementationDetail);

  Symbol F = findSymbol(Symbols, "_f");
  F.References = 24; // TestTU doesn't count references, so fake it.
  Quality = {};
  Quality.merge(F);
  EXPECT_TRUE(Quality.Deprecated);
  EXPECT_FALSE(Quality.ReservedName);
  EXPECT_EQ(Quality.References, 24u);
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Function);

  Quality = {};
  Quality.merge(CodeCompletionResult(&findDecl(AST, "_f"), /*Priority=*/42));
  EXPECT_TRUE(Quality.Deprecated);
  EXPECT_FALSE(Quality.ReservedName);
  EXPECT_EQ(Quality.References, SymbolQualitySignals().References);
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Function);

  Quality = {};
  Quality.merge(CodeCompletionResult("if"));
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Keyword);

  // Testing ReservedName in main file, we don't index those symbols in headers.
  auto MainAST = TestTU::withCode("int _X;").build();
  SymbolSlab MainSymbols = std::get<0>(indexMainDecls(MainAST));

  Quality = {};
  Quality.merge(findSymbol(MainSymbols, "_X"));
  EXPECT_FALSE(Quality.Deprecated);
  EXPECT_FALSE(Quality.ImplementationDetail);
  EXPECT_TRUE(Quality.ReservedName);
}

TEST(QualityTests, SymbolRelevanceSignalExtraction) {
  TestTU Test;
  Test.HeaderCode = R"cpp(
  int header();
  int header_main();

  namespace hdr { class Bar {}; } // namespace hdr

  #define DEFINE_FLAG(X) \
  namespace flags { \
  int FLAGS_##X; \
  } \

  DEFINE_FLAG(FOO)
  )cpp";
  Test.Code = R"cpp(
  using hdr::Bar;

  using flags::FLAGS_FOO;

  int ::header_main() { return 0; }
  int main();

  [[deprecated]]
  int deprecated() { return 0; }

  namespace { struct X { void y() { int z; } }; }
  struct S{};
  )cpp";
  auto AST = Test.build();

  SymbolRelevanceSignals Relevance;
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "deprecated"),
                                       /*Priority=*/42,
                                       /*Qualifier=*/std::nullopt,
                                       /*QualifierIsInformative=*/false,
                                       /*Accessible=*/false));
  EXPECT_EQ(Relevance.NameMatch, SymbolRelevanceSignals().NameMatch);
  EXPECT_TRUE(Relevance.Forbidden);
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::GlobalScope);

  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "main"), 42));
  EXPECT_FLOAT_EQ(Relevance.SemaFileProximityScore, 1.0f)
      << "Decl in current file";
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "header"), 42));
  EXPECT_FLOAT_EQ(Relevance.SemaFileProximityScore, 0.6f) << "Decl from header";
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "header_main"), 42));
  EXPECT_FLOAT_EQ(Relevance.SemaFileProximityScore, 1.0f)
      << "Current file and header";

  auto ConstructShadowDeclCompletionResult = [&](const std::string DeclName) {
    auto *Shadow =
        *dyn_cast<UsingDecl>(&findDecl(AST, [&](const NamedDecl &ND) {
           if (const UsingDecl *Using = dyn_cast<UsingDecl>(&ND))
             if (Using->shadow_size() &&
                 Using->getQualifiedNameAsString() == DeclName)
               return true;
           return false;
         }))->shadow_begin();
    CodeCompletionResult Result(Shadow->getTargetDecl(), 42);
    Result.ShadowDecl = Shadow;
    return Result;
  };

  Relevance = {};
  Relevance.merge(ConstructShadowDeclCompletionResult("Bar"));
  EXPECT_FLOAT_EQ(Relevance.SemaFileProximityScore, 1.0f)
      << "Using declaration in main file";
  Relevance.merge(ConstructShadowDeclCompletionResult("FLAGS_FOO"));
  EXPECT_FLOAT_EQ(Relevance.SemaFileProximityScore, 1.0f)
      << "Using declaration in main file";

  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findUnqualifiedDecl(AST, "X"), 42));
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::FileScope);
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findUnqualifiedDecl(AST, "y"), 42));
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::ClassScope);
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findUnqualifiedDecl(AST, "z"), 42));
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::FunctionScope);
  // The injected class name is treated as the outer class name.
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "S::S"), 42));
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::GlobalScope);

  Relevance = {};
  EXPECT_FALSE(Relevance.InBaseClass);
  auto BaseMember = CodeCompletionResult(&findUnqualifiedDecl(AST, "y"), 42);
  BaseMember.InBaseClass = true;
  Relevance.merge(BaseMember);
  EXPECT_TRUE(Relevance.InBaseClass);

  auto Index = Test.index();
  FuzzyFindRequest Req;
  Req.Query = "X";
  Req.AnyScope = true;
  bool Matched = false;
  Index->fuzzyFind(Req, [&](const Symbol &S) {
    Matched = true;
    Relevance = {};
    Relevance.merge(S);
    EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::FileScope);
  });
  EXPECT_TRUE(Matched);
}

// Do the signals move the scores in the direction we expect?
TEST(QualityTests, SymbolQualitySignalsSanity) {
  SymbolQualitySignals Default;
  EXPECT_EQ(Default.evaluateHeuristics(), 1);

  SymbolQualitySignals Deprecated;
  Deprecated.Deprecated = true;
  EXPECT_LT(Deprecated.evaluateHeuristics(), Default.evaluateHeuristics());

  SymbolQualitySignals ReservedName;
  ReservedName.ReservedName = true;
  EXPECT_LT(ReservedName.evaluateHeuristics(), Default.evaluateHeuristics());

  SymbolQualitySignals ImplementationDetail;
  ImplementationDetail.ImplementationDetail = true;
  EXPECT_LT(ImplementationDetail.evaluateHeuristics(),
            Default.evaluateHeuristics());

  SymbolQualitySignals WithReferences, ManyReferences;
  WithReferences.References = 20;
  ManyReferences.References = 1000;
  EXPECT_GT(WithReferences.evaluateHeuristics(), Default.evaluateHeuristics());
  EXPECT_GT(ManyReferences.evaluateHeuristics(),
            WithReferences.evaluateHeuristics());

  SymbolQualitySignals Keyword, Variable, Macro, Constructor, Function,
      Destructor, Operator;
  Keyword.Category = SymbolQualitySignals::Keyword;
  Variable.Category = SymbolQualitySignals::Variable;
  Macro.Category = SymbolQualitySignals::Macro;
  Constructor.Category = SymbolQualitySignals::Constructor;
  Destructor.Category = SymbolQualitySignals::Destructor;
  Destructor.Category = SymbolQualitySignals::Destructor;
  Operator.Category = SymbolQualitySignals::Operator;
  Function.Category = SymbolQualitySignals::Function;
  EXPECT_GT(Variable.evaluateHeuristics(), Default.evaluateHeuristics());
  EXPECT_GT(Keyword.evaluateHeuristics(), Variable.evaluateHeuristics());
  EXPECT_LT(Macro.evaluateHeuristics(), Default.evaluateHeuristics());
  EXPECT_LT(Operator.evaluateHeuristics(), Default.evaluateHeuristics());
  EXPECT_LT(Constructor.evaluateHeuristics(), Function.evaluateHeuristics());
  EXPECT_LT(Destructor.evaluateHeuristics(), Constructor.evaluateHeuristics());
}

TEST(QualityTests, SymbolRelevanceSignalsSanity) {
  SymbolRelevanceSignals Default;
  EXPECT_EQ(Default.evaluateHeuristics(), 1);

  SymbolRelevanceSignals Forbidden;
  Forbidden.Forbidden = true;
  EXPECT_LT(Forbidden.evaluateHeuristics(), Default.evaluateHeuristics());

  SymbolRelevanceSignals PoorNameMatch;
  PoorNameMatch.NameMatch = 0.2f;
  EXPECT_LT(PoorNameMatch.evaluateHeuristics(), Default.evaluateHeuristics());

  SymbolRelevanceSignals WithSemaFileProximity;
  WithSemaFileProximity.SemaFileProximityScore = 0.2f;
  EXPECT_GT(WithSemaFileProximity.evaluateHeuristics(),
            Default.evaluateHeuristics());

  ScopeDistance ScopeProximity({"x::y::"});

  SymbolRelevanceSignals WithSemaScopeProximity;
  WithSemaScopeProximity.ScopeProximityMatch = &ScopeProximity;
  WithSemaScopeProximity.SemaSaysInScope = true;
  EXPECT_GT(WithSemaScopeProximity.evaluateHeuristics(),
            Default.evaluateHeuristics());

  SymbolRelevanceSignals WithIndexScopeProximity;
  WithIndexScopeProximity.ScopeProximityMatch = &ScopeProximity;
  WithIndexScopeProximity.SymbolScope = "x::";
  EXPECT_GT(WithSemaScopeProximity.evaluateHeuristics(),
            Default.evaluateHeuristics());

  SymbolRelevanceSignals IndexProximate;
  IndexProximate.SymbolURI = "unittest:/foo/bar.h";
  llvm::StringMap<SourceParams> ProxSources;
  ProxSources.try_emplace(testPath("foo/baz.h"));
  URIDistance Distance(ProxSources);
  IndexProximate.FileProximityMatch = &Distance;
  EXPECT_GT(IndexProximate.evaluateHeuristics(), Default.evaluateHeuristics());
  SymbolRelevanceSignals IndexDistant = IndexProximate;
  IndexDistant.SymbolURI = "unittest:/elsewhere/path.h";
  EXPECT_GT(IndexProximate.evaluateHeuristics(),
            IndexDistant.evaluateHeuristics())
      << IndexProximate << IndexDistant;
  EXPECT_GT(IndexDistant.evaluateHeuristics(), Default.evaluateHeuristics());

  SymbolRelevanceSignals Scoped;
  Scoped.Scope = SymbolRelevanceSignals::FileScope;
  EXPECT_LT(Scoped.evaluateHeuristics(), Default.evaluateHeuristics());
  Scoped.Query = SymbolRelevanceSignals::CodeComplete;
  EXPECT_GT(Scoped.evaluateHeuristics(), Default.evaluateHeuristics());

  SymbolRelevanceSignals Instance;
  Instance.IsInstanceMember = false;
  EXPECT_EQ(Instance.evaluateHeuristics(), Default.evaluateHeuristics());
  Instance.Context = CodeCompletionContext::CCC_DotMemberAccess;
  EXPECT_LT(Instance.evaluateHeuristics(), Default.evaluateHeuristics());
  Instance.IsInstanceMember = true;
  EXPECT_EQ(Instance.evaluateHeuristics(), Default.evaluateHeuristics());

  SymbolRelevanceSignals InBaseClass;
  InBaseClass.InBaseClass = true;
  EXPECT_LT(InBaseClass.evaluateHeuristics(), Default.evaluateHeuristics());

  llvm::StringSet<> Words = {"one", "two", "three"};
  SymbolRelevanceSignals WithoutMatchingWord;
  WithoutMatchingWord.ContextWords = &Words;
  WithoutMatchingWord.Name = "four";
  EXPECT_EQ(WithoutMatchingWord.evaluateHeuristics(),
            Default.evaluateHeuristics());
  SymbolRelevanceSignals WithMatchingWord;
  WithMatchingWord.ContextWords = &Words;
  WithMatchingWord.Name = "TheTwoTowers";
  EXPECT_GT(WithMatchingWord.evaluateHeuristics(),
            Default.evaluateHeuristics());
}

TEST(QualityTests, ScopeProximity) {
  SymbolRelevanceSignals Relevance;
  ScopeDistance ScopeProximity({"x::y::z::", "x::", "llvm::", ""});
  Relevance.ScopeProximityMatch = &ScopeProximity;

  Relevance.SymbolScope = "other::";
  float NotMatched = Relevance.evaluateHeuristics();

  Relevance.SymbolScope = "";
  float Global = Relevance.evaluateHeuristics();
  EXPECT_GT(Global, NotMatched);

  Relevance.SymbolScope = "llvm::";
  float NonParent = Relevance.evaluateHeuristics();
  EXPECT_GT(NonParent, Global);

  Relevance.SymbolScope = "x::";
  float GrandParent = Relevance.evaluateHeuristics();
  EXPECT_GT(GrandParent, Global);

  Relevance.SymbolScope = "x::y::";
  float Parent = Relevance.evaluateHeuristics();
  EXPECT_GT(Parent, GrandParent);

  Relevance.SymbolScope = "x::y::z::";
  float Enclosing = Relevance.evaluateHeuristics();
  EXPECT_GT(Enclosing, Parent);
}

TEST(QualityTests, SortText) {
  EXPECT_LT(sortText(std::numeric_limits<float>::infinity()),
            sortText(1000.2f));
  EXPECT_LT(sortText(1000.2f), sortText(1));
  EXPECT_LT(sortText(1), sortText(0.3f));
  EXPECT_LT(sortText(0.3f), sortText(0));
  EXPECT_LT(sortText(0), sortText(-10));
  EXPECT_LT(sortText(-10), sortText(-std::numeric_limits<float>::infinity()));

  EXPECT_LT(sortText(1, "z"), sortText(0, "a"));
  EXPECT_LT(sortText(0, "a"), sortText(0, "z"));
}

TEST(QualityTests, NoBoostForClassConstructor) {
  auto Header = TestTU::withHeaderCode(R"cpp(
    class Foo {
    public:
      Foo(int);
    };
  )cpp");
  auto Symbols = Header.headerSymbols();
  auto AST = Header.build();

  const NamedDecl *Foo = &findDecl(AST, "Foo");
  SymbolRelevanceSignals Cls;
  Cls.merge(CodeCompletionResult(Foo, /*Priority=*/0));

  const NamedDecl *CtorDecl = &findDecl(AST, [](const NamedDecl &ND) {
    return (ND.getQualifiedNameAsString() == "Foo::Foo") &&
           isa<CXXConstructorDecl>(&ND);
  });
  SymbolRelevanceSignals Ctor;
  Ctor.merge(CodeCompletionResult(CtorDecl, /*Priority=*/0));

  EXPECT_EQ(Cls.Scope, SymbolRelevanceSignals::GlobalScope);
  EXPECT_EQ(Ctor.Scope, SymbolRelevanceSignals::GlobalScope);
}

TEST(QualityTests, IsInstanceMember) {
  auto Header = TestTU::withHeaderCode(R"cpp(
    class Foo {
    public:
      static void foo() {}

      template <typename T> void tpl(T *t) {}

      void bar() {}
    };
  )cpp");
  auto Symbols = Header.headerSymbols();

  SymbolRelevanceSignals Rel;
  const Symbol &FooSym = findSymbol(Symbols, "Foo::foo");
  Rel.merge(FooSym);
  EXPECT_FALSE(Rel.IsInstanceMember);
  const Symbol &BarSym = findSymbol(Symbols, "Foo::bar");
  Rel.merge(BarSym);
  EXPECT_TRUE(Rel.IsInstanceMember);

  Rel.IsInstanceMember = false;
  const Symbol &TplSym = findSymbol(Symbols, "Foo::tpl");
  Rel.merge(TplSym);
  EXPECT_TRUE(Rel.IsInstanceMember);

  auto AST = Header.build();
  const NamedDecl *Foo = &findDecl(AST, "Foo::foo");
  const NamedDecl *Bar = &findDecl(AST, "Foo::bar");
  const NamedDecl *Tpl = &findDecl(AST, "Foo::tpl");

  Rel.IsInstanceMember = false;
  Rel.merge(CodeCompletionResult(Foo, /*Priority=*/0));
  EXPECT_FALSE(Rel.IsInstanceMember);
  Rel.merge(CodeCompletionResult(Bar, /*Priority=*/0));
  EXPECT_TRUE(Rel.IsInstanceMember);
  Rel.IsInstanceMember = false;
  Rel.merge(CodeCompletionResult(Tpl, /*Priority=*/0));
  EXPECT_TRUE(Rel.IsInstanceMember);
}

TEST(QualityTests, ConstructorDestructor) {
  auto Header = TestTU::withHeaderCode(R"cpp(
    class Foo {
    public:
      Foo(int);
      ~Foo();
    };
  )cpp");
  auto Symbols = Header.headerSymbols();
  auto AST = Header.build();

  const NamedDecl *CtorDecl = &findDecl(AST, [](const NamedDecl &ND) {
    return (ND.getQualifiedNameAsString() == "Foo::Foo") &&
           isa<CXXConstructorDecl>(&ND);
  });
  const NamedDecl *DtorDecl = &findDecl(AST, [](const NamedDecl &ND) {
    return (ND.getQualifiedNameAsString() == "Foo::~Foo") &&
           isa<CXXDestructorDecl>(&ND);
  });

  SymbolQualitySignals CtorQ;
  CtorQ.merge(CodeCompletionResult(CtorDecl, /*Priority=*/0));
  EXPECT_EQ(CtorQ.Category, SymbolQualitySignals::Constructor);

  CtorQ.Category = SymbolQualitySignals::Unknown;
  const Symbol &CtorSym = findSymbol(Symbols, "Foo::Foo");
  CtorQ.merge(CtorSym);
  EXPECT_EQ(CtorQ.Category, SymbolQualitySignals::Constructor);

  SymbolQualitySignals DtorQ;
  DtorQ.merge(CodeCompletionResult(DtorDecl, /*Priority=*/0));
  EXPECT_EQ(DtorQ.Category, SymbolQualitySignals::Destructor);
}

TEST(QualityTests, Operator) {
  auto Header = TestTU::withHeaderCode(R"cpp(
    class Foo {
    public:
      bool operator<(const Foo& f1);
    };
  )cpp");
  auto AST = Header.build();

  const NamedDecl *Operator = &findDecl(AST, [](const NamedDecl &ND) {
    if (const auto *OD = dyn_cast<FunctionDecl>(&ND))
      if (OD->isOverloadedOperator())
        return true;
    return false;
  });
  SymbolQualitySignals Q;
  Q.merge(CodeCompletionResult(Operator, /*Priority=*/0));
  EXPECT_EQ(Q.Category, SymbolQualitySignals::Operator);
}

TEST(QualityTests, ItemWithFixItsRankedDown) {
  CodeCompleteOptions Opts;
  Opts.IncludeFixIts = true;

  auto Header = TestTU::withHeaderCode(R"cpp(
        int x;
      )cpp");
  auto AST = Header.build();

  SymbolRelevanceSignals RelevanceWithFixIt;
  RelevanceWithFixIt.merge(CodeCompletionResult(
      &findDecl(AST, "x"), /*Priority=*/0, /*Qualifier=*/std::nullopt,
      /*QualifierIsInformative=*/false, /*Accessible=*/true, {FixItHint{}}));
  EXPECT_TRUE(RelevanceWithFixIt.NeedsFixIts);

  SymbolRelevanceSignals RelevanceWithoutFixIt;
  RelevanceWithoutFixIt.merge(CodeCompletionResult(
      &findDecl(AST, "x"), /*Priority=*/0, /*Qualifier=*/std::nullopt,
      /*QualifierIsInformative=*/false, /*Accessible=*/true, {}));
  EXPECT_FALSE(RelevanceWithoutFixIt.NeedsFixIts);

  EXPECT_LT(RelevanceWithFixIt.evaluateHeuristics(),
            RelevanceWithoutFixIt.evaluateHeuristics());
}

} // namespace
} // namespace clangd
} // namespace clang
