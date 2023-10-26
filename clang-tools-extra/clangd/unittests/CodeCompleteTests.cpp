//===-- CodeCompleteTests.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTSignals.h"
#include "Annotations.h"
#include "ClangdServer.h"
#include "CodeComplete.h"
#include "Compiler.h"
#include "Feature.h"
#include "Matchers.h"
#include "Protocol.h"
#include "Quality.h"
#include "SourceCode.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/SymbolOrigin.h"
#include "support/Threading.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

namespace clang {
namespace clangd {

namespace {
using ::llvm::Failed;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::UnorderedElementsAre;
using ContextKind = CodeCompletionContext::Kind;

// GMock helpers for matching completion items.
MATCHER_P(named, Name, "") { return arg.Name == Name; }
MATCHER_P(mainFileRefs, Refs, "") { return arg.MainFileRefs == Refs; }
MATCHER_P(scopeRefs, Refs, "") { return arg.ScopeRefsInFile == Refs; }
MATCHER_P(nameStartsWith, Prefix, "") {
  return llvm::StringRef(arg.Name).startswith(Prefix);
}
MATCHER_P(filterText, F, "") { return arg.FilterText == F; }
MATCHER_P(scope, S, "") { return arg.Scope == S; }
MATCHER_P(qualifier, Q, "") { return arg.RequiredQualifier == Q; }
MATCHER_P(labeled, Label, "") {
  return arg.RequiredQualifier + arg.Name + arg.Signature == Label;
}
MATCHER_P(sigHelpLabeled, Label, "") { return arg.label == Label; }
MATCHER_P(kind, K, "") { return arg.Kind == K; }
MATCHER_P(doc, D, "") {
  return arg.Documentation && arg.Documentation->asPlainText() == D;
}
MATCHER_P(returnType, D, "") { return arg.ReturnType == D; }
MATCHER_P(hasInclude, IncludeHeader, "") {
  return !arg.Includes.empty() && arg.Includes[0].Header == IncludeHeader;
}
MATCHER_P(insertInclude, IncludeHeader, "") {
  return !arg.Includes.empty() && arg.Includes[0].Header == IncludeHeader &&
         bool(arg.Includes[0].Insertion);
}
MATCHER_P(insertIncludeText, InsertedText, "") {
  return !arg.Includes.empty() && arg.Includes[0].Insertion &&
         arg.Includes[0].Insertion->newText == InsertedText;
}
MATCHER(insertInclude, "") {
  return !arg.Includes.empty() && bool(arg.Includes[0].Insertion);
}
MATCHER_P(snippetSuffix, Text, "") { return arg.SnippetSuffix == Text; }
MATCHER_P(origin, OriginSet, "") { return arg.Origin == OriginSet; }
MATCHER_P(signature, S, "") { return arg.Signature == S; }
MATCHER_P(replacesRange, Range, "") {
  return arg.CompletionTokenRange == Range;
}

// Shorthand for Contains(named(Name)).
Matcher<const std::vector<CodeCompletion> &> has(std::string Name) {
  return Contains(named(std::move(Name)));
}
Matcher<const std::vector<CodeCompletion> &> has(std::string Name,
                                                 CompletionItemKind K) {
  return Contains(AllOf(named(std::move(Name)), kind(K)));
}
MATCHER(isDocumented, "") { return arg.Documentation.has_value(); }
MATCHER(deprecated, "") { return arg.Deprecated; }

std::unique_ptr<SymbolIndex> memIndex(std::vector<Symbol> Symbols) {
  SymbolSlab::Builder Slab;
  for (const auto &Sym : Symbols)
    Slab.insert(Sym);
  return MemIndex::build(std::move(Slab).build(), RefSlab(), RelationSlab());
}

// Runs code completion.
// If IndexSymbols is non-empty, an index will be built and passed to opts.
CodeCompleteResult completions(const TestTU &TU, Position Point,
                               std::vector<Symbol> IndexSymbols = {},
                               clangd::CodeCompleteOptions Opts = {}) {
  std::unique_ptr<SymbolIndex> OverrideIndex;
  if (!IndexSymbols.empty()) {
    assert(!Opts.Index && "both Index and IndexSymbols given!");
    OverrideIndex = memIndex(std::move(IndexSymbols));
    Opts.Index = OverrideIndex.get();
  }

  MockFS FS;
  auto Inputs = TU.inputs(FS);
  IgnoreDiagnostics Diags;
  auto CI = buildCompilerInvocation(Inputs, Diags);
  if (!CI) {
    ADD_FAILURE() << "Couldn't build CompilerInvocation";
    return {};
  }
  auto Preamble = buildPreamble(testPath(TU.Filename), *CI, Inputs,
                                /*InMemory=*/true, /*Callback=*/nullptr);
  return codeComplete(testPath(TU.Filename), Point, Preamble.get(), Inputs,
                      Opts);
}

// Runs code completion.
CodeCompleteResult completions(llvm::StringRef Text,
                               std::vector<Symbol> IndexSymbols = {},
                               clangd::CodeCompleteOptions Opts = {},
                               PathRef FilePath = "foo.cpp") {
  Annotations Test(Text);
  auto TU = TestTU::withCode(Test.code());
  // To make sure our tests for completiopns inside templates work on Windows.
  TU.Filename = FilePath.str();
  return completions(TU, Test.point(), std::move(IndexSymbols),
                     std::move(Opts));
}

// Runs code completion without the clang parser.
CodeCompleteResult completionsNoCompile(llvm::StringRef Text,
                                        std::vector<Symbol> IndexSymbols = {},
                                        clangd::CodeCompleteOptions Opts = {},
                                        PathRef FilePath = "foo.cpp") {
  std::unique_ptr<SymbolIndex> OverrideIndex;
  if (!IndexSymbols.empty()) {
    assert(!Opts.Index && "both Index and IndexSymbols given!");
    OverrideIndex = memIndex(std::move(IndexSymbols));
    Opts.Index = OverrideIndex.get();
  }

  MockFS FS;
  Annotations Test(Text);
  ParseInputs ParseInput{tooling::CompileCommand(), &FS, Test.code().str()};
  return codeComplete(FilePath, Test.point(), /*Preamble=*/nullptr, ParseInput,
                      Opts);
}

Symbol withReferences(int N, Symbol S) {
  S.References = N;
  return S;
}

#if CLANGD_DECISION_FOREST
TEST(DecisionForestRankingModel, NameMatchSanityTest) {
  clangd::CodeCompleteOptions Opts;
  Opts.RankingModel = CodeCompleteOptions::DecisionForest;
  auto Results = completions(
      R"cpp(
struct MemberAccess {
  int ABG();
  int AlphaBetaGamma();
};
int func() { MemberAccess().ABG^ }
)cpp",
      /*IndexSymbols=*/{}, Opts);
  EXPECT_THAT(Results.Completions,
              ElementsAre(named("ABG"), named("AlphaBetaGamma")));
}

TEST(DecisionForestRankingModel, ReferencesAffectRanking) {
  clangd::CodeCompleteOptions Opts;
  Opts.RankingModel = CodeCompleteOptions::DecisionForest;
  constexpr int NumReferences = 100000;
  EXPECT_THAT(
      completions("int main() { clang^ }",
                  {ns("clangA"), withReferences(NumReferences, func("clangD"))},
                  Opts)
          .Completions,
      ElementsAre(named("clangD"), named("clangA")));
  EXPECT_THAT(
      completions("int main() { clang^ }",
                  {withReferences(NumReferences, ns("clangA")), func("clangD")},
                  Opts)
          .Completions,
      ElementsAre(named("clangA"), named("clangD")));
}
#endif // CLANGD_DECISION_FOREST

TEST(DecisionForestRankingModel, DecisionForestScorerCallbackTest) {
  clangd::CodeCompleteOptions Opts;
  constexpr float MagicNumber = 1234.5678f;
  Opts.RankingModel = CodeCompleteOptions::DecisionForest;
  Opts.DecisionForestScorer = [&](const SymbolQualitySignals &,
                                  const SymbolRelevanceSignals &, float Base) {
    DecisionForestScores Scores;
    Scores.Total = MagicNumber;
    Scores.ExcludingName = MagicNumber;
    return Scores;
  };
  llvm::StringRef Code = "int func() { int xyz; xy^ }";
  auto Results = completions(Code,
                             /*IndexSymbols=*/{}, Opts);
  ASSERT_EQ(Results.Completions.size(), 1u);
  EXPECT_EQ(Results.Completions[0].Score.Total, MagicNumber);
  EXPECT_EQ(Results.Completions[0].Score.ExcludingName, MagicNumber);

  // Do not use DecisionForestScorer for heuristics model.
  Opts.RankingModel = CodeCompleteOptions::Heuristics;
  Results = completions(Code,
                        /*IndexSymbols=*/{}, Opts);
  ASSERT_EQ(Results.Completions.size(), 1u);
  EXPECT_NE(Results.Completions[0].Score.Total, MagicNumber);
  EXPECT_NE(Results.Completions[0].Score.ExcludingName, MagicNumber);
}

TEST(CompletionTest, Limit) {
  clangd::CodeCompleteOptions Opts;
  Opts.Limit = 2;
  auto Results = completions(R"cpp(
struct ClassWithMembers {
  int AAA();
  int BBB();
  int CCC();
};

int main() { ClassWithMembers().^ }
      )cpp",
                             /*IndexSymbols=*/{}, Opts);

  EXPECT_TRUE(Results.HasMore);
  EXPECT_THAT(Results.Completions, ElementsAre(named("AAA"), named("BBB")));
}

TEST(CompletionTest, Filter) {
  std::string Body = R"cpp(
    #define MotorCar
    int Car;
    struct S {
      int FooBar;
      int FooBaz;
      int Qux;
    };
  )cpp";

  // Only items matching the fuzzy query are returned.
  EXPECT_THAT(completions(Body + "int main() { S().Foba^ }").Completions,
              AllOf(has("FooBar"), has("FooBaz"), Not(has("Qux"))));

  // Macros require prefix match, either from index or AST.
  Symbol Sym = var("MotorCarIndex");
  Sym.SymInfo.Kind = index::SymbolKind::Macro;
  EXPECT_THAT(
      completions(Body + "int main() { C^ }", {Sym}).Completions,
      AllOf(has("Car"), Not(has("MotorCar")), Not(has("MotorCarIndex"))));
  EXPECT_THAT(completions(Body + "int main() { M^ }", {Sym}).Completions,
              AllOf(has("MotorCar"), has("MotorCarIndex")));
}

void testAfterDotCompletion(clangd::CodeCompleteOptions Opts) {
  auto Results = completions(
      R"cpp(
      int global_var;

      int global_func();

      // Make sure this is not in preamble.
      #define MACRO X

      struct GlobalClass {};

      struct ClassWithMembers {
        /// doc for method.
        int method();

        int field;
      private:
        int private_field;
      };

      int test() {
        struct LocalClass {};

        /// doc for local_var.
        int local_var;

        ClassWithMembers().^
      }
      )cpp",
      {cls("IndexClass"), var("index_var"), func("index_func")}, Opts);

  EXPECT_TRUE(Results.RanParser);
  // Class members. The only items that must be present in after-dot
  // completion.
  EXPECT_THAT(Results.Completions,
              AllOf(has("method"), has("field"), Not(has("ClassWithMembers")),
                    Not(has("operator=")), Not(has("~ClassWithMembers"))));
  EXPECT_IFF(Opts.IncludeIneligibleResults, Results.Completions,
             has("private_field"));
  // Global items.
  EXPECT_THAT(
      Results.Completions,
      Not(AnyOf(has("global_var"), has("index_var"), has("global_func"),
                has("global_func()"), has("index_func"), has("GlobalClass"),
                has("IndexClass"), has("MACRO"), has("LocalClass"))));
  // There should be no code patterns (aka snippets) in after-dot
  // completion. At least there aren't any we're aware of.
  EXPECT_THAT(Results.Completions,
              Not(Contains(kind(CompletionItemKind::Snippet))));
  // Check documentation.
  EXPECT_THAT(Results.Completions, Contains(isDocumented()));
}

void testGlobalScopeCompletion(clangd::CodeCompleteOptions Opts) {
  auto Results = completions(
      R"cpp(
      int global_var;
      int global_func();

      // Make sure this is not in preamble.
      #define MACRO X

      struct GlobalClass {};

      struct ClassWithMembers {
        /// doc for method.
        int method();
      };

      int test() {
        struct LocalClass {};

        /// doc for local_var.
        int local_var;

        ^
      }
      )cpp",
      {cls("IndexClass"), var("index_var"), func("index_func")}, Opts);

  EXPECT_TRUE(Results.RanParser);
  // Class members. Should never be present in global completions.
  EXPECT_THAT(Results.Completions,
              Not(AnyOf(has("method"), has("method()"), has("field"))));
  // Global items.
  EXPECT_THAT(Results.Completions,
              AllOf(has("global_var"), has("index_var"), has("global_func"),
                    has("index_func" /* our fake symbol doesn't include () */),
                    has("GlobalClass"), has("IndexClass")));
  // A macro.
  EXPECT_THAT(Results.Completions, has("MACRO"));
  // Local items. Must be present always.
  EXPECT_THAT(Results.Completions,
              AllOf(has("local_var"), has("LocalClass"),
                    Contains(kind(CompletionItemKind::Snippet))));
  // Check documentation.
  EXPECT_THAT(Results.Completions, Contains(isDocumented()));
}

TEST(CompletionTest, CompletionOptions) {
  auto Test = [&](const clangd::CodeCompleteOptions &Opts) {
    testAfterDotCompletion(Opts);
    testGlobalScopeCompletion(Opts);
  };
  // We used to test every combination of options, but that got too slow (2^N).
  auto Flags = {
      &clangd::CodeCompleteOptions::IncludeIneligibleResults,
  };
  // Test default options.
  Test({});
  // Test with one flag flipped.
  for (auto &F : Flags) {
    clangd::CodeCompleteOptions O;
    O.*F ^= true;
    Test(O);
  }
}

TEST(CompletionTest, Accessible) {
  auto Internal = completions(R"cpp(
      class Foo {
        public: void pub();
        protected: void prot();
        private: void priv();
      };
      void Foo::pub() { this->^ }
  )cpp");
  EXPECT_THAT(Internal.Completions,
              AllOf(has("priv"), has("prot"), has("pub")));

  auto External = completions(R"cpp(
      class Foo {
        public: void pub();
        protected: void prot();
        private: void priv();
      };
      void test() {
        Foo F;
        F.^
      }
  )cpp");
  EXPECT_THAT(External.Completions,
              AllOf(has("pub"), Not(has("prot")), Not(has("priv"))));

  auto Results = completions(R"cpp(
      struct Foo {
        public: void pub();
        protected: void prot();
        private: void priv();
      };
      struct Bar : public Foo {
        private: using Foo::pub;
      };
      void test() {
        Bar B;
        B.^
      }
  )cpp");
  EXPECT_THAT(Results.Completions,
              AllOf(Not(has("priv")), Not(has("prot")), Not(has("pub"))));
}

TEST(CompletionTest, Qualifiers) {
  auto Results = completions(R"cpp(
      class Foo {
        public: int foo() const;
        int bar() const;
      };
      class Bar : public Foo {
        int foo() const;
      };
      void test() { Bar().^ }
  )cpp");
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(qualifier(""), named("bar"))));
  // Hidden members are not shown.
  EXPECT_THAT(Results.Completions,
              Not(Contains(AllOf(qualifier("Foo::"), named("foo")))));
  // Private members are not shown.
  EXPECT_THAT(Results.Completions,
              Not(Contains(AllOf(qualifier(""), named("foo")))));
}

// https://github.com/clangd/clangd/issues/1451
TEST(CompletionTest, QualificationWithInlineNamespace) {
  auto Results = completions(R"cpp(
    namespace a { inline namespace b {} }
    using namespace a::b;
    void f() { Foo^ }
  )cpp",
                             {cls("a::Foo")});
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(qualifier("a::"), named("Foo"))));
}

TEST(CompletionTest, InjectedTypename) {
  // These are suppressed when accessed as a member...
  EXPECT_THAT(completions("struct X{}; void foo(){ X().^ }").Completions,
              Not(has("X")));
  EXPECT_THAT(completions("struct X{ void foo(){ this->^ } };").Completions,
              Not(has("X")));
  // ...but accessible in other, more useful cases.
  EXPECT_THAT(completions("struct X{ void foo(){ ^ } };").Completions,
              has("X"));
  EXPECT_THAT(
      completions("struct Y{}; struct X:Y{ void foo(){ ^ } };").Completions,
      has("Y"));
  EXPECT_THAT(
      completions(
          "template<class> struct Y{}; struct X:Y<int>{ void foo(){ ^ } };")
          .Completions,
      has("Y"));
  // This case is marginal (`using X::X` is useful), we allow it for now.
  EXPECT_THAT(completions("struct X{}; void foo(){ X::^ }").Completions,
              has("X"));
}

TEST(CompletionTest, SkipInjectedWhenUnqualified) {
  EXPECT_THAT(completions("struct X { void f() { X^ }};").Completions,
              ElementsAre(named("X"), named("~X")));
}

TEST(CompletionTest, Snippets) {
  clangd::CodeCompleteOptions Opts;
  auto Results = completions(
      R"cpp(
      struct fake {
        int a;
        int f(int i, const float f) const;
      };
      int main() {
        fake f;
        f.^
      }
      )cpp",
      /*IndexSymbols=*/{}, Opts);
  EXPECT_THAT(
      Results.Completions,
      HasSubsequence(named("a"),
                     snippetSuffix("(${1:int i}, ${2:const float f})")));
}

TEST(CompletionTest, HeuristicsForMemberFunctionCompletion) {
  clangd::CodeCompleteOptions Opts;
  Opts.EnableSnippets = true;

  Annotations Code(R"cpp(
      struct Foo {
        static int staticMethod(int);
        int method(int) const;
        template <typename T, typename U, typename V = int>
        T generic(U, V);
        template <typename T, int U>
        static T staticGeneric();
        Foo() {
          this->$canBeCall^
          $canBeCall^
          Foo::$canBeCall^
        }
      };

      struct Derived : Foo {
        using Foo::method;
        using Foo::generic;
        Derived() {
          Foo::$canBeCall^
        }
      };

      struct OtherClass {
        OtherClass() {
          Foo f;
          Derived d;
          f.$canBeCall^
          ; // Prevent parsing as 'f.f'
          f.Foo::$canBeCall^
          &Foo::$canNotBeCall^
          ;
          d.Foo::$canBeCall^
          ;
          d.Derived::$canBeCall^
        }
      };

      int main() {
        Foo f;
        Derived d;
        f.$canBeCall^
        ; // Prevent parsing as 'f.f'
        f.Foo::$canBeCall^
        &Foo::$canNotBeCall^
        ;
        d.Foo::$canBeCall^
        ;
        d.Derived::$canBeCall^
      }
      )cpp");
  auto TU = TestTU::withCode(Code.code());

  for (const auto &P : Code.points("canNotBeCall")) {
    auto Results = completions(TU, P, /*IndexSymbols*/ {}, Opts);
    EXPECT_THAT(Results.Completions,
                Contains(AllOf(named("method"), signature("(int) const"),
                               snippetSuffix(""))));
    // We don't have any arguments to deduce against if this isn't a call.
    // Thus, we should emit these deducible template arguments explicitly.
    EXPECT_THAT(
        Results.Completions,
        Contains(AllOf(named("generic"),
                       signature("<typename T, typename U>(U, V)"),
                       snippetSuffix("<${1:typename T}, ${2:typename U}>"))));
  }

  for (const auto &P : Code.points("canBeCall")) {
    auto Results = completions(TU, P, /*IndexSymbols*/ {}, Opts);
    EXPECT_THAT(Results.Completions,
                Contains(AllOf(named("method"), signature("(int) const"),
                               snippetSuffix("(${1:int})"))));
    EXPECT_THAT(
        Results.Completions,
        Contains(AllOf(named("generic"), signature("<typename T>(U, V)"),
                       snippetSuffix("<${1:typename T}>(${2:U}, ${3:V})"))));
  }

  // static method will always keep the snippet
  for (const auto &P : Code.points()) {
    auto Results = completions(TU, P, /*IndexSymbols*/ {}, Opts);
    EXPECT_THAT(Results.Completions,
                Contains(AllOf(named("staticMethod"), signature("(int)"),
                               snippetSuffix("(${1:int})"))));
    EXPECT_THAT(Results.Completions,
                Contains(AllOf(
                    named("staticGeneric"), signature("<typename T, int U>()"),
                    snippetSuffix("<${1:typename T}, ${2:int U}>()"))));
  }
}

TEST(CompletionTest, NoSnippetsInUsings) {
  clangd::CodeCompleteOptions Opts;
  Opts.EnableSnippets = true;
  auto Results = completions(
      R"cpp(
      namespace ns {
        int func(int a, int b);
      }

      using ns::^;
      )cpp",
      /*IndexSymbols=*/{}, Opts);
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(named("func"), labeled("func(int a, int b)"),
                                snippetSuffix(""))));

  // Check index completions too.
  auto Func = func("ns::func");
  Func.CompletionSnippetSuffix = "(${1:int a}, ${2: int b})";
  Func.Signature = "(int a, int b)";
  Func.ReturnType = "void";

  Results = completions(R"cpp(
      namespace ns {}
      using ns::^;
  )cpp",
                        /*IndexSymbols=*/{Func}, Opts);
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(named("func"), labeled("func(int a, int b)"),
                                snippetSuffix(""))));

  // Check all-scopes completions too.
  Opts.AllScopes = true;
  Results = completions(R"cpp(
      using ^;
  )cpp",
                        /*IndexSymbols=*/{Func}, Opts);
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(named("func"), labeled("ns::func(int a, int b)"),
                             snippetSuffix(""))));
}

TEST(CompletionTest, Kinds) {
  auto Results = completions(
      R"cpp(
          int variable;
          struct Struct {};
          int function();
          // make sure MACRO is not included in preamble.
          #define MACRO 10
          int X = ^
      )cpp",
      {func("indexFunction"), var("indexVariable"), cls("indexClass")});
  EXPECT_THAT(Results.Completions,
              AllOf(has("function", CompletionItemKind::Function),
                    has("variable", CompletionItemKind::Variable),
                    has("int", CompletionItemKind::Keyword),
                    has("Struct", CompletionItemKind::Struct),
                    has("MACRO", CompletionItemKind::Constant),
                    has("indexFunction", CompletionItemKind::Function),
                    has("indexVariable", CompletionItemKind::Variable),
                    has("indexClass", CompletionItemKind::Class)));

  Results = completions("nam^");
  EXPECT_THAT(Results.Completions,
              has("namespace", CompletionItemKind::Snippet));

  // Members of anonymous unions are of kind 'field'.
  Results = completions(
      R"cpp(
        struct X{
            union {
              void *a;
            };
        };
        auto u = X().^
      )cpp");
  EXPECT_THAT(
      Results.Completions,
      UnorderedElementsAre(AllOf(named("a"), kind(CompletionItemKind::Field))));

  // Completion kinds for templates should not be unknown.
  Results = completions(
      R"cpp(
        template <class T> struct complete_class {};
        template <class T> void complete_function();
        template <class T> using complete_type_alias = int;
        template <class T> int complete_variable = 10;

        struct X {
          template <class T> static int complete_static_member = 10;

          static auto x = complete_^
        }
      )cpp");
  EXPECT_THAT(
      Results.Completions,
      UnorderedElementsAre(
          AllOf(named("complete_class"), kind(CompletionItemKind::Class)),
          AllOf(named("complete_function"), kind(CompletionItemKind::Function)),
          AllOf(named("complete_type_alias"),
                kind(CompletionItemKind::Interface)),
          AllOf(named("complete_variable"), kind(CompletionItemKind::Variable)),
          AllOf(named("complete_static_member"),
                kind(CompletionItemKind::Property))));

  Results = completions(
      R"cpp(
        enum Color {
          Red
        };
        Color u = ^
      )cpp");
  EXPECT_THAT(
      Results.Completions,
      Contains(AllOf(named("Red"), kind(CompletionItemKind::EnumMember))));
}

TEST(CompletionTest, NoDuplicates) {
  auto Results = completions(
      R"cpp(
          class Adapter {
          };

          void f() {
            Adapter^
          }
      )cpp",
      {cls("Adapter")});

  // Make sure there are no duplicate entries of 'Adapter'.
  EXPECT_THAT(Results.Completions, ElementsAre(named("Adapter")));
}

TEST(CompletionTest, ScopedNoIndex) {
  auto Results = completions(
      R"cpp(
          namespace fake { int BigBang, Babble, Box; };
          int main() { fake::ba^ }
      ")cpp");
  // Babble is a better match than BigBang. Box doesn't match at all.
  EXPECT_THAT(Results.Completions,
              ElementsAre(named("Babble"), named("BigBang")));
}

TEST(CompletionTest, Scoped) {
  auto Results = completions(
      R"cpp(
          namespace fake { int Babble, Box; };
          int main() { fake::ba^ }
      ")cpp",
      {var("fake::BigBang")});
  EXPECT_THAT(Results.Completions,
              ElementsAre(named("Babble"), named("BigBang")));
}

TEST(CompletionTest, ScopedWithFilter) {
  auto Results = completions(
      R"cpp(
          void f() { ns::x^ }
      )cpp",
      {cls("ns::XYZ"), func("ns::foo")});
  EXPECT_THAT(Results.Completions, UnorderedElementsAre(named("XYZ")));
}

TEST(CompletionTest, ReferencesAffectRanking) {
  EXPECT_THAT(completions("int main() { abs^ }", {func("absA"), func("absB")})
                  .Completions,
              HasSubsequence(named("absA"), named("absB")));
  EXPECT_THAT(completions("int main() { abs^ }",
                          {func("absA"), withReferences(1000, func("absB"))})
                  .Completions,
              HasSubsequence(named("absB"), named("absA")));
}

TEST(CompletionTest, ContextWords) {
  auto Results = completions(R"cpp(
  enum class Color { RED, YELLOW, BLUE };

  // (blank lines so the definition above isn't "context")

  // "It was a yellow car," he said. "Big yellow car, new."
  auto Finish = Color::^
  )cpp");
  // Yellow would normally sort last (alphabetic).
  // But the recent mention should bump it up.
  ASSERT_THAT(Results.Completions,
              HasSubsequence(named("YELLOW"), named("BLUE")));
}

TEST(CompletionTest, GlobalQualified) {
  auto Results = completions(
      R"cpp(
          void f() { ::^ }
      )cpp",
      {cls("XYZ")});
  EXPECT_THAT(Results.Completions,
              AllOf(has("XYZ", CompletionItemKind::Class),
                    has("f", CompletionItemKind::Function)));
}

TEST(CompletionTest, FullyQualified) {
  auto Results = completions(
      R"cpp(
          namespace ns { void bar(); }
          void f() { ::ns::^ }
      )cpp",
      {cls("ns::XYZ")});
  EXPECT_THAT(Results.Completions,
              AllOf(has("XYZ", CompletionItemKind::Class),
                    has("bar", CompletionItemKind::Function)));
}

TEST(CompletionTest, SemaIndexMerge) {
  auto Results = completions(
      R"cpp(
          namespace ns { int local; void both(); }
          void f() { ::ns::^ }
      )cpp",
      {func("ns::both"), cls("ns::Index")});
  // We get results from both index and sema, with no duplicates.
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(
                  AllOf(named("local"), origin(SymbolOrigin::AST)),
                  AllOf(named("Index"), origin(SymbolOrigin::Static)),
                  AllOf(named("both"),
                        origin(SymbolOrigin::AST | SymbolOrigin::Static))));
}

TEST(CompletionTest, SemaIndexMergeWithLimit) {
  clangd::CodeCompleteOptions Opts;
  Opts.Limit = 1;
  auto Results = completions(
      R"cpp(
          namespace ns { int local; void both(); }
          void f() { ::ns::^ }
      )cpp",
      {func("ns::both"), cls("ns::Index")}, Opts);
  EXPECT_EQ(Results.Completions.size(), Opts.Limit);
  EXPECT_TRUE(Results.HasMore);
}

TEST(CompletionTest, IncludeInsertionPreprocessorIntegrationTests) {
  TestTU TU;
  TU.ExtraArgs.push_back("-I" + testPath("sub"));
  TU.AdditionalFiles["sub/bar.h"] = "";
  auto BarURI = URI::create(testPath("sub/bar.h")).toString();

  Symbol Sym = cls("ns::X");
  Sym.CanonicalDeclaration.FileURI = BarURI.c_str();
  Sym.IncludeHeaders.emplace_back(BarURI, 1, Symbol::Include);
  // Shorten include path based on search directory and insert.
  Annotations Test("int main() { ns::^ }");
  TU.Code = Test.code().str();
  auto Results = completions(TU, Test.point(), {Sym});
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(named("X"), insertInclude("\"bar.h\""))));
  // Can be disabled via option.
  CodeCompleteOptions NoInsertion;
  NoInsertion.InsertIncludes = CodeCompleteOptions::NeverInsert;
  Results = completions(TU, Test.point(), {Sym}, NoInsertion);
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(named("X"), Not(insertInclude()))));
  // Duplicate based on inclusions in preamble.
  Test = Annotations(R"cpp(
          #include "sub/bar.h"  // not shortest, so should only match resolved.
          int main() { ns::^ }
      )cpp");
  TU.Code = Test.code().str();
  Results = completions(TU, Test.point(), {Sym});
  EXPECT_THAT(Results.Completions, ElementsAre(AllOf(named("X"), labeled("X"),
                                                     Not(insertInclude()))));
}

TEST(CompletionTest, NoIncludeInsertionWhenDeclFoundInFile) {
  Symbol SymX = cls("ns::X");
  Symbol SymY = cls("ns::Y");
  std::string BarHeader = testPath("bar.h");
  auto BarURI = URI::create(BarHeader).toString();
  SymX.CanonicalDeclaration.FileURI = BarURI.c_str();
  SymY.CanonicalDeclaration.FileURI = BarURI.c_str();
  SymX.IncludeHeaders.emplace_back("<bar>", 1, Symbol::Include);
  SymY.IncludeHeaders.emplace_back("<bar>", 1, Symbol::Include);
  // Shorten include path based on search directory and insert.
  auto Results = completions(R"cpp(
          namespace ns {
            class X;
            class Y {};
          }
          int main() { ns::^ }
      )cpp",
                             {SymX, SymY});
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(named("X"), Not(insertInclude())),
                          AllOf(named("Y"), Not(insertInclude()))));
}

TEST(CompletionTest, IndexSuppressesPreambleCompletions) {
  Annotations Test(R"cpp(
      #include "bar.h"
      namespace ns { int local; }
      void f() { ns::^; }
      void f2() { ns::preamble().$2^; }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.AdditionalFiles["bar.h"] =
      R"cpp(namespace ns { struct preamble { int member; }; })cpp";

  clangd::CodeCompleteOptions Opts = {};
  auto I = memIndex({var("ns::index")});
  Opts.Index = I.get();
  auto WithIndex = completions(TU, Test.point(), {}, Opts);
  EXPECT_THAT(WithIndex.Completions,
              UnorderedElementsAre(named("local"), named("index")));
  auto ClassFromPreamble = completions(TU, Test.point("2"), {}, Opts);
  EXPECT_THAT(ClassFromPreamble.Completions, Contains(named("member")));

  Opts.Index = nullptr;
  auto WithoutIndex = completions(TU, Test.point(), {}, Opts);
  EXPECT_THAT(WithoutIndex.Completions,
              UnorderedElementsAre(named("local"), named("preamble")));
}

// This verifies that we get normal preprocessor completions in the preamble.
// This is a regression test for an old bug: if we override the preamble and
// try to complete inside it, clang kicks our completion point just outside the
// preamble, resulting in always getting top-level completions.
TEST(CompletionTest, CompletionInPreamble) {
  auto Results = completions(R"cpp(
    #ifnd^ef FOO_H_
    #define BAR_H_
    #include <bar.h>
    int foo() {}
    #endif
    )cpp")
                     .Completions;
  EXPECT_THAT(Results, ElementsAre(named("ifndef")));
}

TEST(CompletionTest, CompletionRecoveryASTType) {
  auto Results = completions(R"cpp(
    struct S { int member; };
    S overloaded(int);
    void foo() {
      // No overload matches, but we have recovery-expr with the correct type.
      overloaded().^
    })cpp")
                     .Completions;
  EXPECT_THAT(Results, ElementsAre(named("member")));
}

TEST(CompletionTest, DynamicIndexIncludeInsertion) {
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer::Options Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true;
  ClangdServer Server(CDB, FS, Opts);

  FS.Files[testPath("foo_header.h")] = R"cpp(
    #pragma once
    struct Foo {
       // Member doc
       int foo();
    };
  )cpp";
  const std::string FileContent(R"cpp(
    #include "foo_header.h"
    int Foo::foo() {
      return 42;
    }
  )cpp");
  Server.addDocument(testPath("foo_impl.cpp"), FileContent);
  // Wait for the dynamic index being built.
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  auto File = testPath("foo.cpp");
  Annotations Test("Foo^ foo;");
  runAddDocument(Server, File, Test.code());
  auto CompletionList =
      llvm::cantFail(runCodeComplete(Server, File, Test.point(), {}));

  EXPECT_THAT(CompletionList.Completions,
              ElementsAre(AllOf(named("Foo"), hasInclude("\"foo_header.h\""),
                                insertInclude())));
}

TEST(CompletionTest, DynamicIndexMultiFile) {
  MockFS FS;
  MockCompilationDatabase CDB;
  auto Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true;
  ClangdServer Server(CDB, FS, Opts);

  FS.Files[testPath("foo.h")] = R"cpp(
      namespace ns { class XYZ {}; void foo(int x) {} }
  )cpp";
  runAddDocument(Server, testPath("foo.cpp"), R"cpp(
      #include "foo.h"
  )cpp");

  auto File = testPath("bar.cpp");
  Annotations Test(R"cpp(
      namespace ns {
      class XXX {};
      /// Doooc
      void fooooo() {}
      }
      void f() { ns::^ }
  )cpp");
  runAddDocument(Server, File, Test.code());

  auto Results = cantFail(runCodeComplete(Server, File, Test.point(), {}));
  // "XYZ" and "foo" are not included in the file being completed but are still
  // visible through the index.
  EXPECT_THAT(Results.Completions, has("XYZ", CompletionItemKind::Class));
  EXPECT_THAT(Results.Completions, has("foo", CompletionItemKind::Function));
  EXPECT_THAT(Results.Completions, has("XXX", CompletionItemKind::Class));
  EXPECT_THAT(Results.Completions,
              Contains((named("fooooo"), kind(CompletionItemKind::Function),
                        doc("Doooc"), returnType("void"))));
}

TEST(CompletionTest, Documentation) {
  auto Results = completions(
      R"cpp(
      // Non-doxygen comment.
      __attribute__((annotate("custom_annotation"))) int foo();
      /// Doxygen comment.
      /// \param int a
      int bar(int a);
      /* Multi-line
         block comment
      */
      int baz();

      int x = ^
     )cpp");
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(
                  named("foo"),
                  doc("Annotation: custom_annotation\nNon-doxygen comment."))));
  EXPECT_THAT(
      Results.Completions,
      Contains(AllOf(named("bar"), doc("Doxygen comment.\n\\param int a"))));
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(named("baz"), doc("Multi-line block comment"))));
}

TEST(CompletionTest, CommentsFromSystemHeaders) {
  MockFS FS;
  MockCompilationDatabase CDB;

  auto Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true;

  ClangdServer Server(CDB, FS, Opts);

  FS.Files[testPath("foo.h")] = R"cpp(
    #pragma GCC system_header

    // This comment should be retained!
    int foo();
  )cpp";

  auto File = testPath("foo.cpp");
  Annotations Test(R"cpp(
#include "foo.h"
int x = foo^
     )cpp");
  runAddDocument(Server, File, Test.code());
  auto CompletionList =
      llvm::cantFail(runCodeComplete(Server, File, Test.point(), {}));

  EXPECT_THAT(
      CompletionList.Completions,
      Contains(AllOf(named("foo"), doc("This comment should be retained!"))));
}

TEST(CompletionTest, GlobalCompletionFiltering) {

  Symbol Class = cls("XYZ");
  Class.Flags = static_cast<Symbol::SymbolFlag>(
      Class.Flags & ~(Symbol::IndexedForCodeCompletion));
  Symbol Func = func("XYZ::foooo");
  Func.Flags = static_cast<Symbol::SymbolFlag>(
      Func.Flags & ~(Symbol::IndexedForCodeCompletion));

  auto Results = completions(R"(//      void f() {
      XYZ::foooo^
      })",
                             {Class, Func});
  EXPECT_THAT(Results.Completions, IsEmpty());
}

TEST(CodeCompleteTest, DisableTypoCorrection) {
  auto Results = completions(R"cpp(
     namespace clang { int v; }
     void f() { clangd::^
  )cpp");
  EXPECT_TRUE(Results.Completions.empty());
}

TEST(CodeCompleteTest, NoColonColonAtTheEnd) {
  auto Results = completions(R"cpp(
    namespace clang { }
    void f() {
      clan^
    }
  )cpp");

  EXPECT_THAT(Results.Completions, Contains(labeled("clang")));
  EXPECT_THAT(Results.Completions, Not(Contains(labeled("clang::"))));
}

TEST(CompletionTests, EmptySnippetDoesNotCrash) {
    // See https://github.com/clangd/clangd/issues/1216
    auto Results = completions(R"cpp(
        int main() {
          auto w = [&](auto &&f) { return f(f); };
          auto f = w([&](auto &&f) {
            return [&](auto &&n) {
              if (n == 0) {
                return 1;
              }
              return n * ^(f)(n - 1);
            };
          })(10);
        }
    )cpp");
}

TEST(CompletionTest, Issue1427Crash) {
    // Need to provide main file signals to ensure that the branch in
    // SymbolRelevanceSignals::computeASTSignals() that tries to
    // compute a symbol ID is taken.
    ASTSignals MainFileSignals;
    CodeCompleteOptions Opts;
    Opts.MainFileSignals = &MainFileSignals;
    completions(R"cpp(
      auto f = []() {
        1.0_^
      };
    )cpp",
                {}, Opts);
}

TEST(CompletionTest, BacktrackCrashes) {
  // Sema calls code completion callbacks twice in these cases.
  auto Results = completions(R"cpp(
      namespace ns {
      struct FooBarBaz {};
      } // namespace ns

     int foo(ns::FooBar^
  )cpp");

  EXPECT_THAT(Results.Completions, ElementsAre(labeled("FooBarBaz")));

  // Check we don't crash in that case too.
  completions(R"cpp(
    struct FooBarBaz {};
    void test() {
      if (FooBarBaz * x^) {}
    }
)cpp");
}

TEST(CompletionTest, CompleteInMacroWithStringification) {
  auto Results = completions(R"cpp(
void f(const char *, int x);
#define F(x) f(#x, x)

namespace ns {
int X;
int Y;
}  // namespace ns

int f(int input_num) {
  F(ns::^)
}
)cpp");

  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(named("X"), named("Y")));
}

TEST(CompletionTest, CompleteInMacroAndNamespaceWithStringification) {
  auto Results = completions(R"cpp(
void f(const char *, int x);
#define F(x) f(#x, x)

namespace ns {
int X;

int f(int input_num) {
  F(^)
}
}  // namespace ns
)cpp");

  EXPECT_THAT(Results.Completions, Contains(named("X")));
}

TEST(CompletionTest, IgnoreCompleteInExcludedPPBranchWithRecoveryContext) {
  auto Results = completions(R"cpp(
    int bar(int param_in_bar) {
    }

    int foo(int param_in_foo) {
#if 0
  // In recovery mode, "param_in_foo" will also be suggested among many other
  // unrelated symbols; however, this is really a special case where this works.
  // If the #if block is outside of the function, "param_in_foo" is still
  // suggested, but "bar" and "foo" are missing. So the recovery mode doesn't
  // really provide useful results in excluded branches.
  par^
#endif
    }
)cpp");

  EXPECT_TRUE(Results.Completions.empty());
}

TEST(CompletionTest, DefaultArgs) {
  clangd::CodeCompleteOptions Opts;
  std::string Context = R"cpp(
    int X(int A = 0);
    int Y(int A, int B = 0);
    int Z(int A, int B = 0, int C = 0, int D = 0);
  )cpp";
  EXPECT_THAT(completions(Context + "int y = X^", {}, Opts).Completions,
              UnorderedElementsAre(labeled("X(int A = 0)")));
  EXPECT_THAT(completions(Context + "int y = Y^", {}, Opts).Completions,
              UnorderedElementsAre(AllOf(labeled("Y(int A, int B = 0)"),
                                         snippetSuffix("(${1:int A})"))));
  EXPECT_THAT(completions(Context + "int y = Z^", {}, Opts).Completions,
              UnorderedElementsAre(
                  AllOf(labeled("Z(int A, int B = 0, int C = 0, int D = 0)"),
                        snippetSuffix("(${1:int A})"))));
}

TEST(CompletionTest, NoCrashWithTemplateParamsAndPreferredTypes) {
  auto Completions = completions(R"cpp(
template <template <class> class TT> int foo() {
  int a = ^
}
)cpp")
                         .Completions;
  EXPECT_THAT(Completions, Contains(named("TT")));
}

TEST(CompletionTest, NestedTemplateHeuristics) {
  auto Completions = completions(R"cpp(
struct Plain { int xxx; };
template <typename T> class Templ { Plain ppp; };
template <typename T> void foo(Templ<T> &t) {
  // Formally ppp has DependentTy, because Templ may be specialized.
  // However we sholud be able to see into it using the primary template.
  t.ppp.^
}
)cpp")
                         .Completions;
  EXPECT_THAT(Completions, Contains(named("xxx")));
}

TEST(CompletionTest, RecordCCResultCallback) {
  std::vector<CodeCompletion> RecordedCompletions;
  CodeCompleteOptions Opts;
  Opts.RecordCCResult = [&RecordedCompletions](const CodeCompletion &CC,
                                               const SymbolQualitySignals &,
                                               const SymbolRelevanceSignals &,
                                               float Score) {
    RecordedCompletions.push_back(CC);
  };

  completions("int xy1, xy2; int a = xy^", /*IndexSymbols=*/{}, Opts);
  EXPECT_THAT(RecordedCompletions,
              UnorderedElementsAre(named("xy1"), named("xy2")));
}

TEST(CompletionTest, ASTSignals) {
  struct Completion {
    std::string Name;
    unsigned MainFileRefs;
    unsigned ScopeRefsInFile;
  };
  CodeCompleteOptions Opts;
  std::vector<Completion> RecordedCompletions;
  Opts.RecordCCResult = [&RecordedCompletions](const CodeCompletion &CC,
                                               const SymbolQualitySignals &,
                                               const SymbolRelevanceSignals &R,
                                               float Score) {
    RecordedCompletions.push_back({CC.Name, R.MainFileRefs, R.ScopeRefsInFile});
  };
  ASTSignals MainFileSignals;
  MainFileSignals.ReferencedSymbols[var("xy1").ID] = 3;
  MainFileSignals.ReferencedSymbols[var("xy2").ID] = 1;
  MainFileSignals.ReferencedSymbols[var("xyindex").ID] = 10;
  MainFileSignals.RelatedNamespaces["tar::"] = 5;
  MainFileSignals.RelatedNamespaces["bar::"] = 3;
  Opts.MainFileSignals = &MainFileSignals;
  Opts.AllScopes = true;
  completions(
      R"cpp(
      int xy1;
      int xy2;
      namespace bar {
      int xybar = 1;
      int a = xy^
      }
      )cpp",
      /*IndexSymbols=*/{var("xyindex"), var("tar::xytar"), var("bar::xybar")},
      Opts);
  EXPECT_THAT(RecordedCompletions,
              UnorderedElementsAre(
                  AllOf(named("xy1"), mainFileRefs(3u), scopeRefs(0u)),
                  AllOf(named("xy2"), mainFileRefs(1u), scopeRefs(0u)),
                  AllOf(named("xyindex"), mainFileRefs(10u), scopeRefs(0u)),
                  AllOf(named("xytar"), mainFileRefs(0u), scopeRefs(5u)),
                  AllOf(/*both from sema and index*/ named("xybar"),
                        mainFileRefs(0u), scopeRefs(3u))));
}

SignatureHelp
signatures(llvm::StringRef Text, Position Point,
           std::vector<Symbol> IndexSymbols = {},
           MarkupKind DocumentationFormat = MarkupKind::PlainText) {
  std::unique_ptr<SymbolIndex> Index;
  if (!IndexSymbols.empty())
    Index = memIndex(IndexSymbols);

  auto TU = TestTU::withCode(Text);
  MockFS FS;
  auto Inputs = TU.inputs(FS);
  Inputs.Index = Index.get();
  IgnoreDiagnostics Diags;
  auto CI = buildCompilerInvocation(Inputs, Diags);
  if (!CI) {
    ADD_FAILURE() << "Couldn't build CompilerInvocation";
    return {};
  }
  auto Preamble = buildPreamble(testPath(TU.Filename), *CI, Inputs,
                                /*InMemory=*/true, /*Callback=*/nullptr);
  if (!Preamble) {
    ADD_FAILURE() << "Couldn't build Preamble";
    return {};
  }
  return signatureHelp(testPath(TU.Filename), Point, *Preamble, Inputs,
                       DocumentationFormat);
}

SignatureHelp
signatures(llvm::StringRef Text, std::vector<Symbol> IndexSymbols = {},
           MarkupKind DocumentationFormat = MarkupKind::PlainText) {
  Annotations Test(Text);
  return signatures(Test.code(), Test.point(), std::move(IndexSymbols),
                    DocumentationFormat);
}

struct ExpectedParameter {
  std::string Text;
  std::pair<unsigned, unsigned> Offsets;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const ExpectedParameter &P) {
  return OS << P.Text;
}
MATCHER_P(paramsAre, P, "") {
  if (P.size() != arg.parameters.size())
    return false;
  for (unsigned I = 0; I < P.size(); ++I) {
    if (P[I].Text != arg.parameters[I].labelString ||
        P[I].Offsets != arg.parameters[I].labelOffsets)
      return false;
  }
  return true;
}
MATCHER_P(sigDoc, doc, "") { return arg.documentation.value == doc; }

/// \p AnnotatedLabel is a signature label with ranges marking parameters, e.g.
///    foo([[int p1]], [[double p2]]) -> void
Matcher<SignatureInformation> sig(llvm::StringRef AnnotatedLabel) {
  llvm::Annotations A(AnnotatedLabel);
  std::string Label = std::string(A.code());
  std::vector<ExpectedParameter> Parameters;
  for (auto Range : A.ranges()) {
    Parameters.emplace_back();

    ExpectedParameter &P = Parameters.back();
    P.Text = Label.substr(Range.Begin, Range.End - Range.Begin);
    P.Offsets.first = lspLength(llvm::StringRef(Label).substr(0, Range.Begin));
    P.Offsets.second = lspLength(llvm::StringRef(Label).substr(1, Range.End));
  }
  return AllOf(sigHelpLabeled(Label), paramsAre(Parameters));
}

TEST(SignatureHelpTest, Overloads) {
  auto Results = signatures(R"cpp(
    void foo(int x, int y);
    void foo(int x, float y);
    void foo(float x, int y);
    void foo(float x, float y);
    void bar(int x, int y = 0);
    int main() { foo(^); }
  )cpp");
  EXPECT_THAT(Results.signatures,
              UnorderedElementsAre(sig("foo([[float x]], [[float y]]) -> void"),
                                   sig("foo([[float x]], [[int y]]) -> void"),
                                   sig("foo([[int x]], [[float y]]) -> void"),
                                   sig("foo([[int x]], [[int y]]) -> void")));
  // We always prefer the first signature.
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(0, Results.activeParameter);
}

TEST(SignatureHelpTest, FunctionPointers) {
  llvm::StringLiteral Tests[] = {
      // Variable of function pointer type
      R"cpp(
    void (*foo)(int x, int y);
    int main() { foo(^); }
  )cpp",
      // Wrapped in an AttributedType
      R"cpp(
    void (__stdcall *foo)(int x, int y);
    int main() { foo(^); }
  )cpp",
      // Another syntax for an AttributedType
      R"cpp(
    void (__attribute__(stdcall) *foo)(int x, int y);
    int main() { foo(^); },
  )cpp",
      // Wrapped in a typedef
      R"cpp(
    typedef void (*fn)(int x, int y);
    fn foo;
    int main() { foo(^); }
  )cpp",
      // Wrapped in both a typedef and an AttributedTyped
      R"cpp(
    typedef void (__stdcall *fn)(int x, int y);
    fn foo;
    int main() { foo(^); }
  )cpp"};
  for (auto Test : Tests)
    EXPECT_THAT(signatures(Test).signatures,
                UnorderedElementsAre(sig("([[int x]], [[int y]]) -> void")));
}

TEST(SignatureHelpTest, Constructors) {
  std::string Top = R"cpp(
    struct S {
      S(int);
      S(const S &) = delete;
    };
  )cpp";

  auto CheckParenInit = [&](std::string Init) {
    EXPECT_THAT(signatures(Top + Init).signatures,
                UnorderedElementsAre(sig("S([[int]])")))
        << Init;
  };
  CheckParenInit("S s(^);");
  CheckParenInit("auto s = S(^);");
  CheckParenInit("auto s = new S(^);");

  auto CheckBracedInit = [&](std::string Init) {
    EXPECT_THAT(signatures(Top + Init).signatures,
                UnorderedElementsAre(sig("S{[[int]]}")))
        << Init;
  };
  CheckBracedInit("S s{^};");
  CheckBracedInit("S s = {^};");
  CheckBracedInit("auto s = S{^};");
  // FIXME: doesn't work: no ExpectedType set in ParseCXXNewExpression.
  // CheckBracedInit("auto s = new S{^};");
  CheckBracedInit("int x(S); int i = x({^});");
}

TEST(SignatureHelpTest, Aggregates) {
  std::string Top = R"cpp(
    struct S {
      int a, b, c, d;
    };
  )cpp";
  auto AggregateSig = sig("S{[[int a]], [[int b]], [[int c]], [[int d]]}");
  EXPECT_THAT(signatures(Top + "S s{^}").signatures,
              UnorderedElementsAre(AggregateSig, sig("S{}"),
                                   sig("S{[[const S &]]}"),
                                   sig("S{[[S &&]]}")));
  EXPECT_THAT(signatures(Top + "S s{1,^}").signatures,
              ElementsAre(AggregateSig));
  EXPECT_EQ(signatures(Top + "S s{1,^}").activeParameter, 1);
  EXPECT_THAT(signatures(Top + "S s{.c=3,^}").signatures,
              ElementsAre(AggregateSig));
  EXPECT_EQ(signatures(Top + "S s{.c=3,^}").activeParameter, 3);
}

TEST(SignatureHelpTest, OverloadInitListRegression) {
  auto Results = signatures(R"cpp(
    struct A {int x;};
    struct B {B(A);};
    void f();
    int main() {
      B b({1});
      f(^);
    }
  )cpp");
  EXPECT_THAT(Results.signatures, UnorderedElementsAre(sig("f() -> void")));
}

TEST(SignatureHelpTest, DefaultArgs) {
  auto Results = signatures(R"cpp(
    void bar(int x, int y = 0);
    void bar(float x = 0, int y = 42);
    int main() { bar(^
  )cpp");
  EXPECT_THAT(Results.signatures,
              UnorderedElementsAre(
                  sig("bar([[int x]], [[int y = 0]]) -> void"),
                  sig("bar([[float x = 0]], [[int y = 42]]) -> void")));
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(0, Results.activeParameter);
}

TEST(SignatureHelpTest, ActiveArg) {
  auto Results = signatures(R"cpp(
    int baz(int a, int b, int c);
    int main() { baz(baz(1,2,3), ^); }
  )cpp");
  EXPECT_THAT(Results.signatures,
              ElementsAre(sig("baz([[int a]], [[int b]], [[int c]]) -> int")));
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(1, Results.activeParameter);
}

TEST(SignatureHelpTest, OpeningParen) {
  llvm::StringLiteral Tests[] = {
      // Recursive function call.
      R"cpp(
        int foo(int a, int b, int c);
        int main() {
          foo(foo $p^( foo(10, 10, 10), ^ )));
        })cpp",
      // Functional type cast.
      R"cpp(
        struct Foo {
          Foo(int a, int b, int c);
        };
        int main() {
          Foo $p^( 10, ^ );
        })cpp",
      // New expression.
      R"cpp(
        struct Foo {
          Foo(int a, int b, int c);
        };
        int main() {
          new Foo $p^( 10, ^ );
        })cpp",
      // Macro expansion.
      R"cpp(
        int foo(int a, int b, int c);
        #define FOO foo(

        int main() {
          // Macro expansions.
          $p^FOO 10, ^ );
        })cpp",
      // Macro arguments.
      R"cpp(
        int foo(int a, int b, int c);
        int main() {
        #define ID(X) X
          // FIXME: figure out why ID(foo (foo(10), )) doesn't work when preserving
          // the recovery expression.
          ID(foo $p^( 10, ^ ))
        })cpp",
      // Dependent args.
      R"cpp(
        int foo(int a, int b);
        template <typename T> void bar(T t) {
          foo$p^(t, ^t);
        })cpp",
      // Dependent args on templated func.
      R"cpp(
        template <typename T>
        int foo(T, T);
        template <typename T> void bar(T t) {
          foo$p^(t, ^t);
        })cpp",
      // Dependent args on member.
      R"cpp(
        struct Foo { int foo(int, int); };
        template <typename T> void bar(T t) {
          Foo f;
          f.foo$p^(t, ^t);
        })cpp",
      // Dependent args on templated member.
      R"cpp(
        struct Foo { template <typename T> int foo(T, T); };
        template <typename T> void bar(T t) {
          Foo f;
          f.foo$p^(t, ^t);
        })cpp",
  };

  for (auto Test : Tests) {
    Annotations Code(Test);
    EXPECT_EQ(signatures(Code.code(), Code.point()).argListStart,
              Code.point("p"))
        << "Test source:" << Test;
  }
}

TEST(SignatureHelpTest, StalePreamble) {
  TestTU TU;
  TU.Code = "";
  IgnoreDiagnostics Diags;
  MockFS FS;
  auto Inputs = TU.inputs(FS);
  auto CI = buildCompilerInvocation(Inputs, Diags);
  ASSERT_TRUE(CI);
  auto EmptyPreamble = buildPreamble(testPath(TU.Filename), *CI, Inputs,
                                     /*InMemory=*/true, /*Callback=*/nullptr);
  ASSERT_TRUE(EmptyPreamble);

  TU.AdditionalFiles["a.h"] = "int foo(int x);";
  const Annotations Test(R"cpp(
    #include "a.h"
    void bar() { foo(^2); })cpp");
  TU.Code = Test.code().str();
  auto Results =
      signatureHelp(testPath(TU.Filename), Test.point(), *EmptyPreamble,
                    TU.inputs(FS), MarkupKind::PlainText);
  EXPECT_THAT(Results.signatures, ElementsAre(sig("foo([[int x]]) -> int")));
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(0, Results.activeParameter);
}

class IndexRequestCollector : public SymbolIndex {
public:
  IndexRequestCollector(std::vector<Symbol> Syms = {}) : Symbols(Syms) {}

  bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const override {
    std::unique_lock<std::mutex> Lock(Mut);
    Requests.push_back(Req);
    ReceivedRequestCV.notify_one();
    for (const auto &Sym : Symbols)
      Callback(Sym);
    return true;
  }

  void lookup(const LookupRequest &,
              llvm::function_ref<void(const Symbol &)>) const override {}

  bool refs(const RefsRequest &,
            llvm::function_ref<void(const Ref &)>) const override {
    return false;
  }

  void relations(const RelationsRequest &,
                 llvm::function_ref<void(const SymbolID &, const Symbol &)>)
      const override {}

  llvm::unique_function<IndexContents(llvm::StringRef) const>
  indexedFiles() const override {
    return [](llvm::StringRef) { return IndexContents::None; };
  }

  // This is incorrect, but IndexRequestCollector is not an actual index and it
  // isn't used in production code.
  size_t estimateMemoryUsage() const override { return 0; }

  const std::vector<FuzzyFindRequest> consumeRequests(size_t Num) const {
    std::unique_lock<std::mutex> Lock(Mut);
    EXPECT_TRUE(wait(Lock, ReceivedRequestCV, timeoutSeconds(30),
                     [this, Num] { return Requests.size() == Num; }));
    auto Reqs = std::move(Requests);
    Requests = {};
    return Reqs;
  }

private:
  std::vector<Symbol> Symbols;
  // We need a mutex to handle async fuzzy find requests.
  mutable std::condition_variable ReceivedRequestCV;
  mutable std::mutex Mut;
  mutable std::vector<FuzzyFindRequest> Requests;
};

// Clients have to consume exactly Num requests.
std::vector<FuzzyFindRequest> captureIndexRequests(llvm::StringRef Code,
                                                   size_t Num = 1) {
  clangd::CodeCompleteOptions Opts;
  IndexRequestCollector Requests;
  Opts.Index = &Requests;
  completions(Code, {}, Opts);
  const auto Reqs = Requests.consumeRequests(Num);
  EXPECT_EQ(Reqs.size(), Num);
  return Reqs;
}

TEST(CompletionTest, UnqualifiedIdQuery) {
  auto Requests = captureIndexRequests(R"cpp(
      namespace std {}
      using namespace std;
      namespace ns {
      void f() {
        vec^
      }
      }
  )cpp");

  EXPECT_THAT(Requests,
              ElementsAre(Field(&FuzzyFindRequest::Scopes,
                                UnorderedElementsAre("", "ns::", "std::"))));
}

TEST(CompletionTest, EnclosingScopeComesFirst) {
  auto Requests = captureIndexRequests(R"cpp(
      namespace std {}
      using namespace std;
      namespace nx {
      namespace ns {
      namespace {
      void f() {
        vec^
      }
      }
      }
      }
  )cpp");

  EXPECT_THAT(Requests,
              ElementsAre(Field(
                  &FuzzyFindRequest::Scopes,
                  UnorderedElementsAre("", "std::", "nx::ns::", "nx::"))));
  EXPECT_EQ(Requests[0].Scopes[0], "nx::ns::");
}

TEST(CompletionTest, ResolvedQualifiedIdQuery) {
  auto Requests = captureIndexRequests(R"cpp(
      namespace ns1 {}
      namespace ns2 {} // ignore
      namespace ns3 { namespace nns3 {} }
      namespace foo {
      using namespace ns1;
      using namespace ns3::nns3;
      }
      namespace ns {
      void f() {
        foo::^
      }
      }
  )cpp");

  EXPECT_THAT(Requests,
              ElementsAre(Field(
                  &FuzzyFindRequest::Scopes,
                  UnorderedElementsAre("foo::", "ns1::", "ns3::nns3::"))));
}

TEST(CompletionTest, UnresolvedQualifierIdQuery) {
  auto Requests = captureIndexRequests(R"cpp(
      namespace a {}
      using namespace a;
      namespace ns {
      void f() {
      bar::^
      }
      } // namespace ns
  )cpp");

  EXPECT_THAT(Requests,
              ElementsAre(Field(
                  &FuzzyFindRequest::Scopes,
                  UnorderedElementsAre("a::bar::", "ns::bar::", "bar::"))));
}

TEST(CompletionTest, UnresolvedNestedQualifierIdQuery) {
  auto Requests = captureIndexRequests(R"cpp(
      namespace a {}
      using namespace a;
      namespace ns {
      void f() {
      ::a::bar::^
      }
      } // namespace ns
  )cpp");

  EXPECT_THAT(Requests, ElementsAre(Field(&FuzzyFindRequest::Scopes,
                                          UnorderedElementsAre("a::bar::"))));
}

TEST(CompletionTest, EmptyQualifiedQuery) {
  auto Requests = captureIndexRequests(R"cpp(
      namespace ns {
      void f() {
      ^
      }
      } // namespace ns
  )cpp");

  EXPECT_THAT(Requests, ElementsAre(Field(&FuzzyFindRequest::Scopes,
                                          UnorderedElementsAre("", "ns::"))));
}

TEST(CompletionTest, GlobalQualifiedQuery) {
  auto Requests = captureIndexRequests(R"cpp(
      namespace ns {
      void f() {
      ::^
      }
      } // namespace ns
  )cpp");

  EXPECT_THAT(Requests, ElementsAre(Field(&FuzzyFindRequest::Scopes,
                                          UnorderedElementsAre(""))));
}

TEST(CompletionTest, NoDuplicatedQueryScopes) {
  auto Requests = captureIndexRequests(R"cpp(
      namespace {}

      namespace na {
      namespace {}
      namespace nb {
      ^
      } // namespace nb
      } // namespace na
  )cpp");

  EXPECT_THAT(Requests,
              ElementsAre(Field(&FuzzyFindRequest::Scopes,
                                UnorderedElementsAre("na::", "na::nb::", ""))));
}

TEST(CompletionTest, NoIndexCompletionsInsideClasses) {
  auto Completions = completions(
      R"cpp(
    struct Foo {
      int SomeNameOfField;
      typedef int SomeNameOfTypedefField;
    };

    Foo::^)cpp",
      {func("::SomeNameInTheIndex"), func("::Foo::SomeNameInTheIndex")});

  EXPECT_THAT(Completions.Completions,
              AllOf(Contains(labeled("SomeNameOfField")),
                    Contains(labeled("SomeNameOfTypedefField")),
                    Not(Contains(labeled("SomeNameInTheIndex")))));
}

TEST(CompletionTest, NoIndexCompletionsInsideDependentCode) {
  {
    auto Completions = completions(
        R"cpp(
      template <class T>
      void foo() {
        T::^
      }
      )cpp",
        {func("::SomeNameInTheIndex")});

    EXPECT_THAT(Completions.Completions,
                Not(Contains(labeled("SomeNameInTheIndex"))));
  }

  {
    auto Completions = completions(
        R"cpp(
      template <class T>
      void foo() {
        T::template Y<int>::^
      }
      )cpp",
        {func("::SomeNameInTheIndex")});

    EXPECT_THAT(Completions.Completions,
                Not(Contains(labeled("SomeNameInTheIndex"))));
  }

  {
    auto Completions = completions(
        R"cpp(
      template <class T>
      void foo() {
        T::foo::^
      }
      )cpp",
        {func("::SomeNameInTheIndex")});

    EXPECT_THAT(Completions.Completions,
                Not(Contains(labeled("SomeNameInTheIndex"))));
  }
}

TEST(CompletionTest, OverloadBundling) {
  clangd::CodeCompleteOptions Opts;
  Opts.BundleOverloads = true;

  std::string Context = R"cpp(
    struct X {
      // Overload with int
      int a(int) __attribute__((deprecated("", "")));
      // Overload with bool
      int a(bool);
      int b(float);

      X(int);
      X(float);
    };
    int GFuncC(int);
    int GFuncD(int);
  )cpp";

  // Member completions are bundled.
  EXPECT_THAT(completions(Context + "int y = X().^", {}, Opts).Completions,
              UnorderedElementsAre(labeled("a(…)"), labeled("b(float)")));

  // Constructor completions are bundled.
  EXPECT_THAT(completions(Context + "X z = X^", {}, Opts).Completions,
              UnorderedElementsAre(labeled("X"), labeled("X(…)")));

  // Non-member completions are bundled, including index+sema.
  Symbol NoArgsGFunc = func("GFuncC");
  EXPECT_THAT(
      completions(Context + "int y = GFunc^", {NoArgsGFunc}, Opts).Completions,
      UnorderedElementsAre(labeled("GFuncC(…)"), labeled("GFuncD(int)")));

  // Differences in header-to-insert suppress bundling.
  std::string DeclFile = URI::create(testPath("foo")).toString();
  NoArgsGFunc.CanonicalDeclaration.FileURI = DeclFile.c_str();
  NoArgsGFunc.IncludeHeaders.emplace_back("<foo>", 1, Symbol::Include);
  EXPECT_THAT(
      completions(Context + "int y = GFunc^", {NoArgsGFunc}, Opts).Completions,
      UnorderedElementsAre(AllOf(named("GFuncC"), insertInclude("<foo>")),
                           labeled("GFuncC(int)"), labeled("GFuncD(int)")));

  // Examine a bundled completion in detail.
  auto A =
      completions(Context + "int y = X().a^", {}, Opts).Completions.front();
  EXPECT_EQ(A.Name, "a");
  EXPECT_EQ(A.Signature, "(…)");
  EXPECT_EQ(A.BundleSize, 2u);
  EXPECT_EQ(A.Kind, CompletionItemKind::Method);
  EXPECT_EQ(A.ReturnType, "int"); // All overloads return int.
  // For now we just return one of the doc strings arbitrarily.
  ASSERT_TRUE(A.Documentation);
  ASSERT_FALSE(A.Deprecated); // Not all overloads deprecated.
  EXPECT_THAT(
      A.Documentation->asPlainText(),
      AnyOf(HasSubstr("Overload with int"), HasSubstr("Overload with bool")));
  EXPECT_EQ(A.SnippetSuffix, "($0)");
}

TEST(CompletionTest, OverloadBundlingSameFileDifferentURI) {
  clangd::CodeCompleteOptions Opts;
  Opts.BundleOverloads = true;

  Symbol SymX = sym("ns::X", index::SymbolKind::Function, "@F@\\0#");
  Symbol SymY = sym("ns::X", index::SymbolKind::Function, "@F@\\0#I#");
  std::string BarHeader = testPath("bar.h");
  auto BarURI = URI::create(BarHeader).toString();
  SymX.CanonicalDeclaration.FileURI = BarURI.c_str();
  SymY.CanonicalDeclaration.FileURI = BarURI.c_str();
  // The include header is different, but really it's the same file.
  SymX.IncludeHeaders.emplace_back("\"bar.h\"", 1, Symbol::Include);
  SymY.IncludeHeaders.emplace_back(BarURI.c_str(), 1, Symbol::Include);

  auto Results = completions("void f() { ::ns::^ }", {SymX, SymY}, Opts);
  // Expect both results are bundled, despite the different-but-same
  // IncludeHeader.
  ASSERT_EQ(1u, Results.Completions.size());
  const auto &R = Results.Completions.front();
  EXPECT_EQ("X", R.Name);
  EXPECT_EQ(2u, R.BundleSize);
}

TEST(CompletionTest, DocumentationFromChangedFileCrash) {
  MockFS FS;
  auto FooH = testPath("foo.h");
  auto FooCpp = testPath("foo.cpp");
  FS.Files[FooH] = R"cpp(
    // this is my documentation comment.
    int func();
  )cpp";
  FS.Files[FooCpp] = "";

  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  Annotations Source(R"cpp(
    #include "foo.h"
    int func() {
      // This makes sure we have func from header in the AST.
    }
    int a = fun^
  )cpp");
  Server.addDocument(FooCpp, Source.code(), "null", WantDiagnostics::Yes);
  // We need to wait for preamble to build.
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  // Change the header file. Completion will reuse the old preamble!
  FS.Files[FooH] = R"cpp(
    int func();
  )cpp";

  clangd::CodeCompleteOptions Opts;
  CodeCompleteResult Completions =
      cantFail(runCodeComplete(Server, FooCpp, Source.point(), Opts));
  // We shouldn't crash. Unfortunately, current workaround is to not produce
  // comments for symbols from headers.
  EXPECT_THAT(Completions.Completions,
              Contains(AllOf(Not(isDocumented()), named("func"))));
}

TEST(CompletionTest, NonDocComments) {
  const char *Text = R"cpp(
    // We ignore namespace comments, for rationale see CodeCompletionStrings.h.
    namespace comments_ns {
    }

    // ------------------
    int comments_foo();

    // A comment and a decl are separated by newlines.
    // Therefore, the comment shouldn't show up as doc comment.

    int comments_bar();

    // this comment should be in the results.
    int comments_baz();


    template <class T>
    struct Struct {
      int comments_qux();
      int comments_quux();
    };


    // This comment should not be there.

    template <class T>
    int Struct<T>::comments_qux() {
    }

    // This comment **should** be in results.
    template <class T>
    int Struct<T>::comments_quux() {
      int a = comments^;
    }
  )cpp";

  // We should not get any of those comments in completion.
  EXPECT_THAT(
      completions(Text).Completions,
      UnorderedElementsAre(AllOf(Not(isDocumented()), named("comments_foo")),
                           AllOf(isDocumented(), named("comments_baz")),
                           AllOf(isDocumented(), named("comments_quux")),
                           AllOf(Not(isDocumented()), named("comments_ns")),
                           // FIXME(ibiryukov): the following items should have
                           // empty documentation, since they are separated from
                           // a comment with an empty line. Unfortunately, I
                           // couldn't make Sema tests pass if we ignore those.
                           AllOf(isDocumented(), named("comments_bar")),
                           AllOf(isDocumented(), named("comments_qux"))));
}

TEST(CompletionTest, CompleteOnInvalidLine) {
  auto FooCpp = testPath("foo.cpp");

  MockCompilationDatabase CDB;
  MockFS FS;
  FS.Files[FooCpp] = "// empty file";

  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());
  // Run completion outside the file range.
  Position Pos;
  Pos.line = 100;
  Pos.character = 0;
  EXPECT_THAT_EXPECTED(
      runCodeComplete(Server, FooCpp, Pos, clangd::CodeCompleteOptions()),
      Failed());
}

TEST(CompletionTest, QualifiedNames) {
  auto Results = completions(
      R"cpp(
          namespace ns { int local; void both(); }
          void f() { ::ns::^ }
      )cpp",
      {func("ns::both"), cls("ns::Index")});
  // We get results from both index and sema, with no duplicates.
  EXPECT_THAT(
      Results.Completions,
      UnorderedElementsAre(scope("ns::"), scope("ns::"), scope("ns::")));
}

TEST(CompletionTest, Render) {
  CodeCompletion C;
  C.Name = "x";
  C.FilterText = "x";
  C.Signature = "(bool) const";
  C.SnippetSuffix = "(${0:bool})";
  C.ReturnType = "int";
  C.RequiredQualifier = "Foo::";
  C.Scope = "ns::Foo::";
  C.Documentation.emplace();
  C.Documentation->addParagraph().appendText("This is ").appendCode("x()");
  C.Includes.emplace_back();
  auto &Include = C.Includes.back();
  Include.Header = "\"foo.h\"";
  C.Kind = CompletionItemKind::Method;
  C.Score.Total = 1.0;
  C.Score.ExcludingName = .5;
  C.Origin = SymbolOrigin::AST | SymbolOrigin::Static;

  CodeCompleteOptions Opts;
  Opts.IncludeIndicator.Insert = "^";
  Opts.IncludeIndicator.NoInsert = "";
  Opts.EnableSnippets = false;

  auto R = C.render(Opts);
  EXPECT_EQ(R.label, "Foo::x");
  EXPECT_EQ(R.labelDetails->detail, "(bool) const");
  EXPECT_EQ(R.insertText, "Foo::x");
  EXPECT_EQ(R.insertTextFormat, InsertTextFormat::PlainText);
  EXPECT_EQ(R.filterText, "x");
  EXPECT_EQ(R.detail, "int");
  EXPECT_EQ(R.documentation->value, "From \"foo.h\"\nThis is x()");
  EXPECT_THAT(R.additionalTextEdits, IsEmpty());
  EXPECT_EQ(R.sortText, sortText(1.0, "x"));
  EXPECT_FALSE(R.deprecated);
  EXPECT_EQ(R.score, .5f);

  C.FilterText = "xtra";
  R = C.render(Opts);
  EXPECT_EQ(R.filterText, "xtra");
  EXPECT_EQ(R.sortText, sortText(1.0, "xtra"));

  Opts.EnableSnippets = true;
  R = C.render(Opts);
  EXPECT_EQ(R.insertText, "Foo::x(${0:bool})");
  EXPECT_EQ(R.insertTextFormat, InsertTextFormat::Snippet);

  C.SnippetSuffix = "";
  R = C.render(Opts);
  EXPECT_EQ(R.insertText, "Foo::x");
  EXPECT_EQ(R.insertTextFormat, InsertTextFormat::PlainText);

  Include.Insertion.emplace();
  R = C.render(Opts);
  EXPECT_EQ(R.label, "^Foo::x");
  EXPECT_EQ(R.labelDetails->detail, "(bool) const");
  EXPECT_THAT(R.additionalTextEdits, Not(IsEmpty()));

  Opts.ShowOrigins = true;
  R = C.render(Opts);
  EXPECT_EQ(R.label, "^[AS]Foo::x");
  EXPECT_EQ(R.labelDetails->detail, "(bool) const");

  C.BundleSize = 2;
  R = C.render(Opts);
  EXPECT_EQ(R.detail, "[2 overloads]");
  EXPECT_EQ(R.documentation->value, "From \"foo.h\"\nThis is x()");

  C.Deprecated = true;
  R = C.render(Opts);
  EXPECT_TRUE(R.deprecated);

  Opts.DocumentationFormat = MarkupKind::Markdown;
  R = C.render(Opts);
  EXPECT_EQ(R.documentation->value, "From `\"foo.h\"`  \nThis is `x()`");
}

TEST(CompletionTest, IgnoreRecoveryResults) {
  auto Results = completions(
      R"cpp(
          namespace ns { int NotRecovered() { return 0; } }
          void f() {
            // Sema enters recovery mode first and then normal mode.
            if (auto x = ns::NotRecover^)
          }
      )cpp");
  EXPECT_THAT(Results.Completions, UnorderedElementsAre(named("NotRecovered")));
}

TEST(CompletionTest, ScopeOfClassFieldInConstructorInitializer) {
  auto Results = completions(
      R"cpp(
        namespace ns {
          class X { public: X(); int x_; };
          X::X() : x_^(0) {}
        }
      )cpp");
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(scope("ns::X::"), named("x_"))));
}

// Like other class members, constructor init lists have to parse what's below,
// after the completion point.
// But recovering from an incomplete constructor init list is particularly
// tricky because the bulk of the list is not surrounded by brackets.
TEST(CompletionTest, ConstructorInitListIncomplete) {
  auto Results = completions(
      R"cpp(
        namespace ns {
          struct X {
            X() : x^
            int xyz_;
          };
        }
      )cpp");
  EXPECT_THAT(Results.Completions, ElementsAre(named("xyz_")));

  Results = completions(
      R"cpp(
        int foo();

        namespace ns {
          struct X {
            X() : xyz_(fo^
            int xyz_;
          };
        }
      )cpp");
  EXPECT_THAT(Results.Completions, ElementsAre(named("foo")));
}

TEST(CompletionTest, CodeCompletionContext) {
  auto Results = completions(
      R"cpp(
        namespace ns {
          class X { public: X(); int x_; };
          void f() {
            X x;
            x.^;
          }
        }
      )cpp");

  EXPECT_THAT(Results.Context, CodeCompletionContext::CCC_DotMemberAccess);
}

TEST(CompletionTest, FixItForArrowToDot) {
  MockFS FS;
  MockCompilationDatabase CDB;

  CodeCompleteOptions Opts;
  Opts.IncludeFixIts = true;
  const char *Code =
      R"cpp(
        class Auxilary {
         public:
          void AuxFunction();
        };
        class ClassWithPtr {
         public:
          void MemberFunction();
          Auxilary* operator->() const;
          Auxilary* Aux;
        };
        void f() {
          ClassWithPtr x;
          x[[->]]^;
        }
      )cpp";
  auto Results = completions(Code, {}, Opts);
  EXPECT_EQ(Results.Completions.size(), 3u);

  TextEdit ReplacementEdit;
  ReplacementEdit.range = Annotations(Code).range();
  ReplacementEdit.newText = ".";
  for (const auto &C : Results.Completions) {
    EXPECT_TRUE(C.FixIts.size() == 1u || C.Name == "AuxFunction");
    if (!C.FixIts.empty()) {
      EXPECT_THAT(C.FixIts, ElementsAre(ReplacementEdit));
    }
  }
}

TEST(CompletionTest, FixItForDotToArrow) {
  CodeCompleteOptions Opts;
  Opts.IncludeFixIts = true;
  const char *Code =
      R"cpp(
        class Auxilary {
         public:
          void AuxFunction();
        };
        class ClassWithPtr {
         public:
          void MemberFunction();
          Auxilary* operator->() const;
          Auxilary* Aux;
        };
        void f() {
          ClassWithPtr x;
          x[[.]]^;
        }
      )cpp";
  auto Results = completions(Code, {}, Opts);
  EXPECT_EQ(Results.Completions.size(), 3u);

  TextEdit ReplacementEdit;
  ReplacementEdit.range = Annotations(Code).range();
  ReplacementEdit.newText = "->";
  for (const auto &C : Results.Completions) {
    EXPECT_TRUE(C.FixIts.empty() || C.Name == "AuxFunction");
    if (!C.FixIts.empty()) {
      EXPECT_THAT(C.FixIts, ElementsAre(ReplacementEdit));
    }
  }
}

TEST(CompletionTest, RenderWithFixItMerged) {
  TextEdit FixIt;
  FixIt.range.end.character = 5;
  FixIt.newText = "->";

  CodeCompletion C;
  C.Name = "x";
  C.RequiredQualifier = "Foo::";
  C.FixIts = {FixIt};
  C.CompletionTokenRange.start.character = 5;

  CodeCompleteOptions Opts;
  Opts.IncludeFixIts = true;

  auto R = C.render(Opts);
  EXPECT_TRUE(R.textEdit);
  EXPECT_EQ(R.textEdit->newText, "->Foo::x");
  EXPECT_TRUE(R.additionalTextEdits.empty());
}

TEST(CompletionTest, RenderWithFixItNonMerged) {
  TextEdit FixIt;
  FixIt.range.end.character = 4;
  FixIt.newText = "->";

  CodeCompletion C;
  C.Name = "x";
  C.RequiredQualifier = "Foo::";
  C.FixIts = {FixIt};
  C.CompletionTokenRange.start.character = 5;

  CodeCompleteOptions Opts;
  Opts.IncludeFixIts = true;

  auto R = C.render(Opts);
  EXPECT_TRUE(R.textEdit);
  EXPECT_EQ(R.textEdit->newText, "Foo::x");
  EXPECT_THAT(R.additionalTextEdits, UnorderedElementsAre(FixIt));
}

TEST(CompletionTest, CompletionTokenRange) {
  MockFS FS;
  MockCompilationDatabase CDB;
  TestTU TU;
  TU.AdditionalFiles["foo/abc/foo.h"] = "";

  constexpr const char *TestCodes[] = {
      R"cpp(
        class Auxilary {
         public:
          void AuxFunction();
        };
        void f() {
          Auxilary x;
          x.[[Aux]]^;
        }
      )cpp",
      R"cpp(
        class Auxilary {
         public:
          void AuxFunction();
        };
        void f() {
          Auxilary x;
          x.[[]]^;
        }
      )cpp",
      R"cpp(
        #include "foo/[[a^/]]foo.h"
      )cpp",
      R"cpp(
        #include "foo/abc/[[fo^o.h"]]
      )cpp",
  };
  for (const auto &Text : TestCodes) {
    Annotations TestCode(Text);
    TU.Code = TestCode.code().str();
    auto Results = completions(TU, TestCode.point());
    if (Results.Completions.size() != 1) {
      ADD_FAILURE() << "Results.Completions.size() != 1" << Text;
      continue;
    }
    EXPECT_THAT(Results.Completions.front().CompletionTokenRange,
                TestCode.range());
  }
}

TEST(SignatureHelpTest, OverloadsOrdering) {
  const auto Results = signatures(R"cpp(
    void foo(int x);
    void foo(int x, float y);
    void foo(float x, int y);
    void foo(float x, float y);
    void foo(int x, int y = 0);
    int main() { foo(^); }
  )cpp");
  EXPECT_THAT(Results.signatures,
              ElementsAre(sig("foo([[int x]]) -> void"),
                          sig("foo([[int x]], [[int y = 0]]) -> void"),
                          sig("foo([[float x]], [[int y]]) -> void"),
                          sig("foo([[int x]], [[float y]]) -> void"),
                          sig("foo([[float x]], [[float y]]) -> void")));
  // We always prefer the first signature.
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(0, Results.activeParameter);
}

TEST(SignatureHelpTest, InstantiatedSignatures) {
  StringRef Sig0 = R"cpp(
    template <class T>
    void foo(T, T, T);

    int main() {
      foo<int>(^);
    }
  )cpp";

  EXPECT_THAT(signatures(Sig0).signatures,
              ElementsAre(sig("foo([[T]], [[T]], [[T]]) -> void")));

  StringRef Sig1 = R"cpp(
    template <class T>
    void foo(T, T, T);

    int main() {
      foo(10, ^);
    })cpp";

  EXPECT_THAT(signatures(Sig1).signatures,
              ElementsAre(sig("foo([[T]], [[T]], [[T]]) -> void")));

  StringRef Sig2 = R"cpp(
    template <class ...T>
    void foo(T...);

    int main() {
      foo<int>(^);
    }
  )cpp";

  EXPECT_THAT(signatures(Sig2).signatures,
              ElementsAre(sig("foo([[T...]]) -> void")));

  // It is debatable whether we should substitute the outer template parameter
  // ('T') in that case. Currently we don't substitute it in signature help, but
  // do substitute in code complete.
  // FIXME: make code complete and signature help consistent, figure out which
  // way is better.
  StringRef Sig3 = R"cpp(
    template <class T>
    struct X {
      template <class U>
      void foo(T, U);
    };

    int main() {
      X<int>().foo<double>(^)
    }
  )cpp";

  EXPECT_THAT(signatures(Sig3).signatures,
              ElementsAre(sig("foo([[T]], [[U]]) -> void")));
}

TEST(SignatureHelpTest, IndexDocumentation) {
  Symbol Foo0 = sym("foo", index::SymbolKind::Function, "@F@\\0#");
  Foo0.Documentation = "doc from the index";
  Symbol Foo1 = sym("foo", index::SymbolKind::Function, "@F@\\0#I#");
  Foo1.Documentation = "doc from the index";
  Symbol Foo2 = sym("foo", index::SymbolKind::Function, "@F@\\0#I#I#");

  StringRef Sig0 = R"cpp(
    int foo();
    int foo(double);

    void test() {
      foo(^);
    }
  )cpp";

  EXPECT_THAT(
      signatures(Sig0, {Foo0}).signatures,
      ElementsAre(AllOf(sig("foo() -> int"), sigDoc("doc from the index")),
                  AllOf(sig("foo([[double]]) -> int"), sigDoc(""))));

  StringRef Sig1 = R"cpp(
    int foo();
    // Overriden doc from sema
    int foo(int);
    // doc from sema
    int foo(int, int);

    void test() {
      foo(^);
    }
  )cpp";

  EXPECT_THAT(
      signatures(Sig1, {Foo0, Foo1, Foo2}).signatures,
      ElementsAre(
          AllOf(sig("foo() -> int"), sigDoc("doc from the index")),
          AllOf(sig("foo([[int]]) -> int"), sigDoc("Overriden doc from sema")),
          AllOf(sig("foo([[int]], [[int]]) -> int"), sigDoc("doc from sema"))));
}

TEST(SignatureHelpTest, DynamicIndexDocumentation) {
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer::Options Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true;
  ClangdServer Server(CDB, FS, Opts);

  FS.Files[testPath("foo.h")] = R"cpp(
    struct Foo {
       // Member doc
       int foo();
    };
  )cpp";
  Annotations FileContent(R"cpp(
    #include "foo.h"
    void test() {
      Foo f;
      f.foo(^);
    }
  )cpp");
  auto File = testPath("test.cpp");
  Server.addDocument(File, FileContent.code());
  // Wait for the dynamic index being built.
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  EXPECT_THAT(llvm::cantFail(runSignatureHelp(Server, File, FileContent.point(),
                                              MarkupKind::PlainText))
                  .signatures,
              ElementsAre(AllOf(sig("foo() -> int"), sigDoc("Member doc"))));
}

TEST(CompletionTest, CompletionFunctionArgsDisabled) {
  CodeCompleteOptions Opts;
  Opts.EnableSnippets = true;
  Opts.EnableFunctionArgSnippets = false;

  {
    auto Results = completions(
        R"cpp(
      void xfoo();
      void xfoo(int x, int y);
      void f() { xfo^ })cpp",
        {}, Opts);
    EXPECT_THAT(
        Results.Completions,
        UnorderedElementsAre(AllOf(named("xfoo"), snippetSuffix("()")),
                             AllOf(named("xfoo"), snippetSuffix("($0)"))));
  }
  {
    auto Results = completions(
        R"cpp(
      void xbar();
      void f() { xba^ })cpp",
        {}, Opts);
    EXPECT_THAT(Results.Completions, UnorderedElementsAre(AllOf(
                                         named("xbar"), snippetSuffix("()"))));
  }
  {
    Opts.BundleOverloads = true;
    auto Results = completions(
        R"cpp(
      void xfoo();
      void xfoo(int x, int y);
      void f() { xfo^ })cpp",
        {}, Opts);
    EXPECT_THAT(
        Results.Completions,
        UnorderedElementsAre(AllOf(named("xfoo"), snippetSuffix("($0)"))));
  }
  {
    auto Results = completions(
        R"cpp(
      template <class T, class U>
      void xfoo(int a, U b);
      void f() { xfo^ })cpp",
        {}, Opts);
    EXPECT_THAT(
        Results.Completions,
        UnorderedElementsAre(AllOf(named("xfoo"), snippetSuffix("<$1>($0)"))));
  }
  {
    auto Results = completions(
        R"cpp(
      template <class T>
      class foo_class{};
      template <class T>
      using foo_alias = T**;
      void f() { foo_^ })cpp",
        {}, Opts);
    EXPECT_THAT(
        Results.Completions,
        UnorderedElementsAre(AllOf(named("foo_class"), snippetSuffix("<$0>")),
                             AllOf(named("foo_alias"), snippetSuffix("<$0>"))));
  }
  {
    auto Results = completions(
        R"cpp(
      #define FOO(x, y) x##f
      FO^ )cpp",
        {}, Opts);
    EXPECT_THAT(Results.Completions, UnorderedElementsAre(AllOf(
                                         named("FOO"), snippetSuffix("($0)"))));
  }
}

TEST(CompletionTest, SuggestOverrides) {
  constexpr const char *const Text(R"cpp(
  class A {
   public:
    virtual void vfunc(bool param);
    virtual void vfunc(bool param, int p);
    void func(bool param);
  };
  class B : public A {
  virtual void ttt(bool param) const;
  void vfunc(bool param, int p) override;
  };
  class C : public B {
   public:
    void vfunc(bool param) override;
    ^
  };
  )cpp");
  const auto Results = completions(Text);
  EXPECT_THAT(
      Results.Completions,
      AllOf(Contains(AllOf(labeled("void vfunc(bool param, int p) override"),
                           nameStartsWith("vfunc"))),
            Contains(AllOf(labeled("void ttt(bool param) const override"),
                           nameStartsWith("ttt"))),
            Not(Contains(labeled("void vfunc(bool param) override")))));
}

TEST(CompletionTest, OverridesNonIdentName) {
  // Check the completions call does not crash.
  completions(R"cpp(
    struct Base {
      virtual ~Base() = 0;
      virtual operator int() = 0;
      virtual Base& operator+(Base&) = 0;
    };

    struct Derived : Base {
      ^
    };
  )cpp");
}

TEST(CompletionTest, NoCrashOnMissingNewLineAtEOF) {
  auto FooCpp = testPath("foo.cpp");

  MockCompilationDatabase CDB;
  MockFS FS;
  Annotations F("#pragma ^ // no new line");
  FS.Files[FooCpp] = F.code().str();
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());
  runAddDocument(Server, FooCpp, F.code());
  // Run completion outside the file range.
  EXPECT_THAT(cantFail(runCodeComplete(Server, FooCpp, F.point(),
                                       clangd::CodeCompleteOptions()))
                  .Completions,
              IsEmpty());
  EXPECT_THAT(cantFail(runSignatureHelp(Server, FooCpp, F.point(),
                                        MarkupKind::PlainText))
                  .signatures,
              IsEmpty());
}

TEST(GuessCompletionPrefix, Filters) {
  for (llvm::StringRef Case : {
           "[[scope::]][[ident]]^",
           "[[]][[]]^",
           "\n[[]][[]]^",
           "[[]][[ab]]^",
           "x.[[]][[ab]]^",
           "x.[[]][[]]^",
           "[[x::]][[ab]]^",
           "[[x::]][[]]^",
           "[[::x::]][[ab]]^",
           "some text [[scope::more::]][[identif]]^ier",
           "some text [[scope::]][[mor]]^e::identifier",
           "weird case foo::[[::bar::]][[baz]]^",
           "/* [[]][[]]^ */",
       }) {
    Annotations F(Case);
    auto Offset = cantFail(positionToOffset(F.code(), F.point()));
    auto ToStringRef = [&](Range R) {
      return F.code().slice(cantFail(positionToOffset(F.code(), R.start)),
                            cantFail(positionToOffset(F.code(), R.end)));
    };
    auto WantQualifier = ToStringRef(F.ranges()[0]),
         WantName = ToStringRef(F.ranges()[1]);

    auto Prefix = guessCompletionPrefix(F.code(), Offset);
    // Even when components are empty, check their offsets are correct.
    EXPECT_EQ(WantQualifier, Prefix.Qualifier) << Case;
    EXPECT_EQ(WantQualifier.begin(), Prefix.Qualifier.begin()) << Case;
    EXPECT_EQ(WantName, Prefix.Name) << Case;
    EXPECT_EQ(WantName.begin(), Prefix.Name.begin()) << Case;
  }
}

TEST(CompletionTest, EnableSpeculativeIndexRequest) {
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  auto File = testPath("foo.cpp");
  Annotations Test(R"cpp(
      namespace ns1 { int abc; }
      namespace ns2 { int abc; }
      void f() { ns1::ab$1^; ns1::ab$2^; }
      void f2() { ns2::ab$3^; }
  )cpp");
  runAddDocument(Server, File, Test.code());
  clangd::CodeCompleteOptions Opts = {};

  IndexRequestCollector Requests;
  Opts.Index = &Requests;

  auto CompleteAtPoint = [&](StringRef P) {
    auto CCR = cantFail(runCodeComplete(Server, File, Test.point(P), Opts));
    EXPECT_TRUE(CCR.HasMore);
  };

  CompleteAtPoint("1");
  auto Reqs1 = Requests.consumeRequests(1);
  ASSERT_EQ(Reqs1.size(), 1u);
  EXPECT_THAT(Reqs1[0].Scopes, UnorderedElementsAre("ns1::"));

  CompleteAtPoint("2");
  auto Reqs2 = Requests.consumeRequests(1);
  // Speculation succeeded. Used speculative index result.
  ASSERT_EQ(Reqs2.size(), 1u);
  EXPECT_EQ(Reqs2[0], Reqs1[0]);

  CompleteAtPoint("3");
  // Speculation failed. Sent speculative index request and the new index
  // request after sema.
  auto Reqs3 = Requests.consumeRequests(2);
  ASSERT_EQ(Reqs3.size(), 2u);
}

TEST(CompletionTest, InsertTheMostPopularHeader) {
  std::string DeclFile = URI::create(testPath("foo")).toString();
  Symbol Sym = func("Func");
  Sym.CanonicalDeclaration.FileURI = DeclFile.c_str();
  Sym.IncludeHeaders.emplace_back("\"foo.h\"", 2, Symbol::Include);
  Sym.IncludeHeaders.emplace_back("\"bar.h\"", 1000, Symbol::Include);

  auto Results = completions("Fun^", {Sym}).Completions;
  assert(!Results.empty());
  EXPECT_THAT(Results[0], AllOf(named("Func"), insertInclude("\"bar.h\"")));
  EXPECT_EQ(Results[0].Includes.size(), 2u);
}

TEST(CompletionTest, InsertIncludeOrImport) {
  std::string DeclFile = URI::create(testPath("foo")).toString();
  Symbol Sym = func("Func");
  Sym.CanonicalDeclaration.FileURI = DeclFile.c_str();
  Sym.IncludeHeaders.emplace_back("\"bar.h\"", 1000,
                                  Symbol::Include | Symbol::Import);
  CodeCompleteOptions Opts;
  // Should only take effect in import contexts.
  Opts.ImportInsertions = true;
  auto Results = completions("Fun^", {Sym}, Opts).Completions;
  assert(!Results.empty());
  EXPECT_THAT(Results[0],
              AllOf(named("Func"), insertIncludeText("#include \"bar.h\"\n")));

  ASTSignals Signals;
  Signals.InsertionDirective = Symbol::IncludeDirective::Import;
  Opts.MainFileSignals = &Signals;
  Results = completions("Fun^", {Sym}, Opts, "Foo.m").Completions;
  assert(!Results.empty());
  EXPECT_THAT(Results[0],
              AllOf(named("Func"), insertIncludeText("#import \"bar.h\"\n")));

  Sym.IncludeHeaders[0].SupportedDirectives = Symbol::Import;
  Results = completions("Fun^", {Sym}).Completions;
  assert(!Results.empty());
  EXPECT_THAT(Results[0], AllOf(named("Func"), Not(insertInclude())));
}

TEST(CompletionTest, NoInsertIncludeIfOnePresent) {
  Annotations Test(R"cpp(
    #include "foo.h"
    Fun^
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.AdditionalFiles["foo.h"] = "";

  std::string DeclFile = URI::create(testPath("foo")).toString();
  Symbol Sym = func("Func");
  Sym.CanonicalDeclaration.FileURI = DeclFile.c_str();
  Sym.IncludeHeaders.emplace_back("\"foo.h\"", 2, Symbol::Include);
  Sym.IncludeHeaders.emplace_back("\"bar.h\"", 1000, Symbol::Include);

  EXPECT_THAT(completions(TU, Test.point(), {Sym}).Completions,
              UnorderedElementsAre(AllOf(named("Func"), hasInclude("\"foo.h\""),
                                         Not(insertInclude()))));
}

TEST(CompletionTest, MergeMacrosFromIndexAndSema) {
  Symbol Sym;
  Sym.Name = "Clangd_Macro_Test";
  Sym.ID = SymbolID("c:foo.cpp@8@macro@Clangd_Macro_Test");
  Sym.SymInfo.Kind = index::SymbolKind::Macro;
  Sym.Flags |= Symbol::IndexedForCodeCompletion;
  EXPECT_THAT(completions("#define Clangd_Macro_Test\nClangd_Macro_T^", {Sym})
                  .Completions,
              UnorderedElementsAre(named("Clangd_Macro_Test")));
}

TEST(CompletionTest, MacroFromPreamble) {
  Annotations Test(R"cpp(#define CLANGD_PREAMBLE_MAIN x

          int x = 0;
          #define CLANGD_MAIN x
          void f() { CLANGD_^ }
      )cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.HeaderCode = "#define CLANGD_PREAMBLE_HEADER x";
  auto Results = completions(TU, Test.point(), {func("CLANGD_INDEX")});
  // We should get results from the main file, including the preamble section.
  // However no results from included files (the index should cover them).
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(named("CLANGD_PREAMBLE_MAIN"),
                                   named("CLANGD_MAIN"),
                                   named("CLANGD_INDEX")));
}

TEST(CompletionTest, DeprecatedResults) {
  std::string Body = R"cpp(
    void TestClangd();
    void TestClangc() __attribute__((deprecated("", "")));
  )cpp";

  EXPECT_THAT(
      completions(Body + "int main() { TestClang^ }").Completions,
      UnorderedElementsAre(AllOf(named("TestClangd"), Not(deprecated())),
                           AllOf(named("TestClangc"), deprecated())));
}

TEST(SignatureHelpTest, PartialSpec) {
  const auto Results = signatures(R"cpp(
      template <typename T> struct Foo {};
      template <typename T> struct Foo<T*> { Foo(T); };
      Foo<int*> F(^);)cpp");
  EXPECT_THAT(Results.signatures, Contains(sig("Foo([[T]])")));
  EXPECT_EQ(0, Results.activeParameter);
}

TEST(SignatureHelpTest, InsideArgument) {
  {
    const auto Results = signatures(R"cpp(
      void foo(int x);
      void foo(int x, int y);
      int main() { foo(1+^); }
    )cpp");
    EXPECT_THAT(Results.signatures,
                ElementsAre(sig("foo([[int x]]) -> void"),
                            sig("foo([[int x]], [[int y]]) -> void")));
    EXPECT_EQ(0, Results.activeParameter);
  }
  {
    const auto Results = signatures(R"cpp(
      void foo(int x);
      void foo(int x, int y);
      int main() { foo(1^); }
    )cpp");
    EXPECT_THAT(Results.signatures,
                ElementsAre(sig("foo([[int x]]) -> void"),
                            sig("foo([[int x]], [[int y]]) -> void")));
    EXPECT_EQ(0, Results.activeParameter);
  }
  {
    const auto Results = signatures(R"cpp(
      void foo(int x);
      void foo(int x, int y);
      int main() { foo(1^0); }
    )cpp");
    EXPECT_THAT(Results.signatures,
                ElementsAre(sig("foo([[int x]]) -> void"),
                            sig("foo([[int x]], [[int y]]) -> void")));
    EXPECT_EQ(0, Results.activeParameter);
  }
  {
    const auto Results = signatures(R"cpp(
      void foo(int x);
      void foo(int x, int y);
      int bar(int x, int y);
      int main() { bar(foo(2, 3^)); }
    )cpp");
    EXPECT_THAT(Results.signatures,
                ElementsAre(sig("foo([[int x]], [[int y]]) -> void")));
    EXPECT_EQ(1, Results.activeParameter);
  }
}

TEST(SignatureHelpTest, ConstructorInitializeFields) {
  {
    const auto Results = signatures(R"cpp(
      struct A { A(int); };
      struct B {
        B() : a_elem(^) {}
        A a_elem;
      };
    )cpp");
    EXPECT_THAT(Results.signatures,
                UnorderedElementsAre(sig("A([[int]])"), sig("A([[A &&]])"),
                                     sig("A([[const A &]])")));
  }
  {
    const auto Results = signatures(R"cpp(
      struct A { A(int); };
      struct B {
        B() : a_elem(^
        A a_elem;
      };
    )cpp");
    // FIXME: currently the parser skips over the decl of a_elem as part of the
    // (broken) init list, so we don't get signatures for the first member.
    EXPECT_THAT(Results.signatures, IsEmpty());
  }
  {
    const auto Results = signatures(R"cpp(
      struct A { A(int); };
      struct B {
        B() : a_elem(^
        int dummy_elem;
        A a_elem;
      };
    )cpp");
    EXPECT_THAT(Results.signatures,
                UnorderedElementsAre(sig("A([[int]])"), sig("A([[A &&]])"),
                                     sig("A([[const A &]])")));
  }
  {
    const auto Results = signatures(R"cpp(
      struct A {
        A(int);
      };
      struct C {
        C(int);
        C(A);
      };
      struct B {
        B() : c_elem(A(1^)) {}
        C c_elem;
      };
    )cpp");
    EXPECT_THAT(Results.signatures,
                UnorderedElementsAre(sig("A([[int]])"), sig("A([[A &&]])"),
                                     sig("A([[const A &]])")));
  }
}

TEST(SignatureHelpTest, Variadic) {
  const std::string Header = R"cpp(
    void fun(int x, ...) {}
    void test() {)cpp";
  const std::string ExpectedSig = "fun([[int x]], [[...]]) -> void";

  {
    const auto Result = signatures(Header + "fun(^);}");
    EXPECT_EQ(0, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
  {
    const auto Result = signatures(Header + "fun(1, ^);}");
    EXPECT_EQ(1, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
  {
    const auto Result = signatures(Header + "fun(1, 2, ^);}");
    EXPECT_EQ(1, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
}

TEST(SignatureHelpTest, VariadicTemplate) {
  const std::string Header = R"cpp(
    template<typename T, typename ...Args>
    void fun(T t, Args ...args) {}
    void test() {)cpp";
  const std::string ExpectedSig = "fun([[T t]], [[Args args...]]) -> void";

  {
    const auto Result = signatures(Header + "fun(^);}");
    EXPECT_EQ(0, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
  {
    const auto Result = signatures(Header + "fun(1, ^);}");
    EXPECT_EQ(1, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
  {
    const auto Result = signatures(Header + "fun(1, 2, ^);}");
    EXPECT_EQ(1, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
}

TEST(SignatureHelpTest, VariadicMethod) {
  const std::string Header = R"cpp(
  class C {
    template<typename T, typename ...Args>
    void fun(T t, Args ...args) {}
  };
    void test() {C c; )cpp";
  const std::string ExpectedSig = "fun([[T t]], [[Args args...]]) -> void";

  {
    const auto Result = signatures(Header + "c.fun(^);}");
    EXPECT_EQ(0, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
  {
    const auto Result = signatures(Header + "c.fun(1, ^);}");
    EXPECT_EQ(1, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
  {
    const auto Result = signatures(Header + "c.fun(1, 2, ^);}");
    EXPECT_EQ(1, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
}

TEST(SignatureHelpTest, VariadicType) {
  const std::string Header = R"cpp(
  void fun(int x, ...) {}
  auto get_fun() { return fun; }
  void test() {
  )cpp";
  const std::string ExpectedSig = "([[int]], [[...]]) -> void";

  {
    const auto Result = signatures(Header + "get_fun()(^);}");
    EXPECT_EQ(0, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
  {
    const auto Result = signatures(Header + "get_fun()(1, ^);}");
    EXPECT_EQ(1, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
  {
    const auto Result = signatures(Header + "get_fun()(1, 2, ^);}");
    EXPECT_EQ(1, Result.activeParameter);
    EXPECT_THAT(Result.signatures, UnorderedElementsAre(sig(ExpectedSig)));
  }
}

TEST(CompletionTest, IncludedCompletionKinds) {
  Annotations Test(R"cpp(#include "^)cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.AdditionalFiles["sub/bar.h"] = "";
  TU.ExtraArgs.push_back("-I" + testPath("sub"));

  auto Results = completions(TU, Test.point());
  EXPECT_THAT(Results.Completions,
              AllOf(has("sub/", CompletionItemKind::Folder),
                    has("bar.h\"", CompletionItemKind::File)));
}

TEST(CompletionTest, NoCrashAtNonAlphaIncludeHeader) {
  completions(
      R"cpp(
        #include "./^"
      )cpp");
}

TEST(CompletionTest, NoAllScopesCompletionWhenQualified) {
  clangd::CodeCompleteOptions Opts = {};
  Opts.AllScopes = true;

  auto Results = completions(
      R"cpp(
    void f() { na::Clangd^ }
  )cpp",
      {cls("na::ClangdA"), cls("nx::ClangdX"), cls("Clangd3")}, Opts);
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(
                  AllOf(qualifier(""), scope("na::"), named("ClangdA"))));
}

TEST(CompletionTest, AllScopesCompletion) {
  clangd::CodeCompleteOptions Opts = {};
  Opts.AllScopes = true;

  auto Results = completions(
      R"cpp(
    namespace na {
    void f() { Clangd^ }
    }
  )cpp",
      {cls("nx::Clangd1"), cls("ny::Clangd2"), cls("Clangd3"),
       cls("na::nb::Clangd4"), enmConstant("na::C::Clangd5")},
      Opts);
  EXPECT_THAT(
      Results.Completions,
      UnorderedElementsAre(AllOf(qualifier("nx::"), named("Clangd1"),
                                 kind(CompletionItemKind::Class)),
                           AllOf(qualifier("ny::"), named("Clangd2"),
                                 kind(CompletionItemKind::Class)),
                           AllOf(qualifier(""), scope(""), named("Clangd3"),
                                 kind(CompletionItemKind::Class)),
                           AllOf(qualifier("nb::"), named("Clangd4"),
                                 kind(CompletionItemKind::Class)),
                           AllOf(qualifier("C::"), named("Clangd5"),
                                 kind(CompletionItemKind::EnumMember))));
}

TEST(CompletionTest, NoQualifierIfShadowed) {
  clangd::CodeCompleteOptions Opts = {};
  Opts.AllScopes = true;

  auto Results = completions(R"cpp(
    namespace nx { class Clangd1 {}; }
    using nx::Clangd1;
    void f() { Clangd^ }
  )cpp",
                             {cls("nx::Clangd1"), cls("nx::Clangd2")}, Opts);
  // Although Clangd1 is from another namespace, Sema tells us it's in-scope and
  // needs no qualifier.
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(qualifier(""), named("Clangd1")),
                                   AllOf(qualifier("nx::"), named("Clangd2"))));
}

TEST(CompletionTest, NoCompletionsForNewNames) {
  clangd::CodeCompleteOptions Opts;
  Opts.AllScopes = true;
  auto Results = completions(R"cpp(
      void f() { int n^ }
    )cpp",
                             {cls("naber"), cls("nx::naber")}, Opts);
  EXPECT_THAT(Results.Completions, UnorderedElementsAre());
}

TEST(CompletionTest, Lambda) {
  clangd::CodeCompleteOptions Opts = {};

  auto Results = completions(R"cpp(
    void function() {
      auto Lambda = [](int a, const double &b) {return 1.f;};
      Lam^
    }
  )cpp",
                             {}, Opts);

  ASSERT_EQ(Results.Completions.size(), 1u);
  const auto &A = Results.Completions.front();
  EXPECT_EQ(A.Name, "Lambda");
  EXPECT_EQ(A.Signature, "(int a, const double &b) const");
  EXPECT_EQ(A.Kind, CompletionItemKind::Variable);
  EXPECT_EQ(A.ReturnType, "float");
  EXPECT_EQ(A.SnippetSuffix, "(${1:int a}, ${2:const double &b})");
}

TEST(CompletionTest, StructuredBinding) {
  clangd::CodeCompleteOptions Opts = {};

  auto Results = completions(R"cpp(
    struct S {
      using Float = float;
      int x;
      Float y;
    };
    void function() {
      const auto &[xxx, yyy] = S{};
      yyy^
    }
  )cpp",
                             {}, Opts);

  ASSERT_EQ(Results.Completions.size(), 1u);
  const auto &A = Results.Completions.front();
  EXPECT_EQ(A.Name, "yyy");
  EXPECT_EQ(A.Kind, CompletionItemKind::Variable);
  EXPECT_EQ(A.ReturnType, "const Float");
}

TEST(CompletionTest, ObjectiveCMethodNoArguments) {
  auto Results = completions(R"objc(
      @interface Foo
      @property(nonatomic, setter=setXToIgnoreComplete:) int value;
      @end
      Foo *foo = [Foo new]; int y = [foo v^]
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("value")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(returnType("int")));
  EXPECT_THAT(C, ElementsAre(signature("")));
  EXPECT_THAT(C, ElementsAre(snippetSuffix("")));
}

TEST(CompletionTest, ObjectiveCMethodOneArgument) {
  auto Results = completions(R"objc(
      @interface Foo
      - (int)valueForCharacter:(char)c;
      @end
      Foo *foo = [Foo new]; int y = [foo v^]
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("valueForCharacter:")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(returnType("int")));
  EXPECT_THAT(C, ElementsAre(signature("(char)")));
  EXPECT_THAT(C, ElementsAre(snippetSuffix("${1:(char)}")));
}

TEST(CompletionTest, ObjectiveCMethodTwoArgumentsFromBeginning) {
  auto Results = completions(R"objc(
      @interface Foo
      + (id)fooWithValue:(int)value fooey:(unsigned int)fooey;
      @end
      id val = [Foo foo^]
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("fooWithValue:")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(returnType("id")));
  EXPECT_THAT(C, ElementsAre(signature("(int) fooey:(unsigned int)")));
  EXPECT_THAT(
      C, ElementsAre(snippetSuffix("${1:(int)} fooey:${2:(unsigned int)}")));
}

TEST(CompletionTest, ObjectiveCMethodTwoArgumentsFromMiddle) {
  auto Results = completions(R"objc(
      @interface Foo
      + (id)fooWithValue:(int)value fooey:(unsigned int)fooey;
      @end
      id val = [Foo fooWithValue:10 f^]
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("fooey:")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(returnType("id")));
  EXPECT_THAT(C, ElementsAre(signature("(unsigned int)")));
  EXPECT_THAT(C, ElementsAre(snippetSuffix("${1:(unsigned int)}")));
}

TEST(CompletionTest, ObjectiveCMethodFilterOnEntireSelector) {
  auto Results = completions(R"objc(
      @interface Foo
      + (id)player:(id)player willRun:(id)run;
      @end
      id val = [Foo wi^]
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("player:")));
  EXPECT_THAT(C, ElementsAre(filterText("player:willRun:")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(returnType("id")));
  EXPECT_THAT(C, ElementsAre(signature("(id) willRun:(id)")));
  EXPECT_THAT(C, ElementsAre(snippetSuffix("${1:(id)} willRun:${2:(id)}")));
}

TEST(CompletionTest, ObjectiveCSimpleMethodDeclaration) {
  auto Results = completions(R"objc(
      @interface Foo
      - (void)foo;
      @end
      @implementation Foo
      fo^
      @end
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("foo")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(qualifier("- (void)")));
}

TEST(CompletionTest, ObjectiveCMethodDeclaration) {
  auto Results = completions(R"objc(
      @interface Foo
      - (int)valueForCharacter:(char)c secondArgument:(id)object;
      @end
      @implementation Foo
      valueFor^
      @end
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("valueForCharacter:")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(qualifier("- (int)")));
  EXPECT_THAT(C, ElementsAre(signature("(char)c secondArgument:(id)object")));
}

TEST(CompletionTest, ObjectiveCMethodDeclarationFilterOnEntireSelector) {
  auto Results = completions(R"objc(
      @interface Foo
      - (int)valueForCharacter:(char)c secondArgument:(id)object;
      @end
      @implementation Foo
      secondArg^
      @end
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("valueForCharacter:")));
  EXPECT_THAT(C, ElementsAre(filterText("valueForCharacter:secondArgument:")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(qualifier("- (int)")));
  EXPECT_THAT(C, ElementsAre(signature("(char)c secondArgument:(id)object")));
}

TEST(CompletionTest, ObjectiveCMethodDeclarationPrefixTyped) {
  auto Results = completions(R"objc(
      @interface Foo
      - (int)valueForCharacter:(char)c;
      @end
      @implementation Foo
      - (int)valueFor^
      @end
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("valueForCharacter:")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(signature("(char)c")));
}

TEST(CompletionTest, ObjectiveCMethodDeclarationFromMiddle) {
  auto Results = completions(R"objc(
      @interface Foo
      - (int)valueForCharacter:(char)c secondArgument:(id)object;
      @end
      @implementation Foo
      - (int)valueForCharacter:(char)c second^
      @end
    )objc",
                             /*IndexSymbols=*/{},
                             /*Opts=*/{}, "Foo.m");

  auto C = Results.Completions;
  EXPECT_THAT(C, ElementsAre(named("secondArgument:")));
  EXPECT_THAT(C, ElementsAre(kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(signature("(id)object")));
}

TEST(CompletionTest, ObjectiveCProtocolFromIndex) {
  Symbol FoodClass = objcClass("FoodClass");
  Symbol SymFood = objcProtocol("Food");
  Symbol SymFooey = objcProtocol("Fooey");
  auto Results = completions("id<Foo^>", {SymFood, FoodClass, SymFooey},
                             /*Opts=*/{}, "Foo.m");

  // Should only give protocols for ObjC protocol completions.
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(
                  AllOf(named("Food"), kind(CompletionItemKind::Interface)),
                  AllOf(named("Fooey"), kind(CompletionItemKind::Interface))));

  Results = completions("Fo^", {SymFood, FoodClass, SymFooey},
                        /*Opts=*/{}, "Foo.m");
  // Shouldn't give protocols for non protocol completions.
  EXPECT_THAT(
      Results.Completions,
      ElementsAre(AllOf(named("FoodClass"), kind(CompletionItemKind::Class))));
}

TEST(CompletionTest, ObjectiveCProtocolFromIndexSpeculation) {
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  auto File = testPath("Foo.m");
  Annotations Test(R"cpp(
      @protocol Food
      @end
      id<Foo$1^> foo;
      Foo$2^ bar;
  )cpp");
  runAddDocument(Server, File, Test.code());
  clangd::CodeCompleteOptions Opts = {};

  Symbol FoodClass = objcClass("FoodClass");
  IndexRequestCollector Requests({FoodClass});
  Opts.Index = &Requests;

  auto CompleteAtPoint = [&](StringRef P) {
    return cantFail(runCodeComplete(Server, File, Test.point(P), Opts))
        .Completions;
  };

  auto C = CompleteAtPoint("1");
  auto Reqs1 = Requests.consumeRequests(1);
  ASSERT_EQ(Reqs1.size(), 1u);
  EXPECT_THAT(C, ElementsAre(AllOf(named("Food"),
                                   kind(CompletionItemKind::Interface))));

  C = CompleteAtPoint("2");
  auto Reqs2 = Requests.consumeRequests(1);
  // Speculation succeeded. Used speculative index result, but filtering now to
  // now include FoodClass.
  ASSERT_EQ(Reqs2.size(), 1u);
  EXPECT_EQ(Reqs2[0], Reqs1[0]);
  EXPECT_THAT(C, ElementsAre(AllOf(named("FoodClass"),
                                   kind(CompletionItemKind::Class))));
}

TEST(CompletionTest, ObjectiveCCategoryFromIndexIgnored) {
  Symbol FoodCategory = objcCategory("FoodClass", "Extension");
  auto Results = completions(R"objc(
      @interface Foo
      @end
      @interface Foo (^)
      @end
    )objc",
                             {FoodCategory},
                             /*Opts=*/{}, "Foo.m");
  EXPECT_THAT(Results.Completions, IsEmpty());
}

TEST(CompletionTest, ObjectiveCForwardDeclFromIndex) {
  Symbol FoodClass = objcClass("FoodClass");
  FoodClass.IncludeHeaders.emplace_back("\"Foo.h\"", 2, Symbol::Import);
  Symbol SymFood = objcProtocol("Food");
  auto Results = completions("@class Foo^", {SymFood, FoodClass},
                             /*Opts=*/{}, "Foo.m");

  // Should only give class names without any include insertion.
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(named("FoodClass"),
                                         kind(CompletionItemKind::Class),
                                         Not(insertInclude()))));
}

TEST(CompletionTest, CursorInSnippets) {
  clangd::CodeCompleteOptions Options;
  Options.EnableSnippets = true;
  auto Results = completions(
      R"cpp(
    void while_foo(int a, int b);
    void test() {
      whil^
    })cpp",
      /*IndexSymbols=*/{}, Options);

  // Last placeholder in code patterns should be $0 to put the cursor there.
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(named("while"),
                             snippetSuffix(" (${1:condition}) {\n$0\n}"))));
  // However, snippets for functions must *not* end with $0.
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(named("while_foo"),
                             snippetSuffix("(${1:int a}, ${2:int b})"))));

  Results = completions(R"cpp(
    struct Base {
      Base(int a, int b) {}
    };

    struct Derived : Base {
      Derived() : Base^
    };
  )cpp",
                        /*IndexSymbols=*/{}, Options);
  // Constructors from base classes are a kind of pattern that shouldn't end
  // with $0.
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(named("Base"),
                             snippetSuffix("(${1:int a}, ${2:int b})"))));
}

TEST(CompletionTest, WorksWithNullType) {
  auto R = completions(R"cpp(
    int main() {
      for (auto [loopVar] : y ) { // y has to be unresolved.
        int z = loopV^;
      }
    }
  )cpp");
  EXPECT_THAT(R.Completions, ElementsAre(named("loopVar")));
}

TEST(CompletionTest, UsingDecl) {
  const char *Header(R"cpp(
    void foo(int);
    namespace std {
      using ::foo;
    })cpp");
  const char *Source(R"cpp(
    void bar() {
      std::^;
    })cpp");
  auto Index = TestTU::withHeaderCode(Header).index();
  clangd::CodeCompleteOptions Opts;
  Opts.Index = Index.get();
  Opts.AllScopes = true;
  auto R = completions(Source, {}, Opts);
  EXPECT_THAT(R.Completions,
              ElementsAre(AllOf(scope("std::"), named("foo"),
                                kind(CompletionItemKind::Reference))));
}

TEST(CompletionTest, Enums) {
  const char *Header(R"cpp(
    namespace ns {
      enum Unscoped { Clangd1 };
      class C {
      enum Unscoped { Clangd2 };
      };
      enum class Scoped { Clangd3 };
    })cpp");
  const char *Source(R"cpp(
    void bar() {
      Clangd^
    })cpp");
  auto Index = TestTU::withHeaderCode(Header).index();
  clangd::CodeCompleteOptions Opts;
  Opts.Index = Index.get();
  Opts.AllScopes = true;
  auto R = completions(Source, {}, Opts);
  EXPECT_THAT(R.Completions, UnorderedElementsAre(
                                 AllOf(scope("ns::"), named("Clangd1"),
                                       kind(CompletionItemKind::EnumMember)),
                                 AllOf(scope("ns::C::"), named("Clangd2"),
                                       kind(CompletionItemKind::EnumMember)),
                                 AllOf(scope("ns::Scoped::"), named("Clangd3"),
                                       kind(CompletionItemKind::EnumMember))));
}

TEST(CompletionTest, ScopeIsUnresolved) {
  clangd::CodeCompleteOptions Opts = {};
  Opts.AllScopes = true;

  auto Results = completions(R"cpp(
    namespace a {
    void f() { b::X^ }
    }
  )cpp",
                             {cls("a::b::XYZ")}, Opts);
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(qualifier(""), named("XYZ"))));
}

TEST(CompletionTest, NestedScopeIsUnresolved) {
  clangd::CodeCompleteOptions Opts = {};
  Opts.AllScopes = true;

  auto Results = completions(R"cpp(
    namespace a {
    namespace b {}
    void f() { b::c::X^ }
    }
  )cpp",
                             {cls("a::b::c::XYZ")}, Opts);
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(qualifier(""), named("XYZ"))));
}

// Clang parser gets confused here and doesn't report the ns:: prefix.
// Naive behavior is to insert it again. We examine the source and recover.
TEST(CompletionTest, NamespaceDoubleInsertion) {
  clangd::CodeCompleteOptions Opts = {};

  auto Results = completions(R"cpp(
    namespace foo {
    namespace ns {}
    #define M(X) < X
    M(ns::ABC^
    }
  )cpp",
                             {cls("foo::ns::ABCDE")}, Opts);
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(qualifier(""), named("ABCDE"))));
}

TEST(CompletionTest, DerivedMethodsAreAlwaysVisible) {
  // Despite the fact that base method matches the ref-qualifier better,
  // completion results should only include the derived method.
  auto Completions = completions(R"cpp(
    struct deque_base {
      float size();
      double size() const;
    };
    struct deque : deque_base {
        int size() const;
    };

    auto x = deque().^
  )cpp")
                         .Completions;
  EXPECT_THAT(Completions,
              ElementsAre(AllOf(returnType("int"), named("size"))));
}

TEST(CompletionTest, NoCrashWithIncompleteLambda) {
  auto Completions = completions("auto&& x = []{^").Completions;
  // The completion of x itself can cause a problem: in the code completion
  // callback, its type is not known, which affects the linkage calculation.
  // A bad linkage value gets cached, and subsequently updated.
  EXPECT_THAT(Completions, Contains(named("x")));

  auto Signatures = signatures("auto x() { x(^").signatures;
  EXPECT_THAT(Signatures, Contains(sig("x() -> auto")));
}

TEST(CompletionTest, DelayedTemplateParsing) {
  Annotations Test(R"cpp(
    int xxx;
    template <typename T> int foo() { return xx^; }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  // Even though delayed-template-parsing is on, we will disable it to provide
  // completion in templates.
  TU.ExtraArgs.push_back("-fdelayed-template-parsing");

  EXPECT_THAT(completions(TU, Test.point()).Completions,
              Contains(named("xxx")));
}

TEST(CompletionTest, CompletionRange) {
  const char *WithRange = "auto x = [[abc]]^";
  auto Completions = completions(WithRange);
  EXPECT_EQ(Completions.CompletionRange, Annotations(WithRange).range());
  Completions = completionsNoCompile(WithRange);
  EXPECT_EQ(Completions.CompletionRange, Annotations(WithRange).range());

  const char *EmptyRange = "auto x = [[]]^";
  Completions = completions(EmptyRange);
  EXPECT_EQ(Completions.CompletionRange, Annotations(EmptyRange).range());
  Completions = completionsNoCompile(EmptyRange);
  EXPECT_EQ(Completions.CompletionRange, Annotations(EmptyRange).range());

  // Sema doesn't trigger at all here, while the no-sema completion runs
  // heuristics as normal and reports a range. It'd be nice to be consistent.
  const char *NoCompletion = "/* foo [[]]^ */";
  Completions = completions(NoCompletion);
  EXPECT_EQ(Completions.CompletionRange, std::nullopt);
  Completions = completionsNoCompile(NoCompletion);
  EXPECT_EQ(Completions.CompletionRange, Annotations(NoCompletion).range());
}

TEST(NoCompileCompletionTest, Basic) {
  auto Results = completionsNoCompile(R"cpp(
    void func() {
      int xyz;
      int abc;
      ^
    }
  )cpp");
  EXPECT_FALSE(Results.RanParser);
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(named("void"), named("func"), named("int"),
                                   named("xyz"), named("abc")));
}

TEST(NoCompileCompletionTest, WithFilter) {
  auto Results = completionsNoCompile(R"cpp(
    void func() {
      int sym1;
      int sym2;
      int xyz1;
      int xyz2;
      sy^
    }
  )cpp");
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(named("sym1"), named("sym2")));
}

TEST(NoCompileCompletionTest, WithIndex) {
  std::vector<Symbol> Syms = {func("xxx"), func("a::xxx"), func("ns::b::xxx"),
                              func("c::xxx"), func("ns::d::xxx")};
  auto Results = completionsNoCompile(
      R"cpp(
        // Current-scopes, unqualified completion.
        using namespace a;
        namespace ns {
        using namespace b;
        void foo() {
        xx^
        }
        }
      )cpp",
      Syms);
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(qualifier(""), scope("")),
                                   AllOf(qualifier(""), scope("a::")),
                                   AllOf(qualifier(""), scope("ns::b::"))));
  CodeCompleteOptions Opts;
  Opts.AllScopes = true;
  Results = completionsNoCompile(
      R"cpp(
        // All-scopes unqualified completion.
        using namespace a;
        namespace ns {
        using namespace b;
        void foo() {
        xx^
        }
        }
      )cpp",
      Syms, Opts);
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(AllOf(qualifier(""), scope("")),
                                   AllOf(qualifier(""), scope("a::")),
                                   AllOf(qualifier(""), scope("ns::b::")),
                                   AllOf(qualifier("c::"), scope("c::")),
                                   AllOf(qualifier("d::"), scope("ns::d::"))));
  Results = completionsNoCompile(
      R"cpp(
        // Qualified completion.
        using namespace a;
        namespace ns {
        using namespace b;
        void foo() {
        b::xx^
        }
        }
      )cpp",
      Syms, Opts);
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(qualifier(""), scope("ns::b::"))));
  Results = completionsNoCompile(
      R"cpp(
        // Absolutely qualified completion.
        using namespace a;
        namespace ns {
        using namespace b;
        void foo() {
        ::a::xx^
        }
        }
      )cpp",
      Syms, Opts);
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(qualifier(""), scope("a::"))));
}

TEST(AllowImplicitCompletion, All) {
  const char *Yes[] = {
      "foo.^bar",
      "foo->^bar",
      "foo::^bar",
      "  #  include <^foo.h>",
      "#import <foo/^bar.h>",
      "#include_next \"^",
  };
  const char *No[] = {
      "foo>^bar",
      "foo:^bar",
      "foo\n^bar",
      "#include <foo.h> //^",
      "#include \"foo.h\"^",
      "#error <^",
      "#<^",
  };
  for (const char *Test : Yes) {
    llvm::Annotations A(Test);
    EXPECT_TRUE(allowImplicitCompletion(A.code(), A.point())) << Test;
  }
  for (const char *Test : No) {
    llvm::Annotations A(Test);
    EXPECT_FALSE(allowImplicitCompletion(A.code(), A.point())) << Test;
  }
}

TEST(CompletionTest, FunctionArgsExist) {
  clangd::CodeCompleteOptions Opts;
  Opts.EnableSnippets = true;
  std::string Context = R"cpp(
    #define MACRO(x)
    int foo(int A);
    int bar();
    struct Object {
      Object(int B) {}
    };
    template <typename T>
    struct Container {
      Container(int Size) {}
    };
  )cpp";
  EXPECT_THAT(completions(Context + "int y = fo^", {}, Opts).Completions,
              UnorderedElementsAre(
                  AllOf(labeled("foo(int A)"), snippetSuffix("(${1:int A})"))));
  EXPECT_THAT(
      completions(Context + "int y = fo^(42)", {}, Opts).Completions,
      UnorderedElementsAre(AllOf(labeled("foo(int A)"), snippetSuffix(""))));
  // FIXME(kirillbobyrev): No snippet should be produced here.
  EXPECT_THAT(completions(Context + "int y = fo^o(42)", {}, Opts).Completions,
              UnorderedElementsAre(
                  AllOf(labeled("foo(int A)"), snippetSuffix("(${1:int A})"))));
  EXPECT_THAT(
      completions(Context + "int y = ba^", {}, Opts).Completions,
      UnorderedElementsAre(AllOf(labeled("bar()"), snippetSuffix("()"))));
  EXPECT_THAT(completions(Context + "int y = ba^()", {}, Opts).Completions,
              UnorderedElementsAre(AllOf(labeled("bar()"), snippetSuffix(""))));
  EXPECT_THAT(
      completions(Context + "Object o = Obj^", {}, Opts).Completions,
      Contains(AllOf(labeled("Object(int B)"), snippetSuffix("(${1:int B})"),
                     kind(CompletionItemKind::Constructor))));
  EXPECT_THAT(completions(Context + "Object o = Obj^()", {}, Opts).Completions,
              Contains(AllOf(labeled("Object(int B)"), snippetSuffix(""),
                             kind(CompletionItemKind::Constructor))));
  EXPECT_THAT(
      completions(Context + "Container c = Cont^", {}, Opts).Completions,
      Contains(AllOf(labeled("Container<typename T>(int Size)"),
                     snippetSuffix("<${1:typename T}>(${2:int Size})"),
                     kind(CompletionItemKind::Constructor))));
  EXPECT_THAT(
      completions(Context + "Container c = Cont^()", {}, Opts).Completions,
      Contains(AllOf(labeled("Container<typename T>(int Size)"),
                     snippetSuffix("<${1:typename T}>"),
                     kind(CompletionItemKind::Constructor))));
  EXPECT_THAT(
      completions(Context + "Container c = Cont^<int>()", {}, Opts).Completions,
      Contains(AllOf(labeled("Container<typename T>(int Size)"),
                     snippetSuffix(""),
                     kind(CompletionItemKind::Constructor))));
  EXPECT_THAT(completions(Context + "MAC^(2)", {}, Opts).Completions,
              Contains(AllOf(labeled("MACRO(x)"), snippetSuffix(""),
                             kind(CompletionItemKind::Function))));
}

TEST(CompletionTest, NoCrashDueToMacroOrdering) {
  EXPECT_THAT(completions(R"cpp(
    #define ECHO(X) X
    #define ECHO2(X) ECHO(X)
    int finish_preamble = EC^HO(2);)cpp")
                  .Completions,
              UnorderedElementsAre(labeled("ECHO(X)"), labeled("ECHO2(X)")));
}

TEST(CompletionTest, ObjCCategoryDecls) {
  TestTU TU;
  TU.ExtraArgs.push_back("-xobjective-c");
  TU.HeaderCode = R"objc(
  @interface Foo
  @end

  @interface Foo (FooExt1)
  @end

  @interface Foo (FooExt2)
  @end

  @interface Bar
  @end

  @interface Bar (BarExt)
  @end)objc";

  {
    Annotations Test(R"objc(
  @implementation Foo (^)
  @end
  )objc");
    TU.Code = Test.code().str();
    auto Results = completions(TU, Test.point());
    EXPECT_THAT(Results.Completions,
                UnorderedElementsAre(labeled("FooExt1"), labeled("FooExt2")));
  }
  {
    Annotations Test(R"objc(
  @interface Foo (^)
  @end
  )objc");
    TU.Code = Test.code().str();
    auto Results = completions(TU, Test.point());
    EXPECT_THAT(Results.Completions, UnorderedElementsAre(labeled("BarExt")));
  }
}

TEST(CompletionTest, PreambleCodeComplete) {
  llvm::StringLiteral Baseline = "\n#define MACRO 12\nint num = MACRO;";
  llvm::StringLiteral ModifiedCC =
      "#include \"header.h\"\n#define MACRO 12\nint num = MACRO; int num2 = M^";

  Annotations Test(ModifiedCC);
  auto BaselineTU = TestTU::withCode(Baseline);
  auto ModifiedTU = TestTU::withCode(Test.code());

  MockFS FS;
  auto Inputs = ModifiedTU.inputs(FS);
  auto Result = codeComplete(testPath(ModifiedTU.Filename), Test.point(),
                             BaselineTU.preamble().get(), Inputs, {});
  EXPECT_THAT(Result.Completions, Not(testing::IsEmpty()));
}

TEST(CompletionTest, CommentParamName) {
  const std::string Code = R"cpp(
    void fun(int foo, int bar);
    void overloaded(int param_int);
    void overloaded(int param_int, int param_other);
    void overloaded(char param_char);
    int main() {
  )cpp";

  EXPECT_THAT(completions(Code + "fun(/*^").Completions,
              UnorderedElementsAre(labeled("foo=*/")));
  EXPECT_THAT(completions(Code + "fun(1, /*^").Completions,
              UnorderedElementsAre(labeled("bar=*/")));
  EXPECT_THAT(completions(Code + "/*^").Completions, IsEmpty());
  // Test de-duplication.
  EXPECT_THAT(
      completions(Code + "overloaded(/*^").Completions,
      UnorderedElementsAre(labeled("param_int=*/"), labeled("param_char=*/")));
  // Comment already has some text in it.
  EXPECT_THAT(completions(Code + "fun(/*  ^").Completions,
              UnorderedElementsAre(labeled("foo=*/")));
  EXPECT_THAT(completions(Code + "fun(/* f^").Completions,
              UnorderedElementsAre(labeled("foo=*/")));
  EXPECT_THAT(completions(Code + "fun(/* x^").Completions, IsEmpty());
  EXPECT_THAT(completions(Code + "fun(/* f ^").Completions, IsEmpty());

  // Test ranges
  {
    std::string CompletionRangeTest(Code + "fun(/*[[^]]");
    auto Results = completions(CompletionRangeTest);
    EXPECT_THAT(Results.CompletionRange,
                llvm::ValueIs(Annotations(CompletionRangeTest).range()));
    EXPECT_THAT(
        Results.Completions,
        testing::Each(
            AllOf(replacesRange(Annotations(CompletionRangeTest).range()),
                  origin(SymbolOrigin::AST), kind(CompletionItemKind::Text))));
  }
  {
    std::string CompletionRangeTest(Code + "fun(/*[[fo^]]");
    auto Results = completions(CompletionRangeTest);
    EXPECT_THAT(Results.CompletionRange,
                llvm::ValueIs(Annotations(CompletionRangeTest).range()));
    EXPECT_THAT(
        Results.Completions,
        testing::Each(
            AllOf(replacesRange(Annotations(CompletionRangeTest).range()),
                  origin(SymbolOrigin::AST), kind(CompletionItemKind::Text))));
  }
}

TEST(CompletionTest, Concepts) {
  Annotations Code(R"cpp(
    template<class T>
    concept A = sizeof(T) <= 8;

    template<$tparam^A U>
    int foo();

    template<typename T>
    int bar(T t) requires $expr^A<int>;

    template<class T>
    concept b = $expr^A && $expr^sizeof(T) % 2 == 0 || $expr^A && sizeof(T) == 1;

    $toplevel^A auto i = 19;

    template<$toplevel^A auto i> void constrainedNTTP();

    // FIXME: The first parameter should be dropped in this case.
    void abbreviated($expr^A auto x) {}
  )cpp");
  TestTU TU;
  TU.Code = Code.code().str();
  TU.ExtraArgs = {"-std=c++20"};

  auto Sym = conceptSym("same_as");
  Sym.Signature = "<typename Tp, typename Up>";
  Sym.CompletionSnippetSuffix = "<${1:typename Tp}, ${2:typename Up}>";
  std::vector<Symbol> Syms = {Sym};
  for (auto P : Code.points("tparam")) {
    ASSERT_THAT(
        completions(TU, P, Syms).Completions,
        AllOf(Contains(AllOf(named("A"), signature(""), snippetSuffix(""))),
              Contains(AllOf(named("same_as"), signature("<typename Up>"),
                             snippetSuffix("<${2:typename Up}>"))),
              Contains(named("class")), Contains(named("typename"))))
        << "Completing template parameter at position " << P;
  }

  for (auto P : Code.points("toplevel")) {
    EXPECT_THAT(
        completions(TU, P, Syms).Completions,
        AllOf(Contains(AllOf(named("A"), signature(""), snippetSuffix(""))),
              Contains(AllOf(named("same_as"), signature("<typename Up>"),
                             snippetSuffix("<${2:typename Up}>")))))
        << "Completing 'requires' expression at position " << P;
  }

  for (auto P : Code.points("expr")) {
    EXPECT_THAT(
        completions(TU, P, Syms).Completions,
        AllOf(Contains(AllOf(named("A"), signature("<class T>"),
                             snippetSuffix("<${1:class T}>"))),
              Contains(AllOf(
                  named("same_as"), signature("<typename Tp, typename Up>"),
                  snippetSuffix("<${1:typename Tp}, ${2:typename Up}>")))))
        << "Completing 'requires' expression at position " << P;
  }
}

TEST(SignatureHelp, DocFormat) {
  Annotations Code(R"cpp(
    // Comment `with` markup.
    void foo(int);
    void bar() { foo(^); }
  )cpp");
  for (auto DocumentationFormat :
       {MarkupKind::PlainText, MarkupKind::Markdown}) {
    auto Sigs = signatures(Code.code(), Code.point(), /*IndexSymbols=*/{},
                           DocumentationFormat);
    ASSERT_EQ(Sigs.signatures.size(), 1U);
    EXPECT_EQ(Sigs.signatures[0].documentation.kind, DocumentationFormat);
  }
}

TEST(SignatureHelp, TemplateArguments) {
  std::string Top = R"cpp(
    template <typename T, int> bool foo(char);
    template <int I, int> bool foo(float);
  )cpp";

  auto First = signatures(Top + "bool x = foo<^");
  EXPECT_THAT(
      First.signatures,
      UnorderedElementsAre(sig("foo<[[typename T]], [[int]]>() -> bool"),
                           sig("foo<[[int I]], [[int]]>() -> bool")));
  EXPECT_EQ(First.activeParameter, 0);

  auto Second = signatures(Top + "bool x = foo<1, ^");
  EXPECT_THAT(Second.signatures,
              ElementsAre(sig("foo<[[int I]], [[int]]>() -> bool")));
  EXPECT_EQ(Second.activeParameter, 1);
}

TEST(CompletionTest, DoNotCrash) {
  llvm::StringLiteral Cases[] = {
      R"cpp(
    template <typename = int> struct Foo {};
    auto a = [x(3)](Foo<^>){};
    )cpp",
  };
  for (auto Case : Cases) {
    SCOPED_TRACE(Case);
    auto Completions = completions(Case);
  }
}

} // namespace
} // namespace clangd
} // namespace clang
