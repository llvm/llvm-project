//===-- CodeCompleteTests.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdServer.h"
#include "CodeComplete.h"
#include "Compiler.h"
#include "Matchers.h"
#include "Protocol.h"
#include "Quality.h"
#include "SourceCode.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "Threading.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <condition_variable>
#include <mutex>

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

class IgnoreDiagnostics : public DiagnosticsConsumer {
  void onDiagnosticsReady(PathRef File,
                          std::vector<Diag> Diagnostics) override {}
};

// GMock helpers for matching completion items.
MATCHER_P(Named, Name, "") { return arg.Name == Name; }
MATCHER_P(NameStartsWith, Prefix, "") {
  return llvm::StringRef(arg.Name).startswith(Prefix);
}
MATCHER_P(Scope, S, "") { return arg.Scope == S; }
MATCHER_P(Qualifier, Q, "") { return arg.RequiredQualifier == Q; }
MATCHER_P(Labeled, Label, "") {
  return arg.RequiredQualifier + arg.Name + arg.Signature == Label;
}
MATCHER_P(SigHelpLabeled, Label, "") { return arg.label == Label; }
MATCHER_P(Kind, K, "") { return arg.Kind == K; }
MATCHER_P(Doc, D, "") { return arg.Documentation == D; }
MATCHER_P(ReturnType, D, "") { return arg.ReturnType == D; }
MATCHER_P(HasInclude, IncludeHeader, "") {
  return !arg.Includes.empty() && arg.Includes[0].Header == IncludeHeader;
}
MATCHER_P(InsertInclude, IncludeHeader, "") {
  return !arg.Includes.empty() && arg.Includes[0].Header == IncludeHeader &&
         bool(arg.Includes[0].Insertion);
}
MATCHER(InsertInclude, "") {
  return !arg.Includes.empty() && bool(arg.Includes[0].Insertion);
}
MATCHER_P(SnippetSuffix, Text, "") { return arg.SnippetSuffix == Text; }
MATCHER_P(Origin, OriginSet, "") { return arg.Origin == OriginSet; }
MATCHER_P(Signature, S, "") { return arg.Signature == S; }

// Shorthand for Contains(Named(Name)).
Matcher<const std::vector<CodeCompletion> &> Has(std::string Name) {
  return Contains(Named(std::move(Name)));
}
Matcher<const std::vector<CodeCompletion> &> Has(std::string Name,
                                                 CompletionItemKind K) {
  return Contains(AllOf(Named(std::move(Name)), Kind(K)));
}
MATCHER(IsDocumented, "") { return !arg.Documentation.empty(); }
MATCHER(Deprecated, "") { return arg.Deprecated; }

std::unique_ptr<SymbolIndex> memIndex(std::vector<Symbol> Symbols) {
  SymbolSlab::Builder Slab;
  for (const auto &Sym : Symbols)
    Slab.insert(Sym);
  return MemIndex::build(std::move(Slab).build(), RefSlab(), RelationSlab());
}

CodeCompleteResult completions(ClangdServer &Server, llvm::StringRef TestCode,
                               Position Point,
                               std::vector<Symbol> IndexSymbols = {},
                               clangd::CodeCompleteOptions Opts = {}) {
  std::unique_ptr<SymbolIndex> OverrideIndex;
  if (!IndexSymbols.empty()) {
    assert(!Opts.Index && "both Index and IndexSymbols given!");
    OverrideIndex = memIndex(std::move(IndexSymbols));
    Opts.Index = OverrideIndex.get();
  }

  auto File = testPath("foo.cpp");
  runAddDocument(Server, File, TestCode);
  auto CompletionList =
      llvm::cantFail(runCodeComplete(Server, File, Point, Opts));
  return CompletionList;
}

CodeCompleteResult completions(ClangdServer &Server, llvm::StringRef Text,
                               std::vector<Symbol> IndexSymbols = {},
                               clangd::CodeCompleteOptions Opts = {},
                               PathRef FilePath = "foo.cpp") {
  std::unique_ptr<SymbolIndex> OverrideIndex;
  if (!IndexSymbols.empty()) {
    assert(!Opts.Index && "both Index and IndexSymbols given!");
    OverrideIndex = memIndex(std::move(IndexSymbols));
    Opts.Index = OverrideIndex.get();
  }

  auto File = testPath(FilePath);
  Annotations Test(Text);
  runAddDocument(Server, File, Test.code());
  auto CompletionList =
      llvm::cantFail(runCodeComplete(Server, File, Test.point(), Opts));
  return CompletionList;
}

// Builds a server and runs code completion.
// If IndexSymbols is non-empty, an index will be built and passed to opts.
CodeCompleteResult completions(llvm::StringRef Text,
                               std::vector<Symbol> IndexSymbols = {},
                               clangd::CodeCompleteOptions Opts = {},
                               PathRef FilePath = "foo.cpp") {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  // To make sure our tests for completiopns inside templates work on Windows.
  CDB.ExtraClangFlags = {"-fno-delayed-template-parsing"};
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
  return completions(Server, Text, std::move(IndexSymbols), std::move(Opts),
                     FilePath);
}

// Builds a server and runs code completion.
// If IndexSymbols is non-empty, an index will be built and passed to opts.
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

  MockFSProvider FS;
  Annotations Test(Text);
  return codeComplete(FilePath, tooling::CompileCommand(), /*Preamble=*/nullptr,
                      Test.code(), Test.point(), FS.getFileSystem(), Opts);
}

Symbol withReferences(int N, Symbol S) {
  S.References = N;
  return S;
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
  EXPECT_THAT(Results.Completions, ElementsAre(Named("AAA"), Named("BBB")));
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
              AllOf(Has("FooBar"), Has("FooBaz"), Not(Has("Qux"))));

  // Macros require  prefix match.
  EXPECT_THAT(completions(Body + "int main() { C^ }").Completions,
              AllOf(Has("Car"), Not(Has("MotorCar"))));
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
        /// Doc for method.
        int method();

        int field;
      private:
        int private_field;
      };

      int test() {
        struct LocalClass {};

        /// Doc for local_var.
        int local_var;

        ClassWithMembers().^
      }
      )cpp",
      {cls("IndexClass"), var("index_var"), func("index_func")}, Opts);

  EXPECT_TRUE(Results.RanParser);
  // Class members. The only items that must be present in after-dot
  // completion.
  EXPECT_THAT(Results.Completions,
              AllOf(Has("method"), Has("field"), Not(Has("ClassWithMembers")),
                    Not(Has("operator=")), Not(Has("~ClassWithMembers"))));
  EXPECT_IFF(Opts.IncludeIneligibleResults, Results.Completions,
             Has("private_field"));
  // Global items.
  EXPECT_THAT(
      Results.Completions,
      Not(AnyOf(Has("global_var"), Has("index_var"), Has("global_func"),
                Has("global_func()"), Has("index_func"), Has("GlobalClass"),
                Has("IndexClass"), Has("MACRO"), Has("LocalClass"))));
  // There should be no code patterns (aka snippets) in after-dot
  // completion. At least there aren't any we're aware of.
  EXPECT_THAT(Results.Completions,
              Not(Contains(Kind(CompletionItemKind::Snippet))));
  // Check documentation.
  EXPECT_IFF(Opts.IncludeComments, Results.Completions,
             Contains(IsDocumented()));
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
        /// Doc for method.
        int method();
      };

      int test() {
        struct LocalClass {};

        /// Doc for local_var.
        int local_var;

        ^
      }
      )cpp",
      {cls("IndexClass"), var("index_var"), func("index_func")}, Opts);

  EXPECT_TRUE(Results.RanParser);
  // Class members. Should never be present in global completions.
  EXPECT_THAT(Results.Completions,
              Not(AnyOf(Has("method"), Has("method()"), Has("field"))));
  // Global items.
  EXPECT_THAT(Results.Completions,
              AllOf(Has("global_var"), Has("index_var"), Has("global_func"),
                    Has("index_func" /* our fake symbol doesn't include () */),
                    Has("GlobalClass"), Has("IndexClass")));
  // A macro.
  EXPECT_IFF(Opts.IncludeMacros, Results.Completions, Has("MACRO"));
  // Local items. Must be present always.
  EXPECT_THAT(Results.Completions,
              AllOf(Has("local_var"), Has("LocalClass"),
                    Contains(Kind(CompletionItemKind::Snippet))));
  // Check documentation.
  EXPECT_IFF(Opts.IncludeComments, Results.Completions,
             Contains(IsDocumented()));
}

TEST(CompletionTest, CompletionOptions) {
  auto Test = [&](const clangd::CodeCompleteOptions &Opts) {
    testAfterDotCompletion(Opts);
    testGlobalScopeCompletion(Opts);
  };
  // We used to test every combination of options, but that got too slow (2^N).
  auto Flags = {
      &clangd::CodeCompleteOptions::IncludeMacros,
      &clangd::CodeCompleteOptions::IncludeComments,
      &clangd::CodeCompleteOptions::IncludeCodePatterns,
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
              AllOf(Has("priv"), Has("prot"), Has("pub")));

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
              AllOf(Has("pub"), Not(Has("prot")), Not(Has("priv"))));
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
              Contains(AllOf(Qualifier(""), Named("bar"))));
  // Hidden members are not shown.
  EXPECT_THAT(Results.Completions,
              Not(Contains(AllOf(Qualifier("Foo::"), Named("foo")))));
  // Private members are not shown.
  EXPECT_THAT(Results.Completions,
              Not(Contains(AllOf(Qualifier(""), Named("foo")))));
}

TEST(CompletionTest, InjectedTypename) {
  // These are suppressed when accessed as a member...
  EXPECT_THAT(completions("struct X{}; void foo(){ X().^ }").Completions,
              Not(Has("X")));
  EXPECT_THAT(completions("struct X{ void foo(){ this->^ } };").Completions,
              Not(Has("X")));
  // ...but accessible in other, more useful cases.
  EXPECT_THAT(completions("struct X{ void foo(){ ^ } };").Completions,
              Has("X"));
  EXPECT_THAT(
      completions("struct Y{}; struct X:Y{ void foo(){ ^ } };").Completions,
      Has("Y"));
  EXPECT_THAT(
      completions(
          "template<class> struct Y{}; struct X:Y<int>{ void foo(){ ^ } };")
          .Completions,
      Has("Y"));
  // This case is marginal (`using X::X` is useful), we allow it for now.
  EXPECT_THAT(completions("struct X{}; void foo(){ X::^ }").Completions,
              Has("X"));
}

TEST(CompletionTest, SkipInjectedWhenUnqualified) {
  EXPECT_THAT(completions("struct X { void f() { X^ }};").Completions,
              ElementsAre(Named("X"), Named("~X")));
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
      HasSubsequence(Named("a"),
                     SnippetSuffix("(${1:int i}, ${2:const float f})")));
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
              ElementsAre(AllOf(Named("func"), Labeled("func(int a, int b)"),
                                SnippetSuffix(""))));

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
              ElementsAre(AllOf(Named("func"), Labeled("func(int a, int b)"),
                                SnippetSuffix(""))));

  // Check all-scopes completions too.
  Opts.AllScopes = true;
  Results = completions(R"cpp(
      using ^;
  )cpp",
                        /*IndexSymbols=*/{Func}, Opts);
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(Named("func"), Labeled("ns::func(int a, int b)"),
                             SnippetSuffix(""))));
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
              AllOf(Has("function", CompletionItemKind::Function),
                    Has("variable", CompletionItemKind::Variable),
                    Has("int", CompletionItemKind::Keyword),
                    Has("Struct", CompletionItemKind::Class),
                    Has("MACRO", CompletionItemKind::Text),
                    Has("indexFunction", CompletionItemKind::Function),
                    Has("indexVariable", CompletionItemKind::Variable),
                    Has("indexClass", CompletionItemKind::Class)));

  Results = completions("nam^");
  EXPECT_THAT(Results.Completions,
              Has("namespace", CompletionItemKind::Snippet));

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
      UnorderedElementsAre(AllOf(Named("a"), Kind(CompletionItemKind::Field))));

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
          AllOf(Named("complete_class"), Kind(CompletionItemKind::Class)),
          AllOf(Named("complete_function"), Kind(CompletionItemKind::Function)),
          AllOf(Named("complete_type_alias"),
                Kind(CompletionItemKind::Interface)),
          AllOf(Named("complete_variable"), Kind(CompletionItemKind::Variable)),
          AllOf(Named("complete_static_member"),
                Kind(CompletionItemKind::Property))));
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
  EXPECT_THAT(Results.Completions, ElementsAre(Named("Adapter")));
}

TEST(CompletionTest, ScopedNoIndex) {
  auto Results = completions(
      R"cpp(
          namespace fake { int BigBang, Babble, Box; };
          int main() { fake::ba^ }
      ")cpp");
  // Babble is a better match than BigBang. Box doesn't match at all.
  EXPECT_THAT(Results.Completions,
              ElementsAre(Named("Babble"), Named("BigBang")));
}

TEST(CompletionTest, Scoped) {
  auto Results = completions(
      R"cpp(
          namespace fake { int Babble, Box; };
          int main() { fake::ba^ }
      ")cpp",
      {var("fake::BigBang")});
  EXPECT_THAT(Results.Completions,
              ElementsAre(Named("Babble"), Named("BigBang")));
}

TEST(CompletionTest, ScopedWithFilter) {
  auto Results = completions(
      R"cpp(
          void f() { ns::x^ }
      )cpp",
      {cls("ns::XYZ"), func("ns::foo")});
  EXPECT_THAT(Results.Completions, UnorderedElementsAre(Named("XYZ")));
}

TEST(CompletionTest, ReferencesAffectRanking) {
  auto Results = completions("int main() { abs^ }", {ns("absl"), func("absb")});
  EXPECT_THAT(Results.Completions,
              HasSubsequence(Named("absb"), Named("absl")));
  Results = completions("int main() { abs^ }",
                        {withReferences(10000, ns("absl")), func("absb")});
  EXPECT_THAT(Results.Completions,
              HasSubsequence(Named("absl"), Named("absb")));
}

TEST(CompletionTest, ContextWords) {
  auto Results = completions(R"cpp(
  enum class Color { RED, YELLOW, BLUE };

  // (blank lines so the definition above isn't "context")

  // "It was a yellow car," he said. "Big yellow car, new."
  auto Finish = Color::^
  )cpp");
  // Yellow would normally sort last (alphabetic).
  // But the recent mention shuold bump it up.
  ASSERT_THAT(Results.Completions,
              HasSubsequence(Named("YELLOW"), Named("BLUE")));
}

TEST(CompletionTest, GlobalQualified) {
  auto Results = completions(
      R"cpp(
          void f() { ::^ }
      )cpp",
      {cls("XYZ")});
  EXPECT_THAT(Results.Completions,
              AllOf(Has("XYZ", CompletionItemKind::Class),
                    Has("f", CompletionItemKind::Function)));
}

TEST(CompletionTest, FullyQualified) {
  auto Results = completions(
      R"cpp(
          namespace ns { void bar(); }
          void f() { ::ns::^ }
      )cpp",
      {cls("ns::XYZ")});
  EXPECT_THAT(Results.Completions,
              AllOf(Has("XYZ", CompletionItemKind::Class),
                    Has("bar", CompletionItemKind::Function)));
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
                  AllOf(Named("local"), Origin(SymbolOrigin::AST)),
                  AllOf(Named("Index"), Origin(SymbolOrigin::Static)),
                  AllOf(Named("both"),
                        Origin(SymbolOrigin::AST | SymbolOrigin::Static))));
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
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  std::string Subdir = testPath("sub");
  std::string SearchDirArg = (Twine("-I") + Subdir).str();
  CDB.ExtraClangFlags = {SearchDirArg.c_str()};
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = "";

  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
  auto BarURI = URI::create(BarHeader).toString();
  Symbol Sym = cls("ns::X");
  Sym.CanonicalDeclaration.FileURI = BarURI.c_str();
  Sym.IncludeHeaders.emplace_back(BarURI, 1);
  // Shoten include path based on search dirctory and insert.
  auto Results = completions(Server,
                             R"cpp(
          int main() { ns::^ }
      )cpp",
                             {Sym});
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(Named("X"), InsertInclude("\"bar.h\""))));
  // Can be disabled via option.
  CodeCompleteOptions NoInsertion;
  NoInsertion.InsertIncludes = CodeCompleteOptions::NeverInsert;
  Results = completions(Server,
                        R"cpp(
          int main() { ns::^ }
      )cpp",
                        {Sym}, NoInsertion);
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(Named("X"), Not(InsertInclude()))));
  // Duplicate based on inclusions in preamble.
  Results = completions(Server,
                        R"cpp(
          #include "sub/bar.h"  // not shortest, so should only match resolved.
          int main() { ns::^ }
      )cpp",
                        {Sym});
  EXPECT_THAT(Results.Completions, ElementsAre(AllOf(Named("X"), Labeled("X"),
                                                     Not(InsertInclude()))));
}

TEST(CompletionTest, NoIncludeInsertionWhenDeclFoundInFile) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;

  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
  Symbol SymX = cls("ns::X");
  Symbol SymY = cls("ns::Y");
  std::string BarHeader = testPath("bar.h");
  auto BarURI = URI::create(BarHeader).toString();
  SymX.CanonicalDeclaration.FileURI = BarURI.c_str();
  SymY.CanonicalDeclaration.FileURI = BarURI.c_str();
  SymX.IncludeHeaders.emplace_back("<bar>", 1);
  SymY.IncludeHeaders.emplace_back("<bar>", 1);
  // Shoten include path based on search dirctory and insert.
  auto Results = completions(Server,
                             R"cpp(
          namespace ns {
            class X;
            class Y {};
          }
          int main() { ns::^ }
      )cpp",
                             {SymX, SymY});
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(Named("X"), Not(InsertInclude())),
                          AllOf(Named("Y"), Not(InsertInclude()))));
}

TEST(CompletionTest, IndexSuppressesPreambleCompletions) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

  FS.Files[testPath("bar.h")] =
      R"cpp(namespace ns { struct preamble { int member; }; })cpp";
  auto File = testPath("foo.cpp");
  Annotations Test(R"cpp(
      #include "bar.h"
      namespace ns { int local; }
      void f() { ns::^; }
      void f2() { ns::preamble().$2^; }
  )cpp");
  runAddDocument(Server, File, Test.code());
  clangd::CodeCompleteOptions Opts = {};

  auto I = memIndex({var("ns::index")});
  Opts.Index = I.get();
  auto WithIndex = cantFail(runCodeComplete(Server, File, Test.point(), Opts));
  EXPECT_THAT(WithIndex.Completions,
              UnorderedElementsAre(Named("local"), Named("index")));
  auto ClassFromPreamble =
      cantFail(runCodeComplete(Server, File, Test.point("2"), Opts));
  EXPECT_THAT(ClassFromPreamble.Completions, Contains(Named("member")));

  Opts.Index = nullptr;
  auto WithoutIndex =
      cantFail(runCodeComplete(Server, File, Test.point(), Opts));
  EXPECT_THAT(WithoutIndex.Completions,
              UnorderedElementsAre(Named("local"), Named("preamble")));
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
  EXPECT_THAT(Results, ElementsAre(Named("ifndef")));
}

TEST(CompletionTest, DynamicIndexIncludeInsertion) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer::Options Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true;
  ClangdServer Server(CDB, FS, DiagConsumer, Opts);

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
  EXPECT_THAT(completions(Server, "Foo^ foo;").Completions,
              ElementsAre(AllOf(Named("Foo"), HasInclude("\"foo_header.h\""),
                                InsertInclude())));
}

TEST(CompletionTest, DynamicIndexMultiFile) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  auto Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true;
  ClangdServer Server(CDB, FS, DiagConsumer, Opts);

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
  EXPECT_THAT(Results.Completions, Has("XYZ", CompletionItemKind::Class));
  EXPECT_THAT(Results.Completions, Has("foo", CompletionItemKind::Function));
  EXPECT_THAT(Results.Completions, Has("XXX", CompletionItemKind::Class));
  EXPECT_THAT(Results.Completions,
              Contains((Named("fooooo"), Kind(CompletionItemKind::Function),
                        Doc("Doooc"), ReturnType("void"))));
}

TEST(CompletionTest, Documentation) {
  auto Results = completions(
      R"cpp(
      // Non-doxygen comment.
      int foo();
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
              Contains(AllOf(Named("foo"), Doc("Non-doxygen comment."))));
  EXPECT_THAT(
      Results.Completions,
      Contains(AllOf(Named("bar"), Doc("Doxygen comment.\n\\param int a"))));
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(Named("baz"), Doc("Multi-line\nblock comment"))));
}

TEST(CompletionTest, CommentsFromSystemHeaders) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;

  auto Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true;

  ClangdServer Server(CDB, FS, DiagConsumer, Opts);

  FS.Files[testPath("foo.h")] = R"cpp(
    #pragma GCC system_header

    // This comment should be retained!
    int foo();
  )cpp";

  auto Results = completions(Server,
                             R"cpp(
#include "foo.h"
int x = foo^
     )cpp");
  EXPECT_THAT(
      Results.Completions,
      Contains(AllOf(Named("foo"), Doc("This comment should be retained!"))));
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

  EXPECT_THAT(Results.Completions, Contains(Labeled("clang")));
  EXPECT_THAT(Results.Completions, Not(Contains(Labeled("clang::"))));
}

TEST(CompletionTest, BacktrackCrashes) {
  // Sema calls code completion callbacks twice in these cases.
  auto Results = completions(R"cpp(
      namespace ns {
      struct FooBarBaz {};
      } // namespace ns

     int foo(ns::FooBar^
  )cpp");

  EXPECT_THAT(Results.Completions, ElementsAre(Labeled("FooBarBaz")));

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
              UnorderedElementsAre(Named("X"), Named("Y")));
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

  EXPECT_THAT(Results.Completions, Contains(Named("X")));
}

TEST(CompletionTest, IgnoreCompleteInExcludedPPBranchWithRecoveryContext) {
  auto Results = completions(R"cpp(
    int bar(int param_in_bar) {
    }

    int foo(int param_in_foo) {
#if 0
  // In recorvery mode, "param_in_foo" will also be suggested among many other
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
              UnorderedElementsAre(Labeled("X(int A = 0)")));
  EXPECT_THAT(completions(Context + "int y = Y^", {}, Opts).Completions,
              UnorderedElementsAre(AllOf(Labeled("Y(int A, int B = 0)"),
                                         SnippetSuffix("(${1:int A})"))));
  EXPECT_THAT(completions(Context + "int y = Z^", {}, Opts).Completions,
              UnorderedElementsAre(
                  AllOf(Labeled("Z(int A, int B = 0, int C = 0, int D = 0)"),
                        SnippetSuffix("(${1:int A})"))));
}

TEST(CompletionTest, NoCrashWithTemplateParamsAndPreferredTypes) {
  auto Completions = completions(R"cpp(
template <template <class> class TT> int foo() {
  int a = ^
}
)cpp")
                         .Completions;
  EXPECT_THAT(Completions, Contains(Named("TT")));
}

SignatureHelp signatures(llvm::StringRef Text, Position Point,
                         std::vector<Symbol> IndexSymbols = {}) {
  std::unique_ptr<SymbolIndex> Index;
  if (!IndexSymbols.empty())
    Index = memIndex(IndexSymbols);

  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer::Options Opts = ClangdServer::optsForTest();
  Opts.StaticIndex = Index.get();

  ClangdServer Server(CDB, FS, DiagConsumer, Opts);
  auto File = testPath("foo.cpp");
  runAddDocument(Server, File, Text);
  return llvm::cantFail(runSignatureHelp(Server, File, Point));
}

SignatureHelp signatures(llvm::StringRef Text,
                         std::vector<Symbol> IndexSymbols = {}) {
  Annotations Test(Text);
  return signatures(Test.code(), Test.point(), std::move(IndexSymbols));
}

struct ExpectedParameter {
  std::string Text;
  std::pair<unsigned, unsigned> Offsets;
};
MATCHER_P(ParamsAre, P, "") {
  if (P.size() != arg.parameters.size())
    return false;
  for (unsigned I = 0; I < P.size(); ++I) {
    if (P[I].Text != arg.parameters[I].labelString ||
        P[I].Offsets != arg.parameters[I].labelOffsets)
      return false;
  }
  return true;
}
MATCHER_P(SigDoc, Doc, "") { return arg.documentation == Doc; }

/// \p AnnotatedLabel is a signature label with ranges marking parameters, e.g.
///    foo([[int p1]], [[double p2]]) -> void
Matcher<SignatureInformation> Sig(llvm::StringRef AnnotatedLabel) {
  llvm::Annotations A(AnnotatedLabel);
  std::string Label = A.code();
  std::vector<ExpectedParameter> Parameters;
  for (auto Range : A.ranges()) {
    Parameters.emplace_back();

    ExpectedParameter &P = Parameters.back();
    P.Text = Label.substr(Range.Begin, Range.End - Range.Begin);
    P.Offsets.first = lspLength(llvm::StringRef(Label).substr(0, Range.Begin));
    P.Offsets.second = lspLength(llvm::StringRef(Label).substr(1, Range.End));
  }
  return AllOf(SigHelpLabeled(Label), ParamsAre(Parameters));
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
              UnorderedElementsAre(Sig("foo([[float x]], [[float y]]) -> void"),
                                   Sig("foo([[float x]], [[int y]]) -> void"),
                                   Sig("foo([[int x]], [[float y]]) -> void"),
                                   Sig("foo([[int x]], [[int y]]) -> void")));
  // We always prefer the first signature.
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(0, Results.activeParameter);
}

TEST(SignatureHelpTest, DefaultArgs) {
  auto Results = signatures(R"cpp(
    void bar(int x, int y = 0);
    void bar(float x = 0, int y = 42);
    int main() { bar(^
  )cpp");
  EXPECT_THAT(Results.signatures,
              UnorderedElementsAre(
                  Sig("bar([[int x]], [[int y = 0]]) -> void"),
                  Sig("bar([[float x = 0]], [[int y = 42]]) -> void")));
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(0, Results.activeParameter);
}

TEST(SignatureHelpTest, ActiveArg) {
  auto Results = signatures(R"cpp(
    int baz(int a, int b, int c);
    int main() { baz(baz(1,2,3), ^); }
  )cpp");
  EXPECT_THAT(Results.signatures,
              ElementsAre(Sig("baz([[int a]], [[int b]], [[int c]]) -> int")));
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(1, Results.activeParameter);
}

TEST(SignatureHelpTest, OpeningParen) {
  llvm::StringLiteral Tests[] = {// Recursive function call.
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
      ID(foo $p^( foo(10), ^ ))
    })cpp"};

  for (auto Test : Tests) {
    Annotations Code(Test);
    EXPECT_EQ(signatures(Code.code(), Code.point()).argListStart,
              Code.point("p"))
        << "Test source:" << Test;
  }
}

class IndexRequestCollector : public SymbolIndex {
public:
  bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const override {
    std::unique_lock<std::mutex> Lock(Mut);
    Requests.push_back(Req);
    ReceivedRequestCV.notify_one();
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
              AllOf(Contains(Labeled("SomeNameOfField")),
                    Contains(Labeled("SomeNameOfTypedefField")),
                    Not(Contains(Labeled("SomeNameInTheIndex")))));
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
                Not(Contains(Labeled("SomeNameInTheIndex"))));
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
                Not(Contains(Labeled("SomeNameInTheIndex"))));
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
                Not(Contains(Labeled("SomeNameInTheIndex"))));
  }
}

TEST(CompletionTest, OverloadBundling) {
  clangd::CodeCompleteOptions Opts;
  Opts.BundleOverloads = true;

  std::string Context = R"cpp(
    struct X {
      // Overload with int
      int a(int);
      // Overload with bool
      int a(bool);
      int b(float);
    };
    int GFuncC(int);
    int GFuncD(int);
  )cpp";

  // Member completions are bundled.
  EXPECT_THAT(completions(Context + "int y = X().^", {}, Opts).Completions,
              UnorderedElementsAre(Labeled("a(…)"), Labeled("b(float)")));

  // Non-member completions are bundled, including index+sema.
  Symbol NoArgsGFunc = func("GFuncC");
  EXPECT_THAT(
      completions(Context + "int y = GFunc^", {NoArgsGFunc}, Opts).Completions,
      UnorderedElementsAre(Labeled("GFuncC(…)"), Labeled("GFuncD(int)")));

  // Differences in header-to-insert suppress bundling.
  std::string DeclFile = URI::create(testPath("foo")).toString();
  NoArgsGFunc.CanonicalDeclaration.FileURI = DeclFile.c_str();
  NoArgsGFunc.IncludeHeaders.emplace_back("<foo>", 1);
  EXPECT_THAT(
      completions(Context + "int y = GFunc^", {NoArgsGFunc}, Opts).Completions,
      UnorderedElementsAre(AllOf(Named("GFuncC"), InsertInclude("<foo>")),
                           Labeled("GFuncC(int)"), Labeled("GFuncD(int)")));

  // Examine a bundled completion in detail.
  auto A =
      completions(Context + "int y = X().a^", {}, Opts).Completions.front();
  EXPECT_EQ(A.Name, "a");
  EXPECT_EQ(A.Signature, "(…)");
  EXPECT_EQ(A.BundleSize, 2u);
  EXPECT_EQ(A.Kind, CompletionItemKind::Method);
  EXPECT_EQ(A.ReturnType, "int"); // All overloads return int.
  // For now we just return one of the doc strings arbitrarily.
  EXPECT_THAT(A.Documentation, AnyOf(HasSubstr("Overload with int"),
                                     HasSubstr("Overload with bool")));
  EXPECT_EQ(A.SnippetSuffix, "($0)");
}

TEST(CompletionTest, DocumentationFromChangedFileCrash) {
  MockFSProvider FS;
  auto FooH = testPath("foo.h");
  auto FooCpp = testPath("foo.cpp");
  FS.Files[FooH] = R"cpp(
    // this is my documentation comment.
    int func();
  )cpp";
  FS.Files[FooCpp] = "";

  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

  Annotations Source(R"cpp(
    #include "foo.h"
    int func() {
      // This makes sure we have func from header in the AST.
    }
    int a = fun^
  )cpp");
  Server.addDocument(FooCpp, Source.code(), WantDiagnostics::Yes);
  // We need to wait for preamble to build.
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  // Change the header file. Completion will reuse the old preamble!
  FS.Files[FooH] = R"cpp(
    int func();
  )cpp";

  clangd::CodeCompleteOptions Opts;
  Opts.IncludeComments = true;
  CodeCompleteResult Completions =
      cantFail(runCodeComplete(Server, FooCpp, Source.point(), Opts));
  // We shouldn't crash. Unfortunately, current workaround is to not produce
  // comments for symbols from headers.
  EXPECT_THAT(Completions.Completions,
              Contains(AllOf(Not(IsDocumented()), Named("func"))));
}

TEST(CompletionTest, NonDocComments) {
  MockFSProvider FS;
  auto FooCpp = testPath("foo.cpp");
  FS.Files[FooCpp] = "";

  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

  Annotations Source(R"cpp(
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
  )cpp");
  // FIXME: Auto-completion in a template requires disabling delayed template
  // parsing.
  CDB.ExtraClangFlags.push_back("-fno-delayed-template-parsing");
  runAddDocument(Server, FooCpp, Source.code(), WantDiagnostics::Yes);
  CodeCompleteResult Completions = cantFail(runCodeComplete(
      Server, FooCpp, Source.point(), clangd::CodeCompleteOptions()));

  // We should not get any of those comments in completion.
  EXPECT_THAT(
      Completions.Completions,
      UnorderedElementsAre(AllOf(Not(IsDocumented()), Named("comments_foo")),
                           AllOf(IsDocumented(), Named("comments_baz")),
                           AllOf(IsDocumented(), Named("comments_quux")),
                           AllOf(Not(IsDocumented()), Named("comments_ns")),
                           // FIXME(ibiryukov): the following items should have
                           // empty documentation, since they are separated from
                           // a comment with an empty line. Unfortunately, I
                           // couldn't make Sema tests pass if we ignore those.
                           AllOf(IsDocumented(), Named("comments_bar")),
                           AllOf(IsDocumented(), Named("comments_qux"))));
}

TEST(CompletionTest, CompleteOnInvalidLine) {
  auto FooCpp = testPath("foo.cpp");

  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  MockFSProvider FS;
  FS.Files[FooCpp] = "// empty file";

  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
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
      UnorderedElementsAre(Scope("ns::"), Scope("ns::"), Scope("ns::")));
}

TEST(CompletionTest, Render) {
  CodeCompletion C;
  C.Name = "x";
  C.Signature = "(bool) const";
  C.SnippetSuffix = "(${0:bool})";
  C.ReturnType = "int";
  C.RequiredQualifier = "Foo::";
  C.Scope = "ns::Foo::";
  C.Documentation = "This is x().";
  C.Includes.emplace_back();
  auto &Include = C.Includes.back();
  Include.Header = "\"foo.h\"";
  C.Kind = CompletionItemKind::Method;
  C.Score.Total = 1.0;
  C.Origin = SymbolOrigin::AST | SymbolOrigin::Static;

  CodeCompleteOptions Opts;
  Opts.IncludeIndicator.Insert = "^";
  Opts.IncludeIndicator.NoInsert = "";
  Opts.EnableSnippets = false;

  auto R = C.render(Opts);
  EXPECT_EQ(R.label, "Foo::x(bool) const");
  EXPECT_EQ(R.insertText, "Foo::x");
  EXPECT_EQ(R.insertTextFormat, InsertTextFormat::PlainText);
  EXPECT_EQ(R.filterText, "x");
  EXPECT_EQ(R.detail, "int\n\"foo.h\"");
  EXPECT_EQ(R.documentation, "This is x().");
  EXPECT_THAT(R.additionalTextEdits, IsEmpty());
  EXPECT_EQ(R.sortText, sortText(1.0, "x"));
  EXPECT_FALSE(R.deprecated);

  Opts.EnableSnippets = true;
  R = C.render(Opts);
  EXPECT_EQ(R.insertText, "Foo::x(${0:bool})");
  EXPECT_EQ(R.insertTextFormat, InsertTextFormat::Snippet);

  Include.Insertion.emplace();
  R = C.render(Opts);
  EXPECT_EQ(R.label, "^Foo::x(bool) const");
  EXPECT_THAT(R.additionalTextEdits, Not(IsEmpty()));

  Opts.ShowOrigins = true;
  R = C.render(Opts);
  EXPECT_EQ(R.label, "^[AS]Foo::x(bool) const");

  C.BundleSize = 2;
  R = C.render(Opts);
  EXPECT_EQ(R.detail, "[2 overloads]\n\"foo.h\"");

  C.Deprecated = true;
  R = C.render(Opts);
  EXPECT_TRUE(R.deprecated);
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
  EXPECT_THAT(Results.Completions, UnorderedElementsAre(Named("NotRecovered")));
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
              UnorderedElementsAre(AllOf(Scope("ns::X::"), Named("x_"))));
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
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

  CodeCompleteOptions Opts;
  Opts.IncludeFixIts = true;
  Annotations TestCode(
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
      )cpp");
  auto Results =
      completions(Server, TestCode.code(), TestCode.point(), {}, Opts);
  EXPECT_EQ(Results.Completions.size(), 3u);

  TextEdit ReplacementEdit;
  ReplacementEdit.range = TestCode.range();
  ReplacementEdit.newText = ".";
  for (const auto &C : Results.Completions) {
    EXPECT_TRUE(C.FixIts.size() == 1u || C.Name == "AuxFunction");
    if (!C.FixIts.empty()) {
      EXPECT_THAT(C.FixIts, ElementsAre(ReplacementEdit));
    }
  }
}

TEST(CompletionTest, FixItForDotToArrow) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

  CodeCompleteOptions Opts;
  Opts.IncludeFixIts = true;
  Annotations TestCode(
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
      )cpp");
  auto Results =
      completions(Server, TestCode.code(), TestCode.point(), {}, Opts);
  EXPECT_EQ(Results.Completions.size(), 3u);

  TextEdit ReplacementEdit;
  ReplacementEdit.range = TestCode.range();
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
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

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
      )cpp"};
  for (const auto &Text : TestCodes) {
    Annotations TestCode(Text);
    auto Results = completions(Server, TestCode.code(), TestCode.point());

    if (Results.Completions.size() != 1) {
      ADD_FAILURE() << "Results.Completions.size() != 1";
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
              ElementsAre(Sig("foo([[int x]]) -> void"),
                          Sig("foo([[int x]], [[int y = 0]]) -> void"),
                          Sig("foo([[float x]], [[int y]]) -> void"),
                          Sig("foo([[int x]], [[float y]]) -> void"),
                          Sig("foo([[float x]], [[float y]]) -> void")));
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
              ElementsAre(Sig("foo([[T]], [[T]], [[T]]) -> void")));

  StringRef Sig1 = R"cpp(
    template <class T>
    void foo(T, T, T);

    int main() {
      foo(10, ^);
    })cpp";

  EXPECT_THAT(signatures(Sig1).signatures,
              ElementsAre(Sig("foo([[T]], [[T]], [[T]]) -> void")));

  StringRef Sig2 = R"cpp(
    template <class ...T>
    void foo(T...);

    int main() {
      foo<int>(^);
    }
  )cpp";

  EXPECT_THAT(signatures(Sig2).signatures,
              ElementsAre(Sig("foo([[T...]]) -> void")));

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
              ElementsAre(Sig("foo([[T]], [[U]]) -> void")));
}

TEST(SignatureHelpTest, IndexDocumentation) {
  Symbol Foo0 = sym("foo", index::SymbolKind::Function, "@F@\\0#");
  Foo0.Documentation = "Doc from the index";
  Symbol Foo1 = sym("foo", index::SymbolKind::Function, "@F@\\0#I#");
  Foo1.Documentation = "Doc from the index";
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
      ElementsAre(AllOf(Sig("foo() -> int"), SigDoc("Doc from the index")),
                  AllOf(Sig("foo([[double]]) -> int"), SigDoc(""))));

  StringRef Sig1 = R"cpp(
    int foo();
    // Overriden doc from sema
    int foo(int);
    // Doc from sema
    int foo(int, int);

    void test() {
      foo(^);
    }
  )cpp";

  EXPECT_THAT(
      signatures(Sig1, {Foo0, Foo1, Foo2}).signatures,
      ElementsAre(
          AllOf(Sig("foo() -> int"), SigDoc("Doc from the index")),
          AllOf(Sig("foo([[int]]) -> int"), SigDoc("Overriden doc from sema")),
          AllOf(Sig("foo([[int]], [[int]]) -> int"), SigDoc("Doc from sema"))));
}

TEST(SignatureHelpTest, DynamicIndexDocumentation) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer::Options Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true;
  ClangdServer Server(CDB, FS, DiagConsumer, Opts);

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
  EXPECT_THAT(
      llvm::cantFail(runSignatureHelp(Server, File, FileContent.point()))
          .signatures,
      ElementsAre(AllOf(Sig("foo() -> int"), SigDoc("Member doc"))));
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
        UnorderedElementsAre(AllOf(Named("xfoo"), SnippetSuffix("()")),
                             AllOf(Named("xfoo"), SnippetSuffix("($0)"))));
  }
  {
    auto Results = completions(
        R"cpp(
      void xbar();
      void f() { xba^ })cpp",
        {}, Opts);
    EXPECT_THAT(Results.Completions, UnorderedElementsAre(AllOf(
                                         Named("xbar"), SnippetSuffix("()"))));
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
        UnorderedElementsAre(AllOf(Named("xfoo"), SnippetSuffix("($0)"))));
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
        UnorderedElementsAre(AllOf(Named("xfoo"), SnippetSuffix("<$1>($0)"))));
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
        UnorderedElementsAre(AllOf(Named("foo_class"), SnippetSuffix("<$0>")),
                             AllOf(Named("foo_alias"), SnippetSuffix("<$0>"))));
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
      AllOf(Contains(AllOf(Labeled("void vfunc(bool param, int p) override"),
                           NameStartsWith("vfunc"))),
            Contains(AllOf(Labeled("void ttt(bool param) const override"),
                           NameStartsWith("ttt"))),
            Not(Contains(Labeled("void vfunc(bool param) override")))));
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
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

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
  Opts.SpeculativeIndexRequest = true;

  auto CompleteAtPoint = [&](StringRef P) {
    cantFail(runCodeComplete(Server, File, Test.point(P), Opts));
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
  Sym.IncludeHeaders.emplace_back("\"foo.h\"", 2);
  Sym.IncludeHeaders.emplace_back("\"bar.h\"", 1000);

  auto Results = completions("Fun^", {Sym}).Completions;
  assert(!Results.empty());
  EXPECT_THAT(Results[0], AllOf(Named("Func"), InsertInclude("\"bar.h\"")));
  EXPECT_EQ(Results[0].Includes.size(), 2u);
}

TEST(CompletionTest, NoInsertIncludeIfOnePresent) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;

  std::string FooHeader = testPath("foo.h");
  FS.Files[FooHeader] = "";

  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

  std::string DeclFile = URI::create(testPath("foo")).toString();
  Symbol Sym = func("Func");
  Sym.CanonicalDeclaration.FileURI = DeclFile.c_str();
  Sym.IncludeHeaders.emplace_back("\"foo.h\"", 2);
  Sym.IncludeHeaders.emplace_back("\"bar.h\"", 1000);

  EXPECT_THAT(
      completions(Server, "#include \"foo.h\"\nFun^", {Sym}).Completions,
      UnorderedElementsAre(
          AllOf(Named("Func"), HasInclude("\"foo.h\""), Not(InsertInclude()))));
}

TEST(CompletionTest, MergeMacrosFromIndexAndSema) {
  Symbol Sym;
  Sym.Name = "Clangd_Macro_Test";
  Sym.ID = SymbolID("c:foo.cpp@8@macro@Clangd_Macro_Test");
  Sym.SymInfo.Kind = index::SymbolKind::Macro;
  Sym.Flags |= Symbol::IndexedForCodeCompletion;
  EXPECT_THAT(completions("#define Clangd_Macro_Test\nClangd_Macro_T^", {Sym})
                  .Completions,
              UnorderedElementsAre(Named("Clangd_Macro_Test")));
}

TEST(CompletionTest, MacroFromPreamble) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  std::string FooHeader = testPath("foo.h");
  FS.Files[FooHeader] = "#define CLANGD_PREAMBLE_HEADER x\n";
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
  auto Results = completions(
      R"cpp(#include "foo.h"
          #define CLANGD_PREAMBLE_MAIN x

          int x = 0;
          #define CLANGD_MAIN x
          void f() { CLANGD_^ }
      )cpp",
      {func("CLANGD_INDEX")});
  // We should get results from the main file, including the preamble section.
  // However no results from included files (the index should cover them).
  EXPECT_THAT(Results.Completions,
              UnorderedElementsAre(Named("CLANGD_PREAMBLE_MAIN"),
                                   Named("CLANGD_MAIN"),
                                   Named("CLANGD_INDEX")));
}

TEST(CompletionTest, DeprecatedResults) {
  std::string Body = R"cpp(
    void TestClangd();
    void TestClangc() __attribute__((deprecated("", "")));
  )cpp";

  EXPECT_THAT(
      completions(Body + "int main() { TestClang^ }").Completions,
      UnorderedElementsAre(AllOf(Named("TestClangd"), Not(Deprecated())),
                           AllOf(Named("TestClangc"), Deprecated())));
}

TEST(SignatureHelpTest, InsideArgument) {
  {
    const auto Results = signatures(R"cpp(
      void foo(int x);
      void foo(int x, int y);
      int main() { foo(1+^); }
    )cpp");
    EXPECT_THAT(Results.signatures,
                ElementsAre(Sig("foo([[int x]]) -> void"),
                            Sig("foo([[int x]], [[int y]]) -> void")));
    EXPECT_EQ(0, Results.activeParameter);
  }
  {
    const auto Results = signatures(R"cpp(
      void foo(int x);
      void foo(int x, int y);
      int main() { foo(1^); }
    )cpp");
    EXPECT_THAT(Results.signatures,
                ElementsAre(Sig("foo([[int x]]) -> void"),
                            Sig("foo([[int x]], [[int y]]) -> void")));
    EXPECT_EQ(0, Results.activeParameter);
  }
  {
    const auto Results = signatures(R"cpp(
      void foo(int x);
      void foo(int x, int y);
      int main() { foo(1^0); }
    )cpp");
    EXPECT_THAT(Results.signatures,
                ElementsAre(Sig("foo([[int x]]) -> void"),
                            Sig("foo([[int x]], [[int y]]) -> void")));
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
                ElementsAre(Sig("foo([[int x]], [[int y]]) -> void")));
    EXPECT_EQ(1, Results.activeParameter);
  }
}

TEST(SignatureHelpTest, ConstructorInitializeFields) {
  {
    const auto Results = signatures(R"cpp(
      struct A {
        A(int);
      };
      struct B {
        B() : a_elem(^) {}
        A a_elem;
      };
    )cpp");
    EXPECT_THAT(Results.signatures,
                UnorderedElementsAre(Sig("A([[int]])"), Sig("A([[A &&]])"),
                                     Sig("A([[const A &]])")));
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
                UnorderedElementsAre(Sig("A([[int]])"), Sig("A([[A &&]])"),
                                     Sig("A([[const A &]])")));
  }
}

TEST(CompletionTest, IncludedCompletionKinds) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  std::string Subdir = testPath("sub");
  std::string SearchDirArg = (Twine("-I") + Subdir).str();
  CDB.ExtraClangFlags = {SearchDirArg.c_str()};
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = "";
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
  auto Results = completions(Server,
                             R"cpp(
        #include "^"
      )cpp");
  EXPECT_THAT(Results.Completions,
              AllOf(Has("sub/", CompletionItemKind::Folder),
                    Has("bar.h\"", CompletionItemKind::File)));
}

TEST(CompletionTest, NoCrashAtNonAlphaIncludeHeader) {
  auto Results = completions(
      R"cpp(
        #include "./^"
      )cpp");
  EXPECT_TRUE(Results.Completions.empty());
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
                  AllOf(Qualifier(""), Scope("na::"), Named("ClangdA"))));
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
       cls("na::nb::Clangd4")},
      Opts);
  EXPECT_THAT(
      Results.Completions,
      UnorderedElementsAre(AllOf(Qualifier("nx::"), Named("Clangd1")),
                           AllOf(Qualifier("ny::"), Named("Clangd2")),
                           AllOf(Qualifier(""), Scope(""), Named("Clangd3")),
                           AllOf(Qualifier("nb::"), Named("Clangd4"))));
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
              UnorderedElementsAre(AllOf(Qualifier(""), Named("Clangd1")),
                                   AllOf(Qualifier("nx::"), Named("Clangd2"))));
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
  EXPECT_THAT(C, ElementsAre(Named("value")));
  EXPECT_THAT(C, ElementsAre(Kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(ReturnType("int")));
  EXPECT_THAT(C, ElementsAre(Signature("")));
  EXPECT_THAT(C, ElementsAre(SnippetSuffix("")));
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
  EXPECT_THAT(C, ElementsAre(Named("valueForCharacter:")));
  EXPECT_THAT(C, ElementsAre(Kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(ReturnType("int")));
  EXPECT_THAT(C, ElementsAre(Signature("(char)")));
  EXPECT_THAT(C, ElementsAre(SnippetSuffix("${1:(char)}")));
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
  EXPECT_THAT(C, ElementsAre(Named("fooWithValue:")));
  EXPECT_THAT(C, ElementsAre(Kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(ReturnType("id")));
  EXPECT_THAT(C, ElementsAre(Signature("(int) fooey:(unsigned int)")));
  EXPECT_THAT(
      C, ElementsAre(SnippetSuffix("${1:(int)} fooey:${2:(unsigned int)}")));
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
  EXPECT_THAT(C, ElementsAre(Named("fooey:")));
  EXPECT_THAT(C, ElementsAre(Kind(CompletionItemKind::Method)));
  EXPECT_THAT(C, ElementsAre(ReturnType("id")));
  EXPECT_THAT(C, ElementsAre(Signature("(unsigned int)")));
  EXPECT_THAT(C, ElementsAre(SnippetSuffix("${1:(unsigned int)}")));
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
              Contains(AllOf(
                  Named("while"),
                  SnippetSuffix(" (${1:condition}) {\n${0:statements}\n}"))));
  // However, snippets for functions must *not* end with $0.
  EXPECT_THAT(Results.Completions,
              Contains(AllOf(Named("while_foo"),
                             SnippetSuffix("(${1:int a}, ${2:int b})"))));
}

TEST(CompletionTest, WorksWithNullType) {
  auto R = completions(R"cpp(
    int main() {
      for (auto [loopVar] : y ) { // y has to be unresolved.
        int z = loopV^;
      }
    }
  )cpp");
  EXPECT_THAT(R.Completions, ElementsAre(Named("loopVar")));
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
              ElementsAre(AllOf(Scope("std::"), Named("foo"),
                                Kind(CompletionItemKind::Reference))));
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
              UnorderedElementsAre(AllOf(Qualifier(""), Named("XYZ"))));
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
              UnorderedElementsAre(AllOf(Qualifier(""), Named("XYZ"))));
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
              UnorderedElementsAre(AllOf(Qualifier(""), Named("ABCDE"))));
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
              ElementsAre(AllOf(ReturnType("int"), Named("size"))));
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
              UnorderedElementsAre(Named("void"), Named("func"), Named("int"),
                                   Named("xyz"), Named("abc")));
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
              UnorderedElementsAre(Named("sym1"), Named("sym2")));
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
              UnorderedElementsAre(AllOf(Qualifier(""), Scope("")),
                                   AllOf(Qualifier(""), Scope("a::")),
                                   AllOf(Qualifier(""), Scope("ns::b::"))));
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
              UnorderedElementsAre(AllOf(Qualifier(""), Scope("")),
                                   AllOf(Qualifier(""), Scope("a::")),
                                   AllOf(Qualifier(""), Scope("ns::b::")),
                                   AllOf(Qualifier("c::"), Scope("c::")),
                                   AllOf(Qualifier("d::"), Scope("ns::d::"))));
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
              ElementsAre(AllOf(Qualifier(""), Scope("ns::b::"))));
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
              ElementsAre(AllOf(Qualifier(""), Scope("a::"))));
}

} // namespace
} // namespace clangd
} // namespace clang
