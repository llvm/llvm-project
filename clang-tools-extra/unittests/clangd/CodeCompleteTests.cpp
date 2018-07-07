//===-- CodeCompleteTests.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "index/MemIndex.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

namespace {
using namespace llvm;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
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
MATCHER_P(Scope, S, "") { return arg.Scope == S; }
MATCHER_P(Qualifier, Q, "") { return arg.RequiredQualifier == Q; }
MATCHER_P(Labeled, Label, "") {
  return arg.RequiredQualifier + arg.Name + arg.Signature == Label;
}
MATCHER_P(SigHelpLabeled, Label, "") { return arg.label == Label; }
MATCHER_P(Kind, K, "") { return arg.Kind == K; }
MATCHER_P(Doc, D, "") { return arg.Documentation == D; }
MATCHER_P(ReturnType, D, "") { return arg.ReturnType == D; }
MATCHER_P(InsertInclude, IncludeHeader, "") {
  return arg.Header == IncludeHeader && bool(arg.HeaderInsertion);
}
MATCHER(InsertInclude, "") { return bool(arg.HeaderInsertion); }
MATCHER_P(SnippetSuffix, Text, "") { return arg.SnippetSuffix == Text; }
MATCHER_P(Origin, OriginSet, "") { return arg.Origin == OriginSet; }

// Shorthand for Contains(Named(Name)).
Matcher<const std::vector<CodeCompletion> &> Has(std::string Name) {
  return Contains(Named(std::move(Name)));
}
Matcher<const std::vector<CodeCompletion> &> Has(std::string Name,
                                                 CompletionItemKind K) {
  return Contains(AllOf(Named(std::move(Name)), Kind(K)));
}
MATCHER(IsDocumented, "") { return !arg.Documentation.empty(); }

std::unique_ptr<SymbolIndex> memIndex(std::vector<Symbol> Symbols) {
  SymbolSlab::Builder Slab;
  for (const auto &Sym : Symbols)
    Slab.insert(Sym);
  return MemIndex::build(std::move(Slab).build());
}

CodeCompleteResult completions(ClangdServer &Server, StringRef Text,
                               std::vector<Symbol> IndexSymbols = {},
                               clangd::CodeCompleteOptions Opts = {}) {
  std::unique_ptr<SymbolIndex> OverrideIndex;
  if (!IndexSymbols.empty()) {
    assert(!Opts.Index && "both Index and IndexSymbols given!");
    OverrideIndex = memIndex(std::move(IndexSymbols));
    Opts.Index = OverrideIndex.get();
  }

  auto File = testPath("foo.cpp");
  Annotations Test(Text);
  runAddDocument(Server, File, Test.code());
  auto CompletionList =
      cantFail(runCodeComplete(Server, File, Test.point(), Opts));
  return CompletionList;
}

// Builds a server and runs code completion.
// If IndexSymbols is non-empty, an index will be built and passed to opts.
CodeCompleteResult completions(StringRef Text,
                               std::vector<Symbol> IndexSymbols = {},
                               clangd::CodeCompleteOptions Opts = {}) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
  return completions(Server, Text, std::move(IndexSymbols), std::move(Opts));
}

std::string replace(StringRef Haystack, StringRef Needle, StringRef Repl) {
  std::string Result;
  raw_string_ostream OS(Result);
  std::pair<StringRef, StringRef> Split;
  for (Split = Haystack.split(Needle); !Split.second.empty();
       Split = Split.first.split(Needle))
    OS << Split.first << Repl;
  Result += Split.first;
  OS.flush();
  return Result;
}

// Helpers to produce fake index symbols for memIndex() or completions().
// USRFormat is a regex replacement string for the unqualified part of the USR.
Symbol sym(StringRef QName, index::SymbolKind Kind, StringRef USRFormat) {
  Symbol Sym;
  std::string USR = "c:"; // We synthesize a few simple cases of USRs by hand!
  size_t Pos = QName.rfind("::");
  if (Pos == llvm::StringRef::npos) {
    Sym.Name = QName;
    Sym.Scope = "";
  } else {
    Sym.Name = QName.substr(Pos + 2);
    Sym.Scope = QName.substr(0, Pos + 2);
    USR += "@N@" + replace(QName.substr(0, Pos), "::", "@N@"); // ns:: -> @N@ns
  }
  USR += Regex("^.*$").sub(USRFormat, Sym.Name); // e.g. func -> @F@func#
  Sym.ID = SymbolID(USR);
  Sym.SymInfo.Kind = Kind;
  Sym.IsIndexedForCodeCompletion = true;
  Sym.Origin = SymbolOrigin::Static;
  return Sym;
}
Symbol func(StringRef Name) { // Assumes the function has no args.
  return sym(Name, index::SymbolKind::Function, "@F@\\0#"); // no args
}
Symbol cls(StringRef Name) {
  return sym(Name, index::SymbolKind::Class, "@S@\\0");
}
Symbol var(StringRef Name) {
  return sym(Name, index::SymbolKind::Variable, "@\\0");
}
Symbol ns(StringRef Name) {
  return sym(Name, index::SymbolKind::Namespace, "@N@\\0");
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
}
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

void TestAfterDotCompletion(clangd::CodeCompleteOptions Opts) {
  auto Results = completions(
      R"cpp(
      #define MACRO X

      int global_var;

      int global_func();

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

void TestGlobalScopeCompletion(clangd::CodeCompleteOptions Opts) {
  auto Results = completions(
      R"cpp(
      #define MACRO X

      int global_var;
      int global_func();

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
    TestAfterDotCompletion(Opts);
    TestGlobalScopeCompletion(Opts);
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

TEST(CompletionTest, Priorities) {
  auto Internal = completions(R"cpp(
      class Foo {
        public: void pub();
        protected: void prot();
        private: void priv();
      };
      void Foo::pub() { this->^ }
  )cpp");
  EXPECT_THAT(Internal.Completions,
              HasSubsequence(Named("priv"), Named("prot"), Named("pub")));

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
              HasSubsequence(AllOf(Qualifier(""), Named("bar")),
                             AllOf(Qualifier("Foo::"), Named("foo"))));
  EXPECT_THAT(Results.Completions,
              Not(Contains(AllOf(Qualifier(""), Named("foo"))))); // private
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

TEST(CompletionTest, Kinds) {
  auto Results = completions(
      R"cpp(
          #define MACRO X
          int variable;
          struct Struct {};
          int function();
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
  auto Results = completions("int main() { abs^ }", {ns("absl"), func("abs")});
  EXPECT_THAT(Results.Completions, HasSubsequence(Named("abs"), Named("absl")));
  Results = completions("int main() { abs^ }",
                        {withReferences(10000, ns("absl")), func("abs")});
  EXPECT_THAT(Results.Completions, HasSubsequence(Named("absl"), Named("abs")));
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
  std::string SearchDirArg = (llvm::Twine("-I") + Subdir).str();
  CDB.ExtraClangFlags = {SearchDirArg.c_str()};
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = "";

  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
  Symbol::Details Scratch;
  auto BarURI = URI::createFile(BarHeader).toString();
  Symbol Sym = cls("ns::X");
  Sym.CanonicalDeclaration.FileURI = BarURI;
  Scratch.IncludeHeader = BarURI;
  Sym.Detail = &Scratch;
  // Shoten include path based on search dirctory and insert.
  auto Results = completions(Server,
                             R"cpp(
          int main() { ns::^ }
      )cpp",
                             {Sym});
  EXPECT_THAT(Results.Completions,
              ElementsAre(AllOf(Named("X"), InsertInclude("\"bar.h\""))));
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
  Symbol::Details Scratch;
  Symbol SymX = cls("ns::X");
  Symbol SymY = cls("ns::Y");
  std::string BarHeader = testPath("bar.h");
  auto BarURI = URI::createFile(BarHeader).toString();
  SymX.CanonicalDeclaration.FileURI = BarURI;
  SymY.CanonicalDeclaration.FileURI = BarURI;
  Scratch.IncludeHeader = "<bar>";
  SymX.Detail = &Scratch;
  SymY.Detail = &Scratch;
  // Shoten include path based on search dirctory and insert.
  auto Results = completions(Server,
                             R"cpp(
          namespace ns {
            class X;
            class Y {}
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
      void f() { ns::preamble().$2^; }
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

TEST(CompletionTest, GlobalCompletionFiltering) {

  Symbol Class = cls("XYZ");
  Class.IsIndexedForCodeCompletion = false;
  Symbol Func = func("XYZ::foooo");
  Func.IsIndexedForCodeCompletion = false;

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

TEST(CompletionTest, CompleteInExcludedPPBranch) {
  auto Results = completions(R"cpp(
    int bar(int param_in_bar) {
    }

    int foo(int param_in_foo) {
#if 0
  par^
#endif
    }
)cpp");

  EXPECT_THAT(Results.Completions, Contains(Labeled("param_in_foo")));
  EXPECT_THAT(Results.Completions, Not(Contains(Labeled("param_in_bar"))));
}

SignatureHelp signatures(StringRef Text) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());
  auto File = testPath("foo.cpp");
  Annotations Test(Text);
  runAddDocument(Server, File, Test.code());
  return cantFail(runSignatureHelp(Server, File, Test.point()));
}

MATCHER_P(ParamsAre, P, "") {
  if (P.size() != arg.parameters.size())
    return false;
  for (unsigned I = 0; I < P.size(); ++I)
    if (P[I] != arg.parameters[I].label)
      return false;
  return true;
}

Matcher<SignatureInformation> Sig(std::string Label,
                                  std::vector<std::string> Params) {
  return AllOf(SigHelpLabeled(Label), ParamsAre(Params));
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
              UnorderedElementsAre(
                  Sig("foo(float x, float y) -> void", {"float x", "float y"}),
                  Sig("foo(float x, int y) -> void", {"float x", "int y"}),
                  Sig("foo(int x, float y) -> void", {"int x", "float y"}),
                  Sig("foo(int x, int y) -> void", {"int x", "int y"})));
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
                  Sig("bar(int x, int y = 0) -> void", {"int x", "int y = 0"}),
                  Sig("bar(float x = 0, int y = 42) -> void",
                      {"float x = 0", "int y = 42"})));
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(0, Results.activeParameter);
}

TEST(SignatureHelpTest, ActiveArg) {
  auto Results = signatures(R"cpp(
    int baz(int a, int b, int c);
    int main() { baz(baz(1,2,3), ^); }
  )cpp");
  EXPECT_THAT(Results.signatures,
              ElementsAre(Sig("baz(int a, int b, int c) -> int",
                              {"int a", "int b", "int c"})));
  EXPECT_EQ(0, Results.activeSignature);
  EXPECT_EQ(1, Results.activeParameter);
}

class IndexRequestCollector : public SymbolIndex {
public:
  bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const override {
    Requests.push_back(Req);
    return true;
  }

  void lookup(const LookupRequest &,
              llvm::function_ref<void(const Symbol &)>) const override {}

  const std::vector<FuzzyFindRequest> allRequests() const { return Requests; }

private:
  mutable std::vector<FuzzyFindRequest> Requests;
};

std::vector<FuzzyFindRequest> captureIndexRequests(llvm::StringRef Code) {
  clangd::CodeCompleteOptions Opts;
  IndexRequestCollector Requests;
  Opts.Index = &Requests;
  completions(Code, {}, Opts);
  return Requests.allRequests();
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

  EXPECT_THAT(Requests, ElementsAre(Field(&FuzzyFindRequest::Scopes,
                                          UnorderedElementsAre("bar::"))));
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
  Symbol::Details Detail;
  std::string DeclFile = URI::createFile(testPath("foo")).toString();
  NoArgsGFunc.CanonicalDeclaration.FileURI = DeclFile;
  Detail.IncludeHeader = "<foo>";
  NoArgsGFunc.Detail = &Detail;
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
  EXPECT_EQ(A.SnippetSuffix, "(${0})");
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
  Server.addDocument(FooCpp, Source.code(), WantDiagnostics::Yes);
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
  C.Header = "\"foo.h\"";
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

  Opts.EnableSnippets = true;
  R = C.render(Opts);
  EXPECT_EQ(R.insertText, "Foo::x(${0:bool})");
  EXPECT_EQ(R.insertTextFormat, InsertTextFormat::Snippet);

  C.HeaderInsertion.emplace();
  R = C.render(Opts);
  EXPECT_EQ(R.label, "^Foo::x(bool) const");
  EXPECT_THAT(R.additionalTextEdits, Not(IsEmpty()));

  Opts.ShowOrigins = true;
  R = C.render(Opts);
  EXPECT_EQ(R.label, "^[AS]Foo::x(bool) const");

  C.BundleSize = 2;
  R = C.render(Opts);
  EXPECT_EQ(R.detail, "[2 overloads]\n\"foo.h\"");
}

} // namespace
} // namespace clangd
} // namespace clang
