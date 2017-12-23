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
#include "Context.h"
#include "Matchers.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "index/MemIndex.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
// Let GMock print completion items and signature help.
void PrintTo(const CompletionItem &I, std::ostream *O) {
  llvm::raw_os_ostream OS(*O);
  OS << I.label << " - " << toJSON(I);
}
void PrintTo(const std::vector<CompletionItem> &V, std::ostream *O) {
  *O << "{\n";
  for (const auto &I : V) {
    *O << "\t";
    PrintTo(I, O);
    *O << "\n";
  }
  *O << "}";
}
void PrintTo(const SignatureInformation &I, std::ostream *O) {
  llvm::raw_os_ostream OS(*O);
  OS << I.label << " - " << toJSON(I);
}
void PrintTo(const std::vector<SignatureInformation> &V, std::ostream *O) {
  *O << "{\n";
  for (const auto &I : V) {
    *O << "\t";
    PrintTo(I, O);
    *O << "\n";
  }
  *O << "}";
}

namespace {
using namespace llvm;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Not;

class IgnoreDiagnostics : public DiagnosticsConsumer {
  void onDiagnosticsReady(
      PathRef File, Tagged<std::vector<DiagWithFixIts>> Diagnostics) override {}
};

// GMock helpers for matching completion items.
MATCHER_P(Named, Name, "") { return arg.insertText == Name; }
MATCHER_P(Labeled, Label, "") { return arg.label == Label; }
MATCHER_P(Kind, K, "") { return arg.kind == K; }
MATCHER_P(Filter, F, "") { return arg.filterText == F; }
MATCHER_P(PlainText, Text, "") {
  return arg.insertTextFormat == clangd::InsertTextFormat::PlainText &&
         arg.insertText == Text;
}
MATCHER_P(Snippet, Text, "") {
  return arg.insertTextFormat == clangd::InsertTextFormat::Snippet &&
         arg.insertText == Text;
}
// Shorthand for Contains(Named(Name)).
Matcher<const std::vector<CompletionItem> &> Has(std::string Name) {
  return Contains(Named(std::move(Name)));
}
Matcher<const std::vector<CompletionItem> &> Has(std::string Name,
                                                 CompletionItemKind K) {
  return Contains(AllOf(Named(std::move(Name)), Kind(K)));
}
MATCHER(IsDocumented, "") { return !arg.documentation.empty(); }

CompletionList completions(StringRef Text,
                           clangd::CodeCompleteOptions Opts = {}) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true);
  auto File = getVirtualTestFilePath("foo.cpp");
  Annotations Test(Text);
  Server.addDocument(Context::empty(), File, Test.code());
  return Server.codeComplete(Context::empty(), File, Test.point(), Opts)
      .get()
      .second.Value;
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
                             Opts);

  EXPECT_TRUE(Results.isIncomplete);
  EXPECT_THAT(Results.items, ElementsAre(Named("AAA"), Named("BBB")));
}

TEST(CompletionTest, Filter) {
  std::string Body = R"cpp(
    int Abracadabra;
    int Alakazam;
    struct S {
      int FooBar;
      int FooBaz;
      int Qux;
    };
  )cpp";
  EXPECT_THAT(completions(Body + "int main() { S().Foba^ }").items,
              AllOf(Has("FooBar"), Has("FooBaz"), Not(Has("Qux"))));

  EXPECT_THAT(completions(Body + "int main() { S().FR^ }").items,
              AllOf(Has("FooBar"), Not(Has("FooBaz")), Not(Has("Qux"))));

  EXPECT_THAT(completions(Body + "int main() { S().opr^ }").items,
              Has("operator="));

  EXPECT_THAT(completions(Body + "int main() { aaa^ }").items,
              AllOf(Has("Abracadabra"), Has("Alakazam")));

  EXPECT_THAT(completions(Body + "int main() { _a^ }").items,
              AllOf(Has("static_cast"), Not(Has("Abracadabra"))));
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
      Opts);

  // Class members. The only items that must be present in after-dot
  // completion.
  EXPECT_THAT(
      Results.items,
      AllOf(Has(Opts.EnableSnippets ? "method()" : "method"), Has("field")));
  EXPECT_IFF(Opts.IncludeIneligibleResults, Results.items,
             Has("private_field"));
  // Global items.
  EXPECT_THAT(Results.items, Not(AnyOf(Has("global_var"), Has("global_func"),
                                       Has("global_func()"), Has("GlobalClass"),
                                       Has("MACRO"), Has("LocalClass"))));
  // There should be no code patterns (aka snippets) in after-dot
  // completion. At least there aren't any we're aware of.
  EXPECT_THAT(Results.items, Not(Contains(Kind(CompletionItemKind::Snippet))));
  // Check documentation.
  EXPECT_IFF(Opts.IncludeBriefComments, Results.items,
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
      Opts);

  // Class members. Should never be present in global completions.
  EXPECT_THAT(Results.items,
              Not(AnyOf(Has("method"), Has("method()"), Has("field"))));
  // Global items.
  EXPECT_IFF(Opts.IncludeGlobals, Results.items,
             AllOf(Has("global_var"),
                   Has(Opts.EnableSnippets ? "global_func()" : "global_func"),
                   Has("GlobalClass")));
  // A macro.
  EXPECT_IFF(Opts.IncludeMacros, Results.items, Has("MACRO"));
  // Local items. Must be present always.
  EXPECT_THAT(Results.items,
              AllOf(Has("local_var"), Has("LocalClass"),
                    Contains(Kind(CompletionItemKind::Snippet))));
  // Check documentation.
  EXPECT_IFF(Opts.IncludeBriefComments, Results.items,
             Contains(IsDocumented()));
}

TEST(CompletionTest, CompletionOptions) {
  clangd::CodeCompleteOptions Opts;
  for (bool IncludeMacros : {true, false}) {
    Opts.IncludeMacros = IncludeMacros;
    for (bool IncludeGlobals : {true, false}) {
      Opts.IncludeGlobals = IncludeGlobals;
      for (bool IncludeBriefComments : {true, false}) {
        Opts.IncludeBriefComments = IncludeBriefComments;
        for (bool EnableSnippets : {true, false}) {
          Opts.EnableSnippets = EnableSnippets;
          for (bool IncludeCodePatterns : {true, false}) {
            Opts.IncludeCodePatterns = IncludeCodePatterns;
            for (bool IncludeIneligibleResults : {true, false}) {
              Opts.IncludeIneligibleResults = IncludeIneligibleResults;
              TestAfterDotCompletion(Opts);
              TestGlobalScopeCompletion(Opts);
            }
          }
        }
      }
    }
  }
}

// Check code completion works when the file contents are overridden.
TEST(CompletionTest, CheckContentsOverride) {
  MockFSProvider FS;
  IgnoreDiagnostics DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true);
  auto File = getVirtualTestFilePath("foo.cpp");
  Server.addDocument(Context::empty(), File, "ignored text!");

  Annotations Example("int cbc; int b = ^;");
  auto Results = Server
                     .codeComplete(Context::empty(), File, Example.point(),
                                   clangd::CodeCompleteOptions(),
                                   StringRef(Example.code()))
                     .get()
                     .second.Value;
  EXPECT_THAT(Results.items, Contains(Named("cbc")));
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
  EXPECT_THAT(Internal.items,
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
  EXPECT_THAT(External.items,
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
  EXPECT_THAT(Results.items, HasSubsequence(Labeled("bar() const"),
                                            Labeled("Foo::foo() const")));
  EXPECT_THAT(Results.items, Not(Contains(Labeled("foo() const")))); // private
}

TEST(CompletionTest, Snippets) {
  clangd::CodeCompleteOptions Opts;
  Opts.EnableSnippets = true;
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
      Opts);
  EXPECT_THAT(Results.items,
              HasSubsequence(Snippet("a"),
                             Snippet("f(${1:int i}, ${2:const float f})")));
}

TEST(CompletionTest, Kinds) {
  auto Results = completions(R"cpp(
      #define MACRO X
      int variable;
      struct Struct {};
      int function();
      int X = ^
  )cpp");
  EXPECT_THAT(Results.items, Has("function", CompletionItemKind::Function));
  EXPECT_THAT(Results.items, Has("variable", CompletionItemKind::Variable));
  EXPECT_THAT(Results.items, Has("int", CompletionItemKind::Keyword));
  EXPECT_THAT(Results.items, Has("Struct", CompletionItemKind::Class));
  EXPECT_THAT(Results.items, Has("MACRO", CompletionItemKind::Text));

  clangd::CodeCompleteOptions Opts;
  Opts.EnableSnippets = true; // Needed for code patterns.

  Results = completions("nam^");
  EXPECT_THAT(Results.items, Has("namespace", CompletionItemKind::Snippet));
}

SignatureHelp signatures(StringRef Text) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true);
  auto File = getVirtualTestFilePath("foo.cpp");
  Annotations Test(Text);
  Server.addDocument(Context::empty(), File, Test.code());
  auto R = Server.signatureHelp(Context::empty(), File, Test.point());
  assert(R);
  return R.get().Value;
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
  return AllOf(Labeled(Label), ParamsAre(Params));
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

std::unique_ptr<SymbolIndex> simpleIndexFromSymbols(
    std::vector<std::pair<std::string, index::SymbolKind>> Symbols) {
  auto I = llvm::make_unique<MemIndex>();
  struct Snapshot {
    SymbolSlab Slab;
    std::vector<const Symbol *> Pointers;
  };
  auto Snap = std::make_shared<Snapshot>();
  SymbolSlab::Builder Slab;
  for (const auto &Pair : Symbols) {
    Symbol Sym;
    Sym.ID = SymbolID(Pair.first);
    llvm::StringRef QName = Pair.first;
    size_t Pos = QName.rfind("::");
    if (Pos == llvm::StringRef::npos) {
      Sym.Name = QName;
      Sym.Scope = "";
    } else {
      Sym.Name = QName.substr(Pos + 2);
      Sym.Scope = QName.substr(0, Pos);
    }
    Sym.SymInfo.Kind = Pair.second;
    Slab.insert(Sym);
  }
  Snap->Slab = std::move(Slab).build();
  for (auto &Iter : Snap->Slab)
    Snap->Pointers.push_back(&Iter);
  auto S = std::shared_ptr<std::vector<const Symbol *>>(std::move(Snap),
                                                        &Snap->Pointers);
  I->build(std::move(S));
  return std::move(I);
}

TEST(CompletionTest, NoIndex) {
  clangd::CodeCompleteOptions Opts;
  Opts.Index = nullptr;

  auto Results = completions(R"cpp(
      namespace ns { class No {}; }
      void f() { ns::^ }
  )cpp",
                             Opts);
  EXPECT_THAT(Results.items, Has("No"));
}

TEST(CompletionTest, SimpleIndexBased) {
  clangd::CodeCompleteOptions Opts;
  auto I = simpleIndexFromSymbols({{"ns::XYZ", index::SymbolKind::Class},
                                   {"nx::XYZ", index::SymbolKind::Class},
                                   {"ns::foo", index::SymbolKind::Function}});
  Opts.Index = I.get();

  auto Results = completions(R"cpp(
      namespace ns { class No {}; }
      void f() { ns::^ }
  )cpp",
                             Opts);
  EXPECT_THAT(Results.items, Has("XYZ", CompletionItemKind::Class));
  EXPECT_THAT(Results.items, Has("foo", CompletionItemKind::Function));
  EXPECT_THAT(Results.items, Not(Has("No")));
}

TEST(CompletionTest, IndexBasedWithFilter) {
  clangd::CodeCompleteOptions Opts;
  auto I = simpleIndexFromSymbols({{"ns::XYZ", index::SymbolKind::Class},
                                   {"ns::foo", index::SymbolKind::Function}});
  Opts.Index = I.get();

  auto Results = completions(R"cpp(
      void f() { ns::x^ }
  )cpp",
                             Opts);
  EXPECT_THAT(Results.items, Contains(AllOf(Named("XYZ"), Filter("x"))));
  EXPECT_THAT(Results.items, Not(Has("foo")));
}

TEST(CompletionTest, GlobalQualified) {
  clangd::CodeCompleteOptions Opts;
  auto I = simpleIndexFromSymbols({{"XYZ", index::SymbolKind::Class}});
  Opts.Index = I.get();

  auto Results = completions(R"cpp(
      void f() { ::^ }
  )cpp",
                             Opts);
  EXPECT_THAT(Results.items, Has("XYZ", CompletionItemKind::Class));
}

TEST(CompletionTest, FullyQualifiedScope) {
  clangd::CodeCompleteOptions Opts;
  auto I = simpleIndexFromSymbols({{"ns::XYZ", index::SymbolKind::Class}});
  Opts.Index = I.get();

  auto Results = completions(R"cpp(
      void f() { ::ns::^ }
  )cpp",
                             Opts);
  EXPECT_THAT(Results.items, Has("XYZ", CompletionItemKind::Class));
}

TEST(CompletionTest, ASTIndexMultiFile) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true,
                      /*BuildDynamicSymbolIndex=*/true);

  Server
      .addDocument(Context::empty(), getVirtualTestFilePath("foo.cpp"), R"cpp(
      namespace ns { class XYZ {}; void foo() {} }
  )cpp")
      .wait();

  auto File = getVirtualTestFilePath("bar.cpp");
  Annotations Test(R"cpp(
      namespace ns { class XXX {}; void fooooo() {} }
      void f() { ns::^ }
  )cpp");
  Server.addDocument(Context::empty(), File, Test.code()).wait();

  auto Results = Server.codeComplete(Context::empty(), File, Test.point(), {})
                     .get()
                     .second.Value;
  // "XYZ" and "foo" are not included in the file being completed but are still
  // visible through the index.
  EXPECT_THAT(Results.items, Has("XYZ", CompletionItemKind::Class));
  EXPECT_THAT(Results.items, Has("foo", CompletionItemKind::Function));
  EXPECT_THAT(Results.items, Has("XXX", CompletionItemKind::Class));
  EXPECT_THAT(Results.items, Has("fooooo", CompletionItemKind::Function));
}

} // namespace
} // namespace clangd
} // namespace clang
