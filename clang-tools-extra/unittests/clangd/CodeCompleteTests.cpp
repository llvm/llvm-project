//===-- CodeCompleteTests.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "ClangdServer.h"
#include "Compiler.h"
#include "Protocol.h"
#include "TestFS.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
using namespace llvm;

class IgnoreDiagnostics : public DiagnosticsConsumer {
  void onDiagnosticsReady(
      PathRef File, Tagged<std::vector<DiagWithFixIts>> Diagnostics) override {}
};

struct StringWithPos {
  std::string Text;
  clangd::Position MarkerPos;
};

/// Returns location of "{mark}" substring in \p Text and removes it from \p
/// Text. Note that \p Text must contain exactly one occurence of "{mark}".
///
/// Marker name can be configured using \p MarkerName parameter.
StringWithPos parseTextMarker(StringRef Text, StringRef MarkerName = "mark") {
  SmallString<16> Marker;
  Twine("{" + MarkerName + "}").toVector(/*ref*/ Marker);

  std::size_t MarkerOffset = Text.find(Marker);
  assert(MarkerOffset != StringRef::npos && "{mark} wasn't found in Text.");

  std::string WithoutMarker;
  WithoutMarker += Text.take_front(MarkerOffset);
  WithoutMarker += Text.drop_front(MarkerOffset + Marker.size());
  assert(StringRef(WithoutMarker).find(Marker) == StringRef::npos &&
         "There were multiple occurences of {mark} inside Text");

  clangd::Position MarkerPos =
      clangd::offsetToPosition(WithoutMarker, MarkerOffset);
  return {std::move(WithoutMarker), MarkerPos};
}

class ClangdCompletionTest : public ::testing::Test {
protected:
  template <class Predicate>
  bool ContainsItemPred(CompletionList const &Items, Predicate Pred) {
    for (const auto &Item : Items.items) {
      if (Pred(Item))
        return true;
    }
    return false;
  }

  bool ContainsItem(CompletionList const &Items, StringRef Name) {
    return ContainsItemPred(Items, [Name](clangd::CompletionItem Item) {
      return Item.insertText == Name;
    });
    return false;
  }
};

TEST_F(ClangdCompletionTest, CheckContentsOverride) {
  MockFSProvider FS;
  IgnoreDiagnostics DiagConsumer;
  MockCompilationDatabase CDB;

  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true,
                      EmptyLogger::getInstance());

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  const auto SourceContents = R"cpp(
int aba;
int b =   ;
)cpp";

  const auto OverridenSourceContents = R"cpp(
int cbc;
int b =   ;
)cpp";

  // Use default options.
  CodeCompleteOptions CCOpts;
  // Complete after '=' sign. We need to be careful to keep the SourceContents'
  // size the same.
  // We complete on the 3rd line (2nd in zero-based numbering), because raw
  // string literal of the SourceContents starts with a newline(it's easy to
  // miss).
  Position CompletePos = {2, 8};
  FS.Files[FooCpp] = SourceContents;
  FS.ExpectedFile = FooCpp;

  // No need to sync reparses here as there are no asserts on diagnostics (or
  // other async operations).
  Server.addDocument(FooCpp, SourceContents);

  {
    auto CodeCompletionResults1 =
        Server.codeComplete(FooCpp, CompletePos, CCOpts, None).get().Value;
    EXPECT_TRUE(ContainsItem(CodeCompletionResults1, "aba"));
    EXPECT_FALSE(ContainsItem(CodeCompletionResults1, "cbc"));
  }

  {
    auto CodeCompletionResultsOverriden =
        Server
            .codeComplete(FooCpp, CompletePos, CCOpts,
                          StringRef(OverridenSourceContents))
            .get()
            .Value;
    EXPECT_TRUE(ContainsItem(CodeCompletionResultsOverriden, "cbc"));
    EXPECT_FALSE(ContainsItem(CodeCompletionResultsOverriden, "aba"));
  }

  {
    auto CodeCompletionResults2 =
        Server.codeComplete(FooCpp, CompletePos, CCOpts, None).get().Value;
    EXPECT_TRUE(ContainsItem(CodeCompletionResults2, "aba"));
    EXPECT_FALSE(ContainsItem(CodeCompletionResults2, "cbc"));
  }
}

TEST_F(ClangdCompletionTest, Limit) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  CDB.ExtraClangFlags.push_back("-xc++");
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true,
                      EmptyLogger::getInstance());

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  FS.Files[FooCpp] = "";
  FS.ExpectedFile = FooCpp;
  StringWithPos Completion = parseTextMarker(R"cpp(
struct ClassWithMembers {
  int AAA();
  int BBB();
  int CCC();
}
int main() { ClassWithMembers().{complete} }
      )cpp",
                                             "complete");
  Server.addDocument(FooCpp, Completion.Text);

  clangd::CodeCompleteOptions Opts;
  Opts.Limit = 2;

  /// For after-dot completion we must always get consistent results.
  auto Results = Server
                     .codeComplete(FooCpp, Completion.MarkerPos, Opts,
                                   StringRef(Completion.Text))
                     .get()
                     .Value;

  EXPECT_TRUE(Results.isIncomplete);
  EXPECT_EQ(Opts.Limit, Results.items.size());
  EXPECT_TRUE(ContainsItem(Results, "AAA"));
  EXPECT_TRUE(ContainsItem(Results, "BBB"));
  EXPECT_FALSE(ContainsItem(Results, "CCC"));
}

TEST_F(ClangdCompletionTest, Filter) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  CDB.ExtraClangFlags.push_back("-xc++");
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true,
                      EmptyLogger::getInstance());

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  FS.Files[FooCpp] = "";
  FS.ExpectedFile = FooCpp;
  const char *Body = R"cpp(
    int Abracadabra;
    int Alakazam;
    struct S {
      int FooBar;
      int FooBaz;
      int Qux;
    };
  )cpp";
  auto Complete = [&](StringRef Query) {
    StringWithPos Completion = parseTextMarker(
        formatv("{0} int main() { {1}{{complete}} }", Body, Query).str(),
        "complete");
    Server.addDocument(FooCpp, Completion.Text);
    return Server
        .codeComplete(FooCpp, Completion.MarkerPos,
                      clangd::CodeCompleteOptions(), StringRef(Completion.Text))
        .get()
        .Value;
  };

  auto Foba = Complete("S().Foba");
  EXPECT_TRUE(ContainsItem(Foba, "FooBar"));
  EXPECT_TRUE(ContainsItem(Foba, "FooBaz"));
  EXPECT_FALSE(ContainsItem(Foba, "Qux"));

  auto FR = Complete("S().FR");
  EXPECT_TRUE(ContainsItem(FR, "FooBar"));
  EXPECT_FALSE(ContainsItem(FR, "FooBaz"));
  EXPECT_FALSE(ContainsItem(FR, "Qux"));

  auto Op = Complete("S().opr");
  EXPECT_TRUE(ContainsItem(Op, "operator="));

  auto Aaa = Complete("aaa");
  EXPECT_TRUE(ContainsItem(Aaa, "Abracadabra"));
  EXPECT_TRUE(ContainsItem(Aaa, "Alakazam"));

  auto UA = Complete("_a");
  EXPECT_TRUE(ContainsItem(UA, "static_cast"));
  EXPECT_FALSE(ContainsItem(UA, "Abracadabra"));
}

TEST_F(ClangdCompletionTest, CompletionOptions) {
  MockFSProvider FS;
  IgnoreDiagnostics DiagConsumer;
  MockCompilationDatabase CDB;
  CDB.ExtraClangFlags.push_back("-xc++");

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  FS.Files[FooCpp] = "";
  FS.ExpectedFile = FooCpp;

  const auto GlobalCompletionSourceTemplate = R"cpp(
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

  {complete}
}
)cpp";
  const auto MemberCompletionSourceTemplate = R"cpp(
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

  ClassWithMembers().{complete}
}
)cpp";

  StringWithPos GlobalCompletion =
      parseTextMarker(GlobalCompletionSourceTemplate, "complete");
  StringWithPos MemberCompletion =
      parseTextMarker(MemberCompletionSourceTemplate, "complete");

  auto TestWithOpts = [&](clangd::CodeCompleteOptions Opts) {
    ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                        /*StorePreamblesInMemory=*/true,
                        EmptyLogger::getInstance());
    // No need to sync reparses here as there are no asserts on diagnostics (or
    // other async operations).
    Server.addDocument(FooCpp, GlobalCompletion.Text);

    StringRef MethodItemText = Opts.EnableSnippets ? "method()" : "method";
    StringRef GlobalFuncItemText =
        Opts.EnableSnippets ? "global_func()" : "global_func";

    /// For after-dot completion we must always get consistent results.
    {
      auto Results = Server
                         .codeComplete(FooCpp, MemberCompletion.MarkerPos, Opts,
                                       StringRef(MemberCompletion.Text))
                         .get()
                         .Value;

      // Class members. The only items that must be present in after-dor
      // completion.
      EXPECT_TRUE(ContainsItem(Results, MethodItemText));
      EXPECT_TRUE(ContainsItem(Results, MethodItemText));
      EXPECT_TRUE(ContainsItem(Results, "field"));
      EXPECT_EQ(Opts.IncludeIneligibleResults,
                ContainsItem(Results, "private_field"));
      // Global items.
      EXPECT_FALSE(ContainsItem(Results, "global_var"));
      EXPECT_FALSE(ContainsItem(Results, GlobalFuncItemText));
      EXPECT_FALSE(ContainsItem(Results, "GlobalClass"));
      // A macro.
      EXPECT_FALSE(ContainsItem(Results, "MACRO"));
      // Local items.
      EXPECT_FALSE(ContainsItem(Results, "LocalClass"));
      // There should be no code patterns (aka snippets) in after-dot
      // completion. At least there aren't any we're aware of.
      EXPECT_FALSE(
          ContainsItemPred(Results, [](clangd::CompletionItem const &Item) {
            return Item.kind == clangd::CompletionItemKind::Snippet;
          }));
      // Check documentation.
      EXPECT_EQ(
          Opts.IncludeBriefComments,
          ContainsItemPred(Results, [](clangd::CompletionItem const &Item) {
            return !Item.documentation.empty();
          }));
    }
    // Global completion differs based on the Opts that were passed.
    {
      auto Results = Server
                         .codeComplete(FooCpp, GlobalCompletion.MarkerPos, Opts,
                                       StringRef(GlobalCompletion.Text))
                         .get()
                         .Value;

      // Class members. Should never be present in global completions.
      EXPECT_FALSE(ContainsItem(Results, MethodItemText));
      EXPECT_FALSE(ContainsItem(Results, "field"));
      // Global items.
      EXPECT_EQ(ContainsItem(Results, "global_var"), Opts.IncludeGlobals);
      EXPECT_EQ(ContainsItem(Results, GlobalFuncItemText), Opts.IncludeGlobals);
      EXPECT_EQ(ContainsItem(Results, "GlobalClass"), Opts.IncludeGlobals);
      // A macro.
      EXPECT_EQ(ContainsItem(Results, "MACRO"), Opts.IncludeMacros);
      // Local items. Must be present always.
      EXPECT_TRUE(ContainsItem(Results, "local_var"));
      EXPECT_TRUE(ContainsItem(Results, "LocalClass"));
      // FIXME(ibiryukov): snippets have wrong Item.kind now. Reenable this
      // check after https://reviews.llvm.org/D38720 makes it in.
      //
      // Code patterns (aka snippets).
      // EXPECT_EQ(
      //     Opts.IncludeCodePatterns && Opts.EnableSnippets,
      //     ContainsItemPred(Results, [](clangd::CompletionItem const &Item) {
      //       return Item.kind == clangd::CompletionItemKind::Snippet;
      //     }));

      // Check documentation.
      EXPECT_EQ(
          Opts.IncludeBriefComments,
          ContainsItemPred(Results, [](clangd::CompletionItem const &Item) {
            return !Item.documentation.empty();
          }));
    }
  };

  clangd::CodeCompleteOptions CCOpts;
  for (bool IncludeMacros : {true, false}) {
    CCOpts.IncludeMacros = IncludeMacros;
    for (bool IncludeGlobals : {true, false}) {
      CCOpts.IncludeGlobals = IncludeGlobals;
      for (bool IncludeBriefComments : {true, false}) {
        CCOpts.IncludeBriefComments = IncludeBriefComments;
        for (bool EnableSnippets : {true, false}) {
          CCOpts.EnableSnippets = EnableSnippets;
          for (bool IncludeCodePatterns : {true, false}) {
            CCOpts.IncludeCodePatterns = IncludeCodePatterns;
            for (bool IncludeIneligibleResults : {true, false}) {
              CCOpts.IncludeIneligibleResults = IncludeIneligibleResults;
              TestWithOpts(CCOpts);
            }
          }
        }
      }
    }
  }
}

} // namespace
} // namespace clangd
} // namespace clang
