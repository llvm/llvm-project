//===- TokensTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.def"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Lex/Token.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include <cassert>
#include <cstdlib>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <ostream>
#include <string>

using namespace clang;
using namespace clang::syntax;

using llvm::ValueIs;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::Not;
using ::testing::Pointee;
using ::testing::StartsWith;

namespace {
// Checks the passed ArrayRef<T> has the same begin() and end() iterators as the
// argument.
MATCHER_P(SameRange, A, "") {
  return A.begin() == arg.begin() && A.end() == arg.end();
}

Matcher<TokenBuffer::Expansion>
IsExpansion(Matcher<llvm::ArrayRef<syntax::Token>> Spelled,
            Matcher<llvm::ArrayRef<syntax::Token>> Expanded) {
  return AllOf(Field(&TokenBuffer::Expansion::Spelled, Spelled),
               Field(&TokenBuffer::Expansion::Expanded, Expanded));
}
// Matchers for syntax::Token.
MATCHER_P(Kind, K, "") { return arg.kind() == K; }
MATCHER_P2(HasText, Text, SourceMgr, "") {
  return arg.text(*SourceMgr) == Text;
}
/// Checks the start and end location of a token are equal to SourceRng.
MATCHER_P(RangeIs, SourceRng, "") {
  return arg.location() == SourceRng.first &&
         arg.endLocation() == SourceRng.second;
}

class TokenCollectorTest : public ::testing::Test {
public:
  /// Run the clang frontend, collect the preprocessed tokens from the frontend
  /// invocation and store them in this->Buffer.
  /// This also clears SourceManager before running the compiler.
  void recordTokens(llvm::StringRef Code) {
    class RecordTokens : public ASTFrontendAction {
    public:
      explicit RecordTokens(TokenBuffer &Result) : Result(Result) {}

      bool BeginSourceFileAction(CompilerInstance &CI) override {
        assert(!Collector && "expected only a single call to BeginSourceFile");
        Collector.emplace(CI.getPreprocessor());
        return true;
      }
      void EndSourceFileAction() override {
        assert(Collector && "BeginSourceFileAction was never called");
        Result = std::move(*Collector).consume();
        Result.indexExpandedTokens();
      }

      std::unique_ptr<ASTConsumer>
      CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        return std::make_unique<ASTConsumer>();
      }

    private:
      TokenBuffer &Result;
      llvm::Optional<TokenCollector> Collector;
    };

    constexpr const char *FileName = "./input.cpp";
    FS->addFile(FileName, time_t(), llvm::MemoryBuffer::getMemBufferCopy(""));
    // Prepare to run a compiler.
    if (!Diags->getClient())
      Diags->setClient(new IgnoringDiagConsumer);
    std::vector<const char *> Args = {"tok-test", "-std=c++03", "-fsyntax-only",
                                      FileName};
    CreateInvocationOptions CIOpts;
    CIOpts.Diags = Diags;
    CIOpts.VFS = FS;
    auto CI = createInvocation(Args, std::move(CIOpts));
    assert(CI);
    CI->getFrontendOpts().DisableFree = false;
    CI->getPreprocessorOpts().addRemappedFile(
        FileName, llvm::MemoryBuffer::getMemBufferCopy(Code).release());
    CompilerInstance Compiler;
    Compiler.setInvocation(std::move(CI));
    Compiler.setDiagnostics(Diags.get());
    Compiler.setFileManager(FileMgr.get());
    Compiler.setSourceManager(SourceMgr.get());

    this->Buffer = TokenBuffer(*SourceMgr);
    RecordTokens Recorder(this->Buffer);
    ASSERT_TRUE(Compiler.ExecuteAction(Recorder))
        << "failed to run the frontend";
  }

  /// Record the tokens and return a test dump of the resulting buffer.
  std::string collectAndDump(llvm::StringRef Code) {
    recordTokens(Code);
    return Buffer.dumpForTests();
  }

  // Adds a file to the test VFS.
  void addFile(llvm::StringRef Path, llvm::StringRef Contents) {
    if (!FS->addFile(Path, time_t(),
                     llvm::MemoryBuffer::getMemBufferCopy(Contents))) {
      ADD_FAILURE() << "could not add a file to VFS: " << Path;
    }
  }

  /// Add a new file, run syntax::tokenize() on the range if any, run it on the
  /// whole file otherwise and return the results.
  std::vector<syntax::Token> tokenize(llvm::StringRef Text) {
    llvm::Annotations Annot(Text);
    auto FID = SourceMgr->createFileID(
        llvm::MemoryBuffer::getMemBufferCopy(Annot.code()));
    // FIXME: pass proper LangOptions.
    if (Annot.ranges().empty())
      return syntax::tokenize(FID, *SourceMgr, LangOptions());
    return syntax::tokenize(
        syntax::FileRange(FID, Annot.range().Begin, Annot.range().End),
        *SourceMgr, LangOptions());
  }

  // Specialized versions of matchers that hide the SourceManager from clients.
  Matcher<syntax::Token> HasText(std::string Text) const {
    return ::HasText(Text, SourceMgr.get());
  }
  Matcher<syntax::Token> RangeIs(llvm::Annotations::Range R) const {
    std::pair<SourceLocation, SourceLocation> Ls;
    Ls.first = SourceMgr->getLocForStartOfFile(SourceMgr->getMainFileID())
                   .getLocWithOffset(R.Begin);
    Ls.second = SourceMgr->getLocForStartOfFile(SourceMgr->getMainFileID())
                    .getLocWithOffset(R.End);
    return ::RangeIs(Ls);
  }

  /// Finds a subrange in O(n * m).
  template <class T, class U, class Eq>
  llvm::ArrayRef<T> findSubrange(llvm::ArrayRef<U> Subrange,
                                 llvm::ArrayRef<T> Range, Eq F) {
    assert(Subrange.size() >= 1);
    if (Range.size() < Subrange.size())
      return llvm::ArrayRef(Range.end(), Range.end());
    for (auto Begin = Range.begin(), Last = Range.end() - Subrange.size();
         Begin <= Last; ++Begin) {
      auto It = Begin;
      for (auto ItSub = Subrange.begin(); ItSub != Subrange.end();
           ++ItSub, ++It) {
        if (!F(*ItSub, *It))
          goto continue_outer;
      }
      return llvm::ArrayRef(Begin, It);
    continue_outer:;
    }
    return llvm::ArrayRef(Range.end(), Range.end());
  }

  /// Finds a subrange in \p Tokens that match the tokens specified in \p Query.
  /// The match should be unique. \p Query is a whitespace-separated list of
  /// tokens to search for.
  llvm::ArrayRef<syntax::Token>
  findTokenRange(llvm::StringRef Query, llvm::ArrayRef<syntax::Token> Tokens) {
    llvm::SmallVector<llvm::StringRef, 8> QueryTokens;
    Query.split(QueryTokens, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (QueryTokens.empty()) {
      ADD_FAILURE() << "will not look for an empty list of tokens";
      std::abort();
    }
    // An equality test for search.
    auto TextMatches = [this](llvm::StringRef Q, const syntax::Token &T) {
      return Q == T.text(*SourceMgr);
    };
    // Find a match.
    auto Found = findSubrange(llvm::ArrayRef(QueryTokens), Tokens, TextMatches);
    if (Found.begin() == Tokens.end()) {
      ADD_FAILURE() << "could not find the subrange for " << Query;
      std::abort();
    }
    // Check that the match is unique.
    if (findSubrange(llvm::ArrayRef(QueryTokens),
                     llvm::ArrayRef(Found.end(), Tokens.end()), TextMatches)
            .begin() != Tokens.end()) {
      ADD_FAILURE() << "match is not unique for " << Query;
      std::abort();
    }
    return Found;
  };

  // Specialized versions of findTokenRange for expanded and spelled tokens.
  llvm::ArrayRef<syntax::Token> findExpanded(llvm::StringRef Query) {
    return findTokenRange(Query, Buffer.expandedTokens());
  }
  llvm::ArrayRef<syntax::Token> findSpelled(llvm::StringRef Query,
                                            FileID File = FileID()) {
    if (!File.isValid())
      File = SourceMgr->getMainFileID();
    return findTokenRange(Query, Buffer.spelledTokens(File));
  }

  // Data fields.
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      new DiagnosticsEngine(new DiagnosticIDs, new DiagnosticOptions);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
      new llvm::vfs::InMemoryFileSystem;
  llvm::IntrusiveRefCntPtr<FileManager> FileMgr =
      new FileManager(FileSystemOptions(), FS);
  llvm::IntrusiveRefCntPtr<SourceManager> SourceMgr =
      new SourceManager(*Diags, *FileMgr);
  /// Contains last result of calling recordTokens().
  TokenBuffer Buffer = TokenBuffer(*SourceMgr);
};

TEST_F(TokenCollectorTest, RawMode) {
  EXPECT_THAT(tokenize("int main() {}"),
              ElementsAre(Kind(tok::kw_int),
                          AllOf(HasText("main"), Kind(tok::identifier)),
                          Kind(tok::l_paren), Kind(tok::r_paren),
                          Kind(tok::l_brace), Kind(tok::r_brace)));
  // Comments are ignored for now.
  EXPECT_THAT(tokenize("/* foo */int a; // more comments"),
              ElementsAre(Kind(tok::kw_int),
                          AllOf(HasText("a"), Kind(tok::identifier)),
                          Kind(tok::semi)));
  EXPECT_THAT(tokenize("int [[main() {]]}"),
              ElementsAre(AllOf(HasText("main"), Kind(tok::identifier)),
                          Kind(tok::l_paren), Kind(tok::r_paren),
                          Kind(tok::l_brace)));
  EXPECT_THAT(tokenize("int [[main() {   ]]}"),
              ElementsAre(AllOf(HasText("main"), Kind(tok::identifier)),
                          Kind(tok::l_paren), Kind(tok::r_paren),
                          Kind(tok::l_brace)));
  // First token is partially parsed, last token is fully included even though
  // only a part of it is contained in the range.
  EXPECT_THAT(tokenize("int m[[ain() {ret]]urn 0;}"),
              ElementsAre(AllOf(HasText("ain"), Kind(tok::identifier)),
                          Kind(tok::l_paren), Kind(tok::r_paren),
                          Kind(tok::l_brace), Kind(tok::kw_return)));
}

TEST_F(TokenCollectorTest, Basic) {
  std::pair</*Input*/ std::string, /*Expected*/ std::string> TestCases[] = {
      {"int main() {}",
       R"(expanded tokens:
  int main ( ) { }
file './input.cpp'
  spelled tokens:
    int main ( ) { }
  no mappings.
)"},
      // All kinds of whitespace are ignored.
      {"\t\n  int\t\n  main\t\n  (\t\n  )\t\n{\t\n  }\t\n",
       R"(expanded tokens:
  int main ( ) { }
file './input.cpp'
  spelled tokens:
    int main ( ) { }
  no mappings.
)"},
      // Annotation tokens are ignored.
      {R"cpp(
        #pragma GCC visibility push (public)
        #pragma GCC visibility pop
      )cpp",
       R"(expanded tokens:
  <empty>
file './input.cpp'
  spelled tokens:
    # pragma GCC visibility push ( public ) # pragma GCC visibility pop
  mappings:
    ['#'_0, '<eof>'_13) => ['<eof>'_0, '<eof>'_0)
)"},
      // Empty files should not crash.
      {R"cpp()cpp", R"(expanded tokens:
  <empty>
file './input.cpp'
  spelled tokens:
    <empty>
  no mappings.
)"},
      // Should not crash on errors inside '#define' directives. Error is that
      // stringification (#B) does not refer to a macro parameter.
      {
          R"cpp(
a
#define MACRO() A #B
)cpp",
          R"(expanded tokens:
  a
file './input.cpp'
  spelled tokens:
    a # define MACRO ( ) A # B
  mappings:
    ['#'_1, '<eof>'_9) => ['<eof>'_1, '<eof>'_1)
)"}};
  for (auto &Test : TestCases)
    EXPECT_EQ(collectAndDump(Test.first), Test.second)
        << collectAndDump(Test.first);
}

TEST_F(TokenCollectorTest, Locations) {
  // Check locations of the tokens.
  llvm::Annotations Code(R"cpp(
    $r1[[int]] $r2[[a]] $r3[[=]] $r4[["foo bar baz"]] $r5[[;]]
  )cpp");
  recordTokens(Code.code());
  // Check expanded tokens.
  EXPECT_THAT(
      Buffer.expandedTokens(),
      ElementsAre(AllOf(Kind(tok::kw_int), RangeIs(Code.range("r1"))),
                  AllOf(Kind(tok::identifier), RangeIs(Code.range("r2"))),
                  AllOf(Kind(tok::equal), RangeIs(Code.range("r3"))),
                  AllOf(Kind(tok::string_literal), RangeIs(Code.range("r4"))),
                  AllOf(Kind(tok::semi), RangeIs(Code.range("r5"))),
                  Kind(tok::eof)));
  // Check spelled tokens.
  EXPECT_THAT(
      Buffer.spelledTokens(SourceMgr->getMainFileID()),
      ElementsAre(AllOf(Kind(tok::kw_int), RangeIs(Code.range("r1"))),
                  AllOf(Kind(tok::identifier), RangeIs(Code.range("r2"))),
                  AllOf(Kind(tok::equal), RangeIs(Code.range("r3"))),
                  AllOf(Kind(tok::string_literal), RangeIs(Code.range("r4"))),
                  AllOf(Kind(tok::semi), RangeIs(Code.range("r5")))));

  auto StartLoc = SourceMgr->getLocForStartOfFile(SourceMgr->getMainFileID());
  for (auto &R : Code.ranges()) {
    EXPECT_THAT(Buffer.spelledTokenAt(StartLoc.getLocWithOffset(R.Begin)),
                Pointee(RangeIs(R)));
  }
}

TEST_F(TokenCollectorTest, MacroDirectives) {
  // Macro directives are not stored anywhere at the moment.
  std::string Code = R"cpp(
    #define FOO a
    #include "unresolved_file.h"
    #undef FOO
    #ifdef X
    #else
    #endif
    #ifndef Y
    #endif
    #if 1
    #elif 2
    #else
    #endif
    #pragma once
    #pragma something lalala

    int a;
  )cpp";
  std::string Expected =
      "expanded tokens:\n"
      "  int a ;\n"
      "file './input.cpp'\n"
      "  spelled tokens:\n"
      "    # define FOO a # include \"unresolved_file.h\" # undef FOO "
      "# ifdef X # else # endif # ifndef Y # endif # if 1 # elif 2 # else "
      "# endif # pragma once # pragma something lalala int a ;\n"
      "  mappings:\n"
      "    ['#'_0, 'int'_39) => ['int'_0, 'int'_0)\n";
  EXPECT_EQ(collectAndDump(Code), Expected);
}

TEST_F(TokenCollectorTest, MacroReplacements) {
  std::pair</*Input*/ std::string, /*Expected*/ std::string> TestCases[] = {
      // A simple object-like macro.
      {R"cpp(
    #define INT int const
    INT a;
  )cpp",
       R"(expanded tokens:
  int const a ;
file './input.cpp'
  spelled tokens:
    # define INT int const INT a ;
  mappings:
    ['#'_0, 'INT'_5) => ['int'_0, 'int'_0)
    ['INT'_5, 'a'_6) => ['int'_0, 'a'_2)
)"},
      // A simple function-like macro.
      {R"cpp(
    #define INT(a) const int
    INT(10+10) a;
  )cpp",
       R"(expanded tokens:
  const int a ;
file './input.cpp'
  spelled tokens:
    # define INT ( a ) const int INT ( 10 + 10 ) a ;
  mappings:
    ['#'_0, 'INT'_8) => ['const'_0, 'const'_0)
    ['INT'_8, 'a'_14) => ['const'_0, 'a'_2)
)"},
      // Recursive macro replacements.
      {R"cpp(
    #define ID(X) X
    #define INT int const
    ID(ID(INT)) a;
  )cpp",
       R"(expanded tokens:
  int const a ;
file './input.cpp'
  spelled tokens:
    # define ID ( X ) X # define INT int const ID ( ID ( INT ) ) a ;
  mappings:
    ['#'_0, 'ID'_12) => ['int'_0, 'int'_0)
    ['ID'_12, 'a'_19) => ['int'_0, 'a'_2)
)"},
      // A little more complicated recursive macro replacements.
      {R"cpp(
    #define ADD(X, Y) X+Y
    #define MULT(X, Y) X*Y

    int a = ADD(MULT(1,2), MULT(3,ADD(4,5)));
  )cpp",
       "expanded tokens:\n"
       "  int a = 1 * 2 + 3 * 4 + 5 ;\n"
       "file './input.cpp'\n"
       "  spelled tokens:\n"
       "    # define ADD ( X , Y ) X + Y # define MULT ( X , Y ) X * Y int "
       "a = ADD ( MULT ( 1 , 2 ) , MULT ( 3 , ADD ( 4 , 5 ) ) ) ;\n"
       "  mappings:\n"
       "    ['#'_0, 'int'_22) => ['int'_0, 'int'_0)\n"
       "    ['ADD'_25, ';'_46) => ['1'_3, ';'_12)\n"},
      // Empty macro replacement.
      // FIXME: the #define directives should not be glued together.
      {R"cpp(
    #define EMPTY
    #define EMPTY_FUNC(X)
    EMPTY
    EMPTY_FUNC(1+2+3)
    )cpp",
       R"(expanded tokens:
  <empty>
file './input.cpp'
  spelled tokens:
    # define EMPTY # define EMPTY_FUNC ( X ) EMPTY EMPTY_FUNC ( 1 + 2 + 3 )
  mappings:
    ['#'_0, 'EMPTY'_9) => ['<eof>'_0, '<eof>'_0)
    ['EMPTY'_9, 'EMPTY_FUNC'_10) => ['<eof>'_0, '<eof>'_0)
    ['EMPTY_FUNC'_10, '<eof>'_18) => ['<eof>'_0, '<eof>'_0)
)"},
      // File ends with a macro replacement.
      {R"cpp(
    #define FOO 10+10;
    int a = FOO
    )cpp",
       R"(expanded tokens:
  int a = 10 + 10 ;
file './input.cpp'
  spelled tokens:
    # define FOO 10 + 10 ; int a = FOO
  mappings:
    ['#'_0, 'int'_7) => ['int'_0, 'int'_0)
    ['FOO'_10, '<eof>'_11) => ['10'_3, '<eof>'_7)
)"},
      {R"cpp(
         #define NUM 42
         #define ID(a) a
         #define M 1 + ID
         M(NUM)
       )cpp",
       R"(expanded tokens:
  1 + 42
file './input.cpp'
  spelled tokens:
    # define NUM 42 # define ID ( a ) a # define M 1 + ID M ( NUM )
  mappings:
    ['#'_0, 'M'_17) => ['1'_0, '1'_0)
    ['M'_17, '<eof>'_21) => ['1'_0, '<eof>'_3)
)"},
  };

  for (auto &Test : TestCases) {
    std::string Dump = collectAndDump(Test.first);
    EXPECT_EQ(Test.second, Dump) << Dump;
  }
}

TEST_F(TokenCollectorTest, SpecialTokens) {
  // Tokens coming from concatenations.
  recordTokens(R"cpp(
    #define CONCAT(a, b) a ## b
    int a = CONCAT(1, 2);
  )cpp");
  EXPECT_THAT(std::vector<syntax::Token>(Buffer.expandedTokens()),
              Contains(HasText("12")));
  // Multi-line tokens with slashes at the end.
  recordTokens("i\\\nn\\\nt");
  EXPECT_THAT(Buffer.expandedTokens(),
              ElementsAre(AllOf(Kind(tok::kw_int), HasText("i\\\nn\\\nt")),
                          Kind(tok::eof)));
  // FIXME: test tokens with digraphs and UCN identifiers.
}

TEST_F(TokenCollectorTest, LateBoundTokens) {
  // The parser eventually breaks the first '>>' into two tokens ('>' and '>'),
  // but we choose to record them as a single token (for now).
  llvm::Annotations Code(R"cpp(
    template <class T>
    struct foo { int a; };
    int bar = foo<foo<int$br[[>>]]().a;
    int baz = 10 $op[[>>]] 2;
  )cpp");
  recordTokens(Code.code());
  EXPECT_THAT(std::vector<syntax::Token>(Buffer.expandedTokens()),
              AllOf(Contains(AllOf(Kind(tok::greatergreater),
                                   RangeIs(Code.range("br")))),
                    Contains(AllOf(Kind(tok::greatergreater),
                                   RangeIs(Code.range("op"))))));
}

TEST_F(TokenCollectorTest, DelayedParsing) {
  llvm::StringLiteral Code = R"cpp(
    struct Foo {
      int method() {
        // Parser will visit method bodies and initializers multiple times, but
        // TokenBuffer should only record the first walk over the tokens;
        return 100;
      }
      int a = 10;

      struct Subclass {
        void foo() {
          Foo().method();
        }
      };
    };
  )cpp";
  std::string ExpectedTokens =
      "expanded tokens:\n"
      "  struct Foo { int method ( ) { return 100 ; } int a = 10 ; struct "
      "Subclass { void foo ( ) { Foo ( ) . method ( ) ; } } ; } ;\n";
  EXPECT_THAT(collectAndDump(Code), StartsWith(ExpectedTokens));
}

TEST_F(TokenCollectorTest, MultiFile) {
  addFile("./foo.h", R"cpp(
    #define ADD(X, Y) X+Y
    int a = 100;
    #include "bar.h"
  )cpp");
  addFile("./bar.h", R"cpp(
    int b = ADD(1, 2);
    #define MULT(X, Y) X*Y
  )cpp");
  llvm::StringLiteral Code = R"cpp(
    #include "foo.h"
    int c = ADD(1, MULT(2,3));
  )cpp";

  std::string Expected = R"(expanded tokens:
  int a = 100 ; int b = 1 + 2 ; int c = 1 + 2 * 3 ;
file './input.cpp'
  spelled tokens:
    # include "foo.h" int c = ADD ( 1 , MULT ( 2 , 3 ) ) ;
  mappings:
    ['#'_0, 'int'_3) => ['int'_12, 'int'_12)
    ['ADD'_6, ';'_17) => ['1'_15, ';'_20)
file './foo.h'
  spelled tokens:
    # define ADD ( X , Y ) X + Y int a = 100 ; # include "bar.h"
  mappings:
    ['#'_0, 'int'_11) => ['int'_0, 'int'_0)
    ['#'_16, '<eof>'_19) => ['int'_5, 'int'_5)
file './bar.h'
  spelled tokens:
    int b = ADD ( 1 , 2 ) ; # define MULT ( X , Y ) X * Y
  mappings:
    ['ADD'_3, ';'_9) => ['1'_8, ';'_11)
    ['#'_10, '<eof>'_21) => ['int'_12, 'int'_12)
)";

  EXPECT_EQ(Expected, collectAndDump(Code))
      << "input: " << Code << "\nresults: " << collectAndDump(Code);
}

class TokenBufferTest : public TokenCollectorTest {};

TEST_F(TokenBufferTest, SpelledByExpanded) {
  recordTokens(R"cpp(
    a1 a2 a3 b1 b2
  )cpp");

  // Expanded and spelled tokens are stored separately.
  EXPECT_THAT(findExpanded("a1 a2"), Not(SameRange(findSpelled("a1 a2"))));
  // Searching for subranges of expanded tokens should give the corresponding
  // spelled ones.
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a1 a2 a3 b1 b2")),
              ValueIs(SameRange(findSpelled("a1 a2 a3 b1 b2"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a1 a2 a3")),
              ValueIs(SameRange(findSpelled("a1 a2 a3"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("b1 b2")),
              ValueIs(SameRange(findSpelled("b1 b2"))));

  // Test search on simple macro expansions.
  recordTokens(R"cpp(
    #define A a1 a2 a3
    #define B b1 b2

    A split B
  )cpp");
  // Ranges going across expansion boundaries.
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a1 a2 a3 split b1 b2")),
              ValueIs(SameRange(findSpelled("A split B"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a1 a2 a3")),
              ValueIs(SameRange(findSpelled("A split").drop_back())));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("b1 b2")),
              ValueIs(SameRange(findSpelled("split B").drop_front())));
  // Ranges not fully covering macro invocations should fail.
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("a1 a2")), std::nullopt);
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("b2")), std::nullopt);
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("a2 a3 split b1 b2")),
            std::nullopt);

  // Recursive macro invocations.
  recordTokens(R"cpp(
    #define ID(x) x
    #define B b1 b2

    ID(ID(ID(a1) a2 a3)) split ID(B)
  )cpp");

  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("b1 b2")),
              ValueIs(SameRange(findSpelled("( B").drop_front())));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a1 a2 a3 split b1 b2")),
              ValueIs(SameRange(findSpelled(
                  "ID ( ID ( ID ( a1 ) a2 a3 ) ) split ID ( B )"))));
  // Mixed ranges with expanded and spelled tokens.
  EXPECT_THAT(
      Buffer.spelledForExpanded(findExpanded("a1 a2 a3 split")),
      ValueIs(SameRange(findSpelled("ID ( ID ( ID ( a1 ) a2 a3 ) ) split"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("split b1 b2")),
              ValueIs(SameRange(findSpelled("split ID ( B )"))));
  // Macro arguments
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a1")),
              ValueIs(SameRange(findSpelled("a1"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a2")),
              ValueIs(SameRange(findSpelled("a2"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a3")),
              ValueIs(SameRange(findSpelled("a3"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a1 a2")),
              ValueIs(SameRange(findSpelled("ID ( a1 ) a2"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a1 a2 a3")),
              ValueIs(SameRange(findSpelled("ID ( a1 ) a2 a3"))));

  // Empty macro expansions.
  recordTokens(R"cpp(
    #define EMPTY
    #define ID(X) X

    EMPTY EMPTY ID(1 2 3) EMPTY EMPTY split1
    EMPTY EMPTY ID(4 5 6) split2
    ID(7 8 9) EMPTY EMPTY
  )cpp");
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("1 2 3")),
              ValueIs(SameRange(findSpelled("1 2 3"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("4 5 6")),
              ValueIs(SameRange(findSpelled("4 5 6"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("7 8 9")),
              ValueIs(SameRange(findSpelled("7 8 9"))));

  // Empty mappings coming from various directives.
  recordTokens(R"cpp(
    #define ID(X) X
    ID(1)
    #pragma lalala
    not_mapped
  )cpp");
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("not_mapped")),
              ValueIs(SameRange(findSpelled("not_mapped"))));

  // Multiple macro arguments
  recordTokens(R"cpp(
    #define ID(X) X
    #define ID2(X, Y) X Y

    ID2(ID(a1), ID(a2) a3) ID2(a4, a5 a6 a7)
  )cpp");
  // Should fail, spans multiple arguments.
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("a1 a2")), std::nullopt);
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a2 a3")),
              ValueIs(SameRange(findSpelled("ID ( a2 ) a3"))));
  EXPECT_THAT(
      Buffer.spelledForExpanded(findExpanded("a1 a2 a3")),
      ValueIs(SameRange(findSpelled("ID2 ( ID ( a1 ) , ID ( a2 ) a3 )"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a5 a6")),
              ValueIs(SameRange(findSpelled("a5 a6"))));
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("a4 a5 a6 a7")),
              ValueIs(SameRange(findSpelled("ID2 ( a4 , a5 a6 a7 )"))));
  // Should fail, spans multiple invocations.
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("a1 a2 a3 a4")),
            std::nullopt);

  // https://github.com/clangd/clangd/issues/1289
  recordTokens(R"cpp(
    #define FOO(X) foo(X)
    #define INDIRECT FOO(y)
    INDIRECT // expands to foo(y)
  )cpp");
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("y")), std::nullopt);

  recordTokens(R"cpp(
    #define FOO(X) a X b
    FOO(y)
  )cpp");
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("y")),
              ValueIs(SameRange(findSpelled("y"))));

  recordTokens(R"cpp(
    #define ID(X) X
    #define BAR ID(1)
    BAR
  )cpp");
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("1")),
              ValueIs(SameRange(findSpelled(") BAR").drop_front())));

  // Critical cases for mapping of Prev/Next in spelledForExpandedSlow.
  recordTokens(R"cpp(
    #define ID(X) X
    ID(prev ID(good))
    #define LARGE ID(prev ID(bad))
    LARGE
  )cpp");
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("good")),
              ValueIs(SameRange(findSpelled("good"))));
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("bad")), std::nullopt);

  recordTokens(R"cpp(
    #define PREV prev
    #define ID(X) X
    PREV ID(good)
    #define LARGE PREV ID(bad)
    LARGE
  )cpp");
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("good")),
            ValueIs(SameRange(findSpelled("good"))));
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("bad")), std::nullopt);

  recordTokens(R"cpp(
    #define ID(X) X
    #define ID2(X, Y) X Y
    ID2(prev, ID(good))
    #define LARGE ID2(prev, bad)
    LARGE
  )cpp");
  EXPECT_THAT(Buffer.spelledForExpanded(findExpanded("good")),
            ValueIs(SameRange(findSpelled("good"))));
  EXPECT_EQ(Buffer.spelledForExpanded(findExpanded("bad")), std::nullopt);
}

TEST_F(TokenBufferTest, ExpandedTokensForRange) {
  recordTokens(R"cpp(
    #define SIGN(X) X##_washere
    A SIGN(B) C SIGN(D) E SIGN(F) G
  )cpp");

  SourceRange R(findExpanded("C").front().location(),
                findExpanded("F_washere").front().location());
  // Expanded and spelled tokens are stored separately.
  EXPECT_THAT(Buffer.expandedTokens(R),
              SameRange(findExpanded("C D_washere E F_washere")));
  EXPECT_THAT(Buffer.expandedTokens(SourceRange()), testing::IsEmpty());
}

TEST_F(TokenBufferTest, ExpansionsOverlapping) {
  // Object-like macro expansions.
  recordTokens(R"cpp(
    #define FOO 3+4
    int a = FOO 1;
    int b = FOO 2;
  )cpp");

  llvm::ArrayRef<syntax::Token> Foo1 = findSpelled("FOO 1");
  EXPECT_THAT(
      Buffer.expansionStartingAt(Foo1.data()),
      ValueIs(IsExpansion(SameRange(Foo1.drop_back()),
                          SameRange(findExpanded("3 + 4 1").drop_back()))));
  EXPECT_THAT(
      Buffer.expansionsOverlapping(Foo1),
      ElementsAre(IsExpansion(SameRange(Foo1.drop_back()),
                              SameRange(findExpanded("3 + 4 1").drop_back()))));

  llvm::ArrayRef<syntax::Token> Foo2 = findSpelled("FOO 2");
  EXPECT_THAT(
      Buffer.expansionStartingAt(Foo2.data()),
      ValueIs(IsExpansion(SameRange(Foo2.drop_back()),
                          SameRange(findExpanded("3 + 4 2").drop_back()))));
  EXPECT_THAT(
      Buffer.expansionsOverlapping(llvm::ArrayRef(Foo1.begin(), Foo2.end())),
      ElementsAre(IsExpansion(SameRange(Foo1.drop_back()), _),
                  IsExpansion(SameRange(Foo2.drop_back()), _)));

  // Function-like macro expansions.
  recordTokens(R"cpp(
    #define ID(X) X
    int a = ID(1+2+3);
    int b = ID(ID(2+3+4));
  )cpp");

  llvm::ArrayRef<syntax::Token> ID1 = findSpelled("ID ( 1 + 2 + 3 )");
  EXPECT_THAT(Buffer.expansionStartingAt(&ID1.front()),
              ValueIs(IsExpansion(SameRange(ID1),
                                  SameRange(findExpanded("1 + 2 + 3")))));
  // Only the first spelled token should be found.
  for (const auto &T : ID1.drop_front())
    EXPECT_EQ(Buffer.expansionStartingAt(&T), std::nullopt);

  llvm::ArrayRef<syntax::Token> ID2 = findSpelled("ID ( ID ( 2 + 3 + 4 ) )");
  EXPECT_THAT(Buffer.expansionStartingAt(&ID2.front()),
              ValueIs(IsExpansion(SameRange(ID2),
                                  SameRange(findExpanded("2 + 3 + 4")))));
  // Only the first spelled token should be found.
  for (const auto &T : ID2.drop_front())
    EXPECT_EQ(Buffer.expansionStartingAt(&T), std::nullopt);

  EXPECT_THAT(Buffer.expansionsOverlapping(llvm::ArrayRef(
                  findSpelled("1 + 2").data(), findSpelled("4").data())),
              ElementsAre(IsExpansion(SameRange(ID1), _),
                          IsExpansion(SameRange(ID2), _)));

  // PP directives.
  recordTokens(R"cpp(
#define FOO 1
int a = FOO;
#pragma once
int b = 1;
  )cpp");

  llvm::ArrayRef<syntax::Token> DefineFoo = findSpelled("# define FOO 1");
  EXPECT_THAT(
      Buffer.expansionStartingAt(&DefineFoo.front()),
      ValueIs(IsExpansion(SameRange(DefineFoo),
                          SameRange(findExpanded("int a").take_front(0)))));
  // Only the first spelled token should be found.
  for (const auto &T : DefineFoo.drop_front())
    EXPECT_EQ(Buffer.expansionStartingAt(&T), std::nullopt);

  llvm::ArrayRef<syntax::Token> PragmaOnce = findSpelled("# pragma once");
  EXPECT_THAT(
      Buffer.expansionStartingAt(&PragmaOnce.front()),
      ValueIs(IsExpansion(SameRange(PragmaOnce),
                          SameRange(findExpanded("int b").take_front(0)))));
  // Only the first spelled token should be found.
  for (const auto &T : PragmaOnce.drop_front())
    EXPECT_EQ(Buffer.expansionStartingAt(&T), std::nullopt);

  EXPECT_THAT(
      Buffer.expansionsOverlapping(findSpelled("FOO ; # pragma")),
      ElementsAre(IsExpansion(SameRange(findSpelled("FOO ;").drop_back()), _),
                  IsExpansion(SameRange(PragmaOnce), _)));
}

TEST_F(TokenBufferTest, TokensToFileRange) {
  addFile("./foo.h", "token_from_header");
  llvm::Annotations Code(R"cpp(
    #define FOO token_from_expansion
    #include "./foo.h"
    $all[[$i[[int]] a = FOO;]]
  )cpp");
  recordTokens(Code.code());

  auto &SM = *SourceMgr;

  // Two simple examples.
  auto Int = findExpanded("int").front();
  auto Semi = findExpanded(";").front();
  EXPECT_EQ(Int.range(SM), FileRange(SM.getMainFileID(), Code.range("i").Begin,
                                     Code.range("i").End));
  EXPECT_EQ(syntax::Token::range(SM, Int, Semi),
            FileRange(SM.getMainFileID(), Code.range("all").Begin,
                      Code.range("all").End));
  // We don't test assertion failures because death tests are slow.
}

TEST_F(TokenBufferTest, MacroExpansions) {
  llvm::Annotations Code(R"cpp(
    #define FOO B
    #define FOO2 BA
    #define CALL(X) int X
    #define G CALL(FOO2)
    int B;
    $macro[[FOO]];
    $macro[[CALL]](A);
    $macro[[G]];
  )cpp");
  recordTokens(Code.code());
  auto &SM = *SourceMgr;
  auto Expansions = Buffer.macroExpansions(SM.getMainFileID());
  std::vector<FileRange> ExpectedMacroRanges;
  for (auto Range : Code.ranges("macro"))
    ExpectedMacroRanges.push_back(
        FileRange(SM.getMainFileID(), Range.Begin, Range.End));
  std::vector<FileRange> ActualMacroRanges;
  for (auto Expansion : Expansions)
    ActualMacroRanges.push_back(Expansion->range(SM));
  EXPECT_EQ(ExpectedMacroRanges, ActualMacroRanges);
}

TEST_F(TokenBufferTest, Touching) {
  llvm::Annotations Code("^i^nt^ ^a^b^=^1;^");
  recordTokens(Code.code());

  auto Touching = [&](int Index) {
    SourceLocation Loc = SourceMgr->getComposedLoc(SourceMgr->getMainFileID(),
                                                   Code.points()[Index]);
    return spelledTokensTouching(Loc, Buffer);
  };
  auto Identifier = [&](int Index) {
    SourceLocation Loc = SourceMgr->getComposedLoc(SourceMgr->getMainFileID(),
                                                   Code.points()[Index]);
    const syntax::Token *Tok = spelledIdentifierTouching(Loc, Buffer);
    return Tok ? Tok->text(*SourceMgr) : "";
  };

  EXPECT_THAT(Touching(0), SameRange(findSpelled("int")));
  EXPECT_EQ(Identifier(0), "");
  EXPECT_THAT(Touching(1), SameRange(findSpelled("int")));
  EXPECT_EQ(Identifier(1), "");
  EXPECT_THAT(Touching(2), SameRange(findSpelled("int")));
  EXPECT_EQ(Identifier(2), "");

  EXPECT_THAT(Touching(3), SameRange(findSpelled("ab")));
  EXPECT_EQ(Identifier(3), "ab");
  EXPECT_THAT(Touching(4), SameRange(findSpelled("ab")));
  EXPECT_EQ(Identifier(4), "ab");

  EXPECT_THAT(Touching(5), SameRange(findSpelled("ab =")));
  EXPECT_EQ(Identifier(5), "ab");

  EXPECT_THAT(Touching(6), SameRange(findSpelled("= 1")));
  EXPECT_EQ(Identifier(6), "");

  EXPECT_THAT(Touching(7), SameRange(findSpelled(";")));
  EXPECT_EQ(Identifier(7), "");

  ASSERT_EQ(Code.points().size(), 8u);
}

TEST_F(TokenBufferTest, ExpandedBySpelled) {
  recordTokens(R"cpp(
    a1 a2 a3 b1 b2
  )cpp");
  // Expanded and spelled tokens are stored separately.
  EXPECT_THAT(findExpanded("a1 a2"), Not(SameRange(findSpelled("a1 a2"))));
  // Searching for subranges of expanded tokens should give the corresponding
  // spelled ones.
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("a1 a2 a3 b1 b2")),
              ElementsAre(SameRange(findExpanded("a1 a2 a3 b1 b2"))));
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("a1 a2 a3")),
              ElementsAre(SameRange(findExpanded("a1 a2 a3"))));
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("b1 b2")),
              ElementsAre(SameRange(findExpanded("b1 b2"))));

  // Test search on simple macro expansions.
  recordTokens(R"cpp(
    #define A a1 a2 a3
    #define B b1 b2

    A split B
  )cpp");
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("A split B")),
              ElementsAre(SameRange(findExpanded("a1 a2 a3 split b1 b2"))));
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("A split").drop_back()),
              ElementsAre(SameRange(findExpanded("a1 a2 a3"))));
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("split B").drop_front()),
              ElementsAre(SameRange(findExpanded("b1 b2"))));

  // Ranges not fully covering macro expansions should fail.
  recordTokens(R"cpp(
    #define ID(x) x

    ID(a)
  )cpp");
  // Spelled don't cover entire mapping (missing ID token) -> empty result
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("( a )")), IsEmpty());
  // Spelled don't cover entire mapping (missing ) token) -> empty result
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("ID ( a")), IsEmpty());

  // Recursive macro invocations.
  recordTokens(R"cpp(
    #define ID(x) x
    #define B b1 b2

    ID(ID(ID(a1) a2 a3)) split ID(B)
  )cpp");

  EXPECT_THAT(
      Buffer.expandedForSpelled(findSpelled("ID ( ID ( ID ( a1 ) a2 a3 ) )")),
      ElementsAre(SameRange(findExpanded("a1 a2 a3"))));
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("ID ( B )")),
              ElementsAre(SameRange(findExpanded("b1 b2"))));
  EXPECT_THAT(Buffer.expandedForSpelled(
                  findSpelled("ID ( ID ( ID ( a1 ) a2 a3 ) ) split ID ( B )")),
              ElementsAre(SameRange(findExpanded("a1 a2 a3 split b1 b2"))));
  // FIXME: these should succeed, but we do not support macro arguments yet.
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("a1")), IsEmpty());
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("ID ( a1 ) a2")),
              IsEmpty());

  // Empty macro expansions.
  recordTokens(R"cpp(
    #define EMPTY
    #define ID(X) X

    EMPTY EMPTY ID(1 2 3) EMPTY EMPTY split1
    EMPTY EMPTY ID(4 5 6) split2
    ID(7 8 9) EMPTY EMPTY
  )cpp");
  // Covered by empty expansions on one of both of the sides.
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("ID ( 1 2 3 )")),
              ElementsAre(SameRange(findExpanded("1 2 3"))));
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("ID ( 4 5 6 )")),
              ElementsAre(SameRange(findExpanded("4 5 6"))));
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("ID ( 7 8 9 )")),
              ElementsAre(SameRange(findExpanded("7 8 9"))));
  // Including the empty macro expansions on the side.
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("EMPTY ID ( 1 2 3 )")),
              ElementsAre(SameRange(findExpanded("1 2 3"))));
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("ID ( 1 2 3 ) EMPTY")),
              ElementsAre(SameRange(findExpanded("1 2 3"))));
  EXPECT_THAT(
      Buffer.expandedForSpelled(findSpelled("EMPTY ID ( 1 2 3 ) EMPTY")),
      ElementsAre(SameRange(findExpanded("1 2 3"))));

  // Empty mappings coming from various directives.
  recordTokens(R"cpp(
    #define ID(X) X
    ID(1)
    #pragma lalala
    not_mapped
  )cpp");
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("# define ID ( X ) X")),
              IsEmpty());
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("# pragma lalala")),
              IsEmpty());

  // Empty macro expansion.
  recordTokens(R"cpp(
    #define EMPTY
    EMPTY int a = 100;
  )cpp");
  EXPECT_THAT(Buffer.expandedForSpelled(findSpelled("EMPTY int").drop_back()),
              IsEmpty());
}

TEST_F(TokenCollectorTest, Pragmas) {
  // Tokens coming from concatenations.
  recordTokens(R"cpp(
    void foo() {
      #pragma unroll 4
      for(int i=0;i<4;++i);
    }
  )cpp");
}
} // namespace
