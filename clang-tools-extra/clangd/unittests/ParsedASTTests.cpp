//===-- ParsedASTTests.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tests cover clangd's logic to build a TU, which generally uses the APIs
// in ParsedAST and Preamble, via the TestTU helper.
//
//===----------------------------------------------------------------------===//

#include "../../clang-tidy/ClangTidyModule.h"
#include "../../clang-tidy/ClangTidyModuleRegistry.h"
#include "AST.h"
#include "Annotations.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Token.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

MATCHER_P(DeclNamed, Name, "") {
  if (NamedDecl *ND = dyn_cast<NamedDecl>(arg))
    if (ND->getName() == Name)
      return true;
  if (auto *Stream = result_listener->stream()) {
    llvm::raw_os_ostream OS(*Stream);
    arg->dump(OS);
  }
  return false;
}

// Matches if the Decl has template args equal to ArgName. If the decl is a
// NamedDecl and ArgName is an empty string it also matches.
MATCHER_P(WithTemplateArgs, ArgName, "") {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(arg)) {
    if (const auto *Args = FD->getTemplateSpecializationArgs()) {
      std::string SpecializationArgs;
      // Without the PrintingPolicy "bool" will be printed as "_Bool".
      LangOptions LO;
      PrintingPolicy Policy(LO);
      Policy.adjustForCPlusPlus();
      for (const auto &Arg : Args->asArray()) {
        if (SpecializationArgs.size() > 0)
          SpecializationArgs += ",";
        SpecializationArgs += Arg.getAsType().getAsString(Policy);
      }
      if (Args->size() == 0)
        return ArgName == SpecializationArgs;
      return ArgName == "<" + SpecializationArgs + ">";
    }
  }
  if (const NamedDecl *ND = dyn_cast<NamedDecl>(arg))
    return printTemplateSpecializationArgs(*ND) == ArgName;
  return false;
}

MATCHER_P(RangeIs, R, "") {
  return arg.beginOffset() == R.Begin && arg.endOffset() == R.End;
}

MATCHER(EqInc, "") {
  Inclusion Actual = testing::get<0>(arg);
  Inclusion Expected = testing::get<1>(arg);
  return std::tie(Actual.HashLine, Actual.Written) ==
         std::tie(Expected.HashLine, Expected.Written);
}

TEST(ParsedASTTest, TopLevelDecls) {
  TestTU TU;
  TU.HeaderCode = R"(
    int header1();
    int header2;
  )";
  TU.Code = "int main();";
  auto AST = TU.build();
  EXPECT_THAT(AST.getLocalTopLevelDecls(), ElementsAre(DeclNamed("main")));
}

TEST(ParsedASTTest, DoesNotGetIncludedTopDecls) {
  TestTU TU;
  TU.HeaderCode = R"cpp(
    #define LL void foo(){}
    template<class T>
    struct H {
      H() {}
      LL
    };
  )cpp";
  TU.Code = R"cpp(
    int main() {
      H<int> h;
      h.foo();
    }
  )cpp";
  auto AST = TU.build();
  EXPECT_THAT(AST.getLocalTopLevelDecls(), ElementsAre(DeclNamed("main")));
}

TEST(ParsedASTTest, DoesNotGetImplicitTemplateTopDecls) {
  TestTU TU;
  TU.Code = R"cpp(
    template<typename T>
    void f(T) {}
    void s() {
      f(10UL);
    }
  )cpp";

  auto AST = TU.build();
  EXPECT_THAT(AST.getLocalTopLevelDecls(),
              ElementsAre(DeclNamed("f"), DeclNamed("s")));
}

TEST(ParsedASTTest,
     GetsExplicitInstantiationAndSpecializationTemplateTopDecls) {
  TestTU TU;
  TU.Code = R"cpp(
    template <typename T>
    void f(T) {}
    template<>
    void f(bool);
    template void f(double);

    template <class T>
    struct V {};
    template<class T>
    struct V<T*> {};
    template <>
    struct V<bool> {};

    template<class T>
    T foo = T(10);
    int i = foo<int>;
    double d = foo<double>;

    template <class T>
    int foo<T*> = 0;
    template <>
    int foo<bool> = 0;
  )cpp";
  // FIXME: Auto-completion in a template requires disabling delayed template
  // parsing.
  TU.ExtraArgs.push_back("-fno-delayed-template-parsing");

  auto AST = TU.build();
  EXPECT_THAT(
      AST.getLocalTopLevelDecls(),
      ElementsAreArray({AllOf(DeclNamed("f"), WithTemplateArgs("")),
                        AllOf(DeclNamed("f"), WithTemplateArgs("<bool>")),
                        AllOf(DeclNamed("f"), WithTemplateArgs("<double>")),
                        AllOf(DeclNamed("V"), WithTemplateArgs("")),
                        AllOf(DeclNamed("V"), WithTemplateArgs("<T *>")),
                        AllOf(DeclNamed("V"), WithTemplateArgs("<bool>")),
                        AllOf(DeclNamed("foo"), WithTemplateArgs("")),
                        AllOf(DeclNamed("i"), WithTemplateArgs("")),
                        AllOf(DeclNamed("d"), WithTemplateArgs("")),
                        AllOf(DeclNamed("foo"), WithTemplateArgs("<T *>")),
                        AllOf(DeclNamed("foo"), WithTemplateArgs("<bool>"))}));
}

TEST(ParsedASTTest, IgnoresDelayedTemplateParsing) {
  auto TU = TestTU::withCode(R"cpp(
    template <typename T> void xxx() {
      int yyy = 0;
    }
  )cpp");
  TU.ExtraArgs.push_back("-fdelayed-template-parsing");
  auto AST = TU.build();
  EXPECT_EQ(Decl::Var, findUnqualifiedDecl(AST, "yyy").getKind());
}

TEST(ParsedASTTest, TokensAfterPreamble) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"(
    int foo();
  )";
  TU.Code = R"cpp(
      #include "foo.h"
      first_token;
      void test() {
        // error-ok: invalid syntax, just examining token stream
      }
      last_token
)cpp";
  auto AST = TU.build();
  const syntax::TokenBuffer &T = AST.getTokens();
  const auto &SM = AST.getSourceManager();

  ASSERT_GT(T.expandedTokens().size(), 2u);
  // Check first token after the preamble.
  EXPECT_EQ(T.expandedTokens().front().text(SM), "first_token");
  // Last token is always 'eof'.
  EXPECT_EQ(T.expandedTokens().back().kind(), tok::eof);
  // Check the token before 'eof'.
  EXPECT_EQ(T.expandedTokens().drop_back().back().text(SM), "last_token");

  // The spelled tokens for the main file should have everything.
  auto Spelled = T.spelledTokens(SM.getMainFileID());
  ASSERT_FALSE(Spelled.empty());
  EXPECT_EQ(Spelled.front().kind(), tok::hash);
  EXPECT_EQ(Spelled.back().text(SM), "last_token");
}

TEST(ParsedASTTest, NoCrashOnTokensWithTidyCheck) {
  TestTU TU;
  // this check runs the preprocessor, we need to make sure it does not break
  // our recording logic.
  TU.ClangTidyChecks = "modernize-use-trailing-return-type";
  TU.Code = "inline int foo() {}";

  auto AST = TU.build();
  const syntax::TokenBuffer &T = AST.getTokens();
  const auto &SM = AST.getSourceManager();

  ASSERT_GT(T.expandedTokens().size(), 7u);
  // Check first token after the preamble.
  EXPECT_EQ(T.expandedTokens().front().text(SM), "inline");
  // Last token is always 'eof'.
  EXPECT_EQ(T.expandedTokens().back().kind(), tok::eof);
  // Check the token before 'eof'.
  EXPECT_EQ(T.expandedTokens().drop_back().back().text(SM), "}");
}

TEST(ParsedASTTest, CanBuildInvocationWithUnknownArgs) {
  // Unknown flags should not prevent a build of compiler invocation.
  ParseInputs Inputs;
  Inputs.FS = buildTestFS({{testPath("foo.cpp"), "void test() {}"}});
  Inputs.CompileCommand.CommandLine = {"clang", "-fsome-unknown-flag",
                                       testPath("foo.cpp")};
  IgnoreDiagnostics IgnoreDiags;
  EXPECT_NE(buildCompilerInvocation(Inputs, IgnoreDiags), nullptr);

  // Unknown forwarded to -cc1 should not a failure either.
  Inputs.CompileCommand.CommandLine = {
      "clang", "-Xclang", "-fsome-unknown-flag", testPath("foo.cpp")};
  EXPECT_NE(buildCompilerInvocation(Inputs, IgnoreDiags), nullptr);
}

TEST(ParsedASTTest, CollectsMainFileMacroExpansions) {
  Annotations TestCase(R"cpp(
    #define ^MACRO_ARGS(X, Y) X Y
    // - preamble ends
    ^ID(int A);
    // Macro arguments included.
    ^MACRO_ARGS(^MACRO_ARGS(^MACRO_EXP(int), E), ^ID(= 2));

    // Macro names inside other macros not included.
    #define ^MACRO_ARGS2(X, Y) X Y
    #define ^FOO BAR
    #define ^BAR 1
    int F = ^FOO;

    // Macros from token concatenations not included.
    #define ^CONCAT(X) X##A()
    #define ^PREPEND(X) MACRO##X()
    #define ^MACROA() 123
    int G = ^CONCAT(MACRO);
    int H = ^PREPEND(A);

    // Macros included not from preamble not included.
    #include "foo.inc"

    int printf(const char*, ...);
    void exit(int);
    #define ^assert(COND) if (!(COND)) { printf("%s", #COND); exit(0); }

    void test() {
      // Includes macro expansions in arguments that are expressions
      ^assert(0 <= ^BAR);
    }

    #ifdef ^UNDEFINED
    #endif

    #define ^MULTIPLE_DEFINITION 1
    #undef ^MULTIPLE_DEFINITION

    #define ^MULTIPLE_DEFINITION 2
    #undef ^MULTIPLE_DEFINITION
  )cpp");
  auto TU = TestTU::withCode(TestCase.code());
  TU.HeaderCode = R"cpp(
    #define ID(X) X
    #define MACRO_EXP(X) ID(X)
    MACRO_EXP(int B);
  )cpp";
  TU.AdditionalFiles["foo.inc"] = R"cpp(
    int C = ID(1);
    #define DEF 1
    int D = DEF;
  )cpp";
  ParsedAST AST = TU.build();
  std::vector<Position> MacroExpansionPositions;
  for (const auto &SIDToRefs : AST.getMacros().MacroRefs) {
    for (const auto &R : SIDToRefs.second)
      MacroExpansionPositions.push_back(R.start);
  }
  for (const auto &R : AST.getMacros().UnknownMacros)
    MacroExpansionPositions.push_back(R.start);
  EXPECT_THAT(MacroExpansionPositions,
              testing::UnorderedElementsAreArray(TestCase.points()));
}

TEST(ParsedASTTest, ReplayPreambleForTidyCheckers) {
  struct Inclusion {
    Inclusion(const SourceManager &SM, SourceLocation HashLoc,
              const Token &IncludeTok, llvm::StringRef FileName, bool IsAngled,
              CharSourceRange FilenameRange)
        : HashOffset(SM.getDecomposedLoc(HashLoc).second), IncTok(IncludeTok),
          IncDirective(IncludeTok.getIdentifierInfo()->getName()),
          FileNameOffset(SM.getDecomposedLoc(FilenameRange.getBegin()).second),
          FileName(FileName), IsAngled(IsAngled) {}
    size_t HashOffset;
    syntax::Token IncTok;
    llvm::StringRef IncDirective;
    size_t FileNameOffset;
    llvm::StringRef FileName;
    bool IsAngled;
  };
  static std::vector<Inclusion> Includes;
  static std::vector<syntax::Token> SkippedFiles;
  struct ReplayPreamblePPCallback : public PPCallbacks {
    const SourceManager &SM;
    explicit ReplayPreamblePPCallback(const SourceManager &SM) : SM(SM) {}

    void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                            StringRef FileName, bool IsAngled,
                            CharSourceRange FilenameRange, const FileEntry *,
                            StringRef, StringRef, const Module *,
                            SrcMgr::CharacteristicKind) override {
      Includes.emplace_back(SM, HashLoc, IncludeTok, FileName, IsAngled,
                            FilenameRange);
    }

    void FileSkipped(const FileEntryRef &, const Token &FilenameTok,
                     SrcMgr::CharacteristicKind) override {
      SkippedFiles.emplace_back(FilenameTok);
    }
  };
  struct ReplayPreambleCheck : public tidy::ClangTidyCheck {
    ReplayPreambleCheck(StringRef Name, tidy::ClangTidyContext *Context)
        : ClangTidyCheck(Name, Context) {}
    void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                             Preprocessor *ModuleExpanderPP) override {
      PP->addPPCallbacks(::std::make_unique<ReplayPreamblePPCallback>(SM));
    }
  };
  struct ReplayPreambleModule : public tidy::ClangTidyModule {
    void
    addCheckFactories(tidy::ClangTidyCheckFactories &CheckFactories) override {
      CheckFactories.registerCheck<ReplayPreambleCheck>(
          "replay-preamble-check");
    }
  };

  static tidy::ClangTidyModuleRegistry::Add<ReplayPreambleModule> X(
      "replay-preamble-module", "");
  TestTU TU;
  // This check records inclusion directives replayed by clangd.
  TU.ClangTidyChecks = "replay-preamble-check";
  llvm::Annotations Test(R"cpp(
    $hash^#$include[[import]] $filebegin^"$filerange[[bar.h]]"
    $hash^#$include[[include_next]] $filebegin^"$filerange[[baz.h]]"
    $hash^#$include[[include]] $filebegin^<$filerange[[a.h]]>)cpp");
  llvm::StringRef Code = Test.code();
  TU.Code = Code.str();
  TU.AdditionalFiles["bar.h"] = "";
  TU.AdditionalFiles["baz.h"] = "";
  TU.AdditionalFiles["a.h"] = "";
  // Since we are also testing #import directives, and they don't make much
  // sense in c++ (also they actually break on windows), just set language to
  // obj-c.
  TU.ExtraArgs = {"-isystem.", "-xobjective-c"};

  const auto &AST = TU.build();
  const auto &SM = AST.getSourceManager();

  auto HashLocs = Test.points("hash");
  ASSERT_EQ(HashLocs.size(), Includes.size());
  auto IncludeRanges = Test.ranges("include");
  ASSERT_EQ(IncludeRanges.size(), Includes.size());
  auto FileBeginLocs = Test.points("filebegin");
  ASSERT_EQ(FileBeginLocs.size(), Includes.size());
  auto FileRanges = Test.ranges("filerange");
  ASSERT_EQ(FileRanges.size(), Includes.size());

  ASSERT_EQ(SkippedFiles.size(), Includes.size());
  for (size_t I = 0; I < Includes.size(); ++I) {
    const auto &Inc = Includes[I];

    EXPECT_EQ(Inc.HashOffset, HashLocs[I]);

    auto IncRange = IncludeRanges[I];
    EXPECT_THAT(Inc.IncTok.range(SM), RangeIs(IncRange));
    EXPECT_EQ(Inc.IncTok.kind(), tok::identifier);
    EXPECT_EQ(Inc.IncDirective,
              Code.substr(IncRange.Begin, IncRange.End - IncRange.Begin));

    EXPECT_EQ(Inc.FileNameOffset, FileBeginLocs[I]);
    EXPECT_EQ(Inc.IsAngled, Code[FileBeginLocs[I]] == '<');

    auto FileRange = FileRanges[I];
    EXPECT_EQ(Inc.FileName,
              Code.substr(FileRange.Begin, FileRange.End - FileRange.Begin));

    EXPECT_EQ(SM.getDecomposedLoc(SkippedFiles[I].location()).second,
              Inc.FileNameOffset);
    // This also contains quotes/angles so increment the range by one from both
    // sides.
    EXPECT_EQ(
        SkippedFiles[I].text(SM),
        Code.substr(FileRange.Begin - 1, FileRange.End - FileRange.Begin + 2));
    EXPECT_EQ(SkippedFiles[I].kind(), tok::header_name);
  }
}

TEST(ParsedASTTest, PatchesAdditionalIncludes) {
  llvm::StringLiteral ModifiedContents = R"cpp(
    #include "baz.h"
    #include "foo.h"
    #include "sub/aux.h"
    void bar() {
      foo();
      baz();
      aux();
    })cpp";
  // Build expected ast with symbols coming from headers.
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.AdditionalFiles["foo.h"] = "void foo();";
  TU.AdditionalFiles["sub/baz.h"] = "void baz();";
  TU.AdditionalFiles["sub/aux.h"] = "void aux();";
  TU.ExtraArgs = {"-I" + testPath("sub")};
  TU.Code = ModifiedContents.str();
  auto ExpectedAST = TU.build();

  // Build preamble with no includes.
  TU.Code = "";
  StoreDiags Diags;
  auto Inputs = TU.inputs();
  auto CI = buildCompilerInvocation(Inputs, Diags);
  auto EmptyPreamble =
      buildPreamble(testPath("foo.cpp"), *CI, Inputs, true, nullptr);
  ASSERT_TRUE(EmptyPreamble);
  EXPECT_THAT(EmptyPreamble->Includes.MainFileIncludes, testing::IsEmpty());

  // Now build an AST using empty preamble and ensure patched includes worked.
  TU.Code = ModifiedContents.str();
  Inputs = TU.inputs();
  auto PatchedAST = ParsedAST::build(testPath("foo.cpp"), Inputs, std::move(CI),
                                     {}, EmptyPreamble);
  ASSERT_TRUE(PatchedAST);
  ASSERT_TRUE(PatchedAST->getDiagnostics().empty());

  // Ensure source location information is correct, including resolved paths.
  EXPECT_THAT(PatchedAST->getIncludeStructure().MainFileIncludes,
              testing::Pointwise(
                  EqInc(), ExpectedAST.getIncludeStructure().MainFileIncludes));
  auto StringMapToVector = [](const llvm::StringMap<unsigned> SM) {
    std::vector<std::pair<std::string, unsigned>> Res;
    for (const auto &E : SM)
      Res.push_back({E.first().str(), E.second});
    llvm::sort(Res);
    return Res;
  };
  // Ensure file proximity signals are correct.
  EXPECT_EQ(StringMapToVector(PatchedAST->getIncludeStructure().includeDepth(
                testPath("foo.cpp"))),
            StringMapToVector(ExpectedAST.getIncludeStructure().includeDepth(
                testPath("foo.cpp"))));
}

TEST(ParsedASTTest, PatchesDeletedIncludes) {
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.Code = "";
  auto ExpectedAST = TU.build();

  // Build preamble with no includes.
  TU.Code = R"cpp(#include <foo.h>)cpp";
  StoreDiags Diags;
  auto Inputs = TU.inputs();
  auto CI = buildCompilerInvocation(Inputs, Diags);
  auto BaselinePreamble =
      buildPreamble(testPath("foo.cpp"), *CI, Inputs, true, nullptr);
  ASSERT_TRUE(BaselinePreamble);
  EXPECT_THAT(BaselinePreamble->Includes.MainFileIncludes,
              ElementsAre(testing::Field(&Inclusion::Written, "<foo.h>")));

  // Now build an AST using additional includes and check that locations are
  // correctly parsed.
  TU.Code = "";
  Inputs = TU.inputs();
  auto PatchedAST = ParsedAST::build(testPath("foo.cpp"), Inputs, std::move(CI),
                                     {}, BaselinePreamble);
  ASSERT_TRUE(PatchedAST);

  // Ensure source location information is correct.
  EXPECT_THAT(PatchedAST->getIncludeStructure().MainFileIncludes,
              testing::Pointwise(
                  EqInc(), ExpectedAST.getIncludeStructure().MainFileIncludes));
  auto StringMapToVector = [](const llvm::StringMap<unsigned> SM) {
    std::vector<std::pair<std::string, unsigned>> Res;
    for (const auto &E : SM)
      Res.push_back({E.first().str(), E.second});
    llvm::sort(Res);
    return Res;
  };
  // Ensure file proximity signals are correct.
  EXPECT_EQ(StringMapToVector(PatchedAST->getIncludeStructure().includeDepth(
                testPath("foo.cpp"))),
            StringMapToVector(ExpectedAST.getIncludeStructure().includeDepth(
                testPath("foo.cpp"))));
}

} // namespace
} // namespace clangd
} // namespace clang
