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

#include "../../clang-tidy/ClangTidyCheck.h"
#include "AST.h"
#include "CompileCommands.h"
#include "Compiler.h"
#include "Config.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "TidyProvider.h"
#include "support/Context.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <utility>

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAreArray;

MATCHER_P(declNamed, Name, "") {
  if (NamedDecl *ND = dyn_cast<NamedDecl>(arg))
    if (ND->getName() == Name)
      return true;
  if (auto *Stream = result_listener->stream()) {
    llvm::raw_os_ostream OS(*Stream);
    arg->dump(OS);
  }
  return false;
}

MATCHER_P(declKind, Kind, "") {
  if (NamedDecl *ND = dyn_cast<NamedDecl>(arg))
    if (ND->getDeclKindName() == llvm::StringRef(Kind))
      return true;
  if (auto *Stream = result_listener->stream()) {
    llvm::raw_os_ostream OS(*Stream);
    arg->dump(OS);
  }
  return false;
}

// Matches if the Decl has template args equal to ArgName. If the decl is a
// NamedDecl and ArgName is an empty string it also matches.
MATCHER_P(withTemplateArgs, ArgName, "") {
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

MATCHER_P(pragmaTrivia, P, "") { return arg.Trivia == P; }

MATCHER(eqInc, "") {
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
  TU.Code = R"cpp(
    int main();
    template <typename> bool X = true;
  )cpp";
  auto AST = TU.build();
  EXPECT_THAT(AST.getLocalTopLevelDecls(),
              testing::UnorderedElementsAreArray(
                  {AllOf(declNamed("main"), declKind("Function")),
                   AllOf(declNamed("X"), declKind("VarTemplate"))}));
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
  EXPECT_THAT(AST.getLocalTopLevelDecls(), ElementsAre(declNamed("main")));
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
              ElementsAre(declNamed("f"), declNamed("s")));
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

  auto AST = TU.build();
  EXPECT_THAT(
      AST.getLocalTopLevelDecls(),
      ElementsAreArray({AllOf(declNamed("f"), withTemplateArgs("")),
                        AllOf(declNamed("f"), withTemplateArgs("<bool>")),
                        AllOf(declNamed("f"), withTemplateArgs("<double>")),
                        AllOf(declNamed("V"), withTemplateArgs("")),
                        AllOf(declNamed("V"), withTemplateArgs("<T *>")),
                        AllOf(declNamed("V"), withTemplateArgs("<bool>")),
                        AllOf(declNamed("foo"), withTemplateArgs("")),
                        AllOf(declNamed("i"), withTemplateArgs("")),
                        AllOf(declNamed("d"), withTemplateArgs("")),
                        AllOf(declNamed("foo"), withTemplateArgs("<T *>")),
                        AllOf(declNamed("foo"), withTemplateArgs("<bool>"))}));
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
  TU.ClangTidyProvider = addTidyChecks("modernize-use-trailing-return-type");
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
  MockFS FS;
  FS.Files = {{testPath("foo.cpp"), "void test() {}"}};
  // Unknown flags should not prevent a build of compiler invocation.
  ParseInputs Inputs;
  Inputs.TFS = &FS;
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
  llvm::Annotations TestCase(R"cpp(
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
  std::vector<size_t> MacroExpansionPositions;
  for (const auto &SIDToRefs : AST.getMacros().MacroRefs) {
    for (const auto &R : SIDToRefs.second)
      MacroExpansionPositions.push_back(R.StartOffset);
  }
  for (const auto &R : AST.getMacros().UnknownMacros)
    MacroExpansionPositions.push_back(R.StartOffset);
  EXPECT_THAT(
      MacroExpansionPositions,
      testing::UnorderedElementsAreArray(TestCase.points()));
}

MATCHER_P(withFileName, Inc, "") { return arg.FileName == Inc; }

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
  MockFS FS;
  auto Inputs = TU.inputs(FS);
  auto CI = buildCompilerInvocation(Inputs, Diags);
  auto EmptyPreamble =
      buildPreamble(testPath("foo.cpp"), *CI, Inputs, true, nullptr);
  ASSERT_TRUE(EmptyPreamble);
  EXPECT_THAT(EmptyPreamble->Includes.MainFileIncludes, IsEmpty());

  // Now build an AST using empty preamble and ensure patched includes worked.
  TU.Code = ModifiedContents.str();
  Inputs = TU.inputs(FS);
  auto PatchedAST = ParsedAST::build(testPath("foo.cpp"), Inputs, std::move(CI),
                                     {}, EmptyPreamble);
  ASSERT_TRUE(PatchedAST);

  // Ensure source location information is correct, including resolved paths.
  EXPECT_THAT(PatchedAST->getIncludeStructure().MainFileIncludes,
              testing::Pointwise(
                  eqInc(), ExpectedAST.getIncludeStructure().MainFileIncludes));
  // Ensure file proximity signals are correct.
  auto &SM = PatchedAST->getSourceManager();
  auto &FM = SM.getFileManager();
  // Copy so that we can use operator[] to get the children.
  IncludeStructure Includes = PatchedAST->getIncludeStructure();
  auto MainFE = FM.getFile(testPath("foo.cpp"));
  ASSERT_TRUE(MainFE);
  auto MainID = Includes.getID(*MainFE);
  auto AuxFE = FM.getFile(testPath("sub/aux.h"));
  ASSERT_TRUE(AuxFE);
  auto AuxID = Includes.getID(*AuxFE);
  EXPECT_THAT(Includes.IncludeChildren[*MainID], Contains(*AuxID));
}

TEST(ParsedASTTest, PatchesDeletedIncludes) {
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.Code = "";
  auto ExpectedAST = TU.build();

  // Build preamble with no includes.
  TU.Code = R"cpp(#include <foo.h>)cpp";
  StoreDiags Diags;
  MockFS FS;
  auto Inputs = TU.inputs(FS);
  auto CI = buildCompilerInvocation(Inputs, Diags);
  auto BaselinePreamble =
      buildPreamble(testPath("foo.cpp"), *CI, Inputs, true, nullptr);
  ASSERT_TRUE(BaselinePreamble);
  EXPECT_THAT(BaselinePreamble->Includes.MainFileIncludes,
              ElementsAre(testing::Field(&Inclusion::Written, "<foo.h>")));

  // Now build an AST using additional includes and check that locations are
  // correctly parsed.
  TU.Code = "";
  Inputs = TU.inputs(FS);
  auto PatchedAST = ParsedAST::build(testPath("foo.cpp"), Inputs, std::move(CI),
                                     {}, BaselinePreamble);
  ASSERT_TRUE(PatchedAST);

  // Ensure source location information is correct.
  EXPECT_THAT(PatchedAST->getIncludeStructure().MainFileIncludes,
              testing::Pointwise(
                  eqInc(), ExpectedAST.getIncludeStructure().MainFileIncludes));
  // Ensure file proximity signals are correct.
  auto &SM = ExpectedAST.getSourceManager();
  auto &FM = SM.getFileManager();
  // Copy so that we can getOrCreateID().
  IncludeStructure Includes = ExpectedAST.getIncludeStructure();
  auto MainFE = FM.getFileRef(testPath("foo.cpp"));
  ASSERT_THAT_EXPECTED(MainFE, llvm::Succeeded());
  auto MainID = Includes.getOrCreateID(*MainFE);
  auto &PatchedFM = PatchedAST->getSourceManager().getFileManager();
  IncludeStructure PatchedIncludes = PatchedAST->getIncludeStructure();
  auto PatchedMainFE = PatchedFM.getFileRef(testPath("foo.cpp"));
  ASSERT_THAT_EXPECTED(PatchedMainFE, llvm::Succeeded());
  auto PatchedMainID = PatchedIncludes.getOrCreateID(*PatchedMainFE);
  EXPECT_EQ(Includes.includeDepth(MainID)[MainID],
            PatchedIncludes.includeDepth(PatchedMainID)[PatchedMainID]);
}

// Returns Code guarded by #ifndef guards
std::string guard(llvm::StringRef Code) {
  static int GuardID = 0;
  std::string GuardName = ("GUARD_" + llvm::Twine(++GuardID)).str();
  return llvm::formatv("#ifndef {0}\n#define {0}\n{1}\n#endif\n", GuardName,
                       Code);
}

std::string once(llvm::StringRef Code) {
  return llvm::formatv("#pragma once\n{0}\n", Code);
}

bool mainIsGuarded(const ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();
  const FileEntry *MainFE = SM.getFileEntryForID(SM.getMainFileID());
  return AST.getPreprocessor()
      .getHeaderSearchInfo()
      .isFileMultipleIncludeGuarded(MainFE);
}

MATCHER_P(diag, Desc, "") {
  return llvm::StringRef(arg.Message).contains(Desc);
}

// Check our understanding of whether the main file is header guarded or not.
TEST(ParsedASTTest, HeaderGuards) {
  TestTU TU;
  TU.ImplicitHeaderGuard = false;

  TU.Code = ";";
  EXPECT_FALSE(mainIsGuarded(TU.build()));

  TU.Code = guard(";");
  EXPECT_TRUE(mainIsGuarded(TU.build()));

  TU.Code = once(";");
  EXPECT_TRUE(mainIsGuarded(TU.build()));

  TU.Code = R"cpp(
    ;
    #pragma once
  )cpp";
  EXPECT_FALSE(mainIsGuarded(TU.build())); // FIXME: true

  TU.Code = R"cpp(
    ;
    #ifndef GUARD
    #define GUARD
    ;
    #endif
  )cpp";
  EXPECT_FALSE(mainIsGuarded(TU.build()));
}

// Check our handling of files that include themselves.
// Ideally we allow this if the file has header guards.
//
// Note: the semicolons (empty statements) are significant!
// - they force the preamble to end and the body to begin. Directives can have
//   different effects in the preamble vs main file (which we try to hide).
// - if the preamble would otherwise cover the whole file, a trailing semicolon
//   forces their sizes to be different. This is significant because the file
//   size is part of the lookup key for HeaderFileInfo, and we don't want to
//   rely on the preamble's HFI being looked up when parsing the main file.
TEST(ParsedASTTest, HeaderGuardsSelfInclude) {
  // Disable include cleaner diagnostics to prevent them from interfering with
  // other diagnostics.
  Config Cfg;
  Cfg.Diagnostics.MissingIncludes = Config::IncludesPolicy::None;
  Cfg.Diagnostics.UnusedIncludes = Config::IncludesPolicy::None;
  WithContextValue Ctx(Config::Key, std::move(Cfg));

  TestTU TU;
  TU.ImplicitHeaderGuard = false;
  TU.Filename = "self.h";

  TU.Code = R"cpp(
    #include "self.h" // error-ok
    ;
  )cpp";
  auto AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(),
              ElementsAre(diag("recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    ;
    #include "self.h" // error-ok
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), ElementsAre(diag("nested too deeply")));
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #pragma once
    #include "self.h"
    ;
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #pragma once
    ;
    #include "self.h"
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    ;
    #pragma once
    #include "self.h"
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #ifndef GUARD
    #define GUARD
    #include "self.h" // error-ok: FIXME, this would be nice to support
    #endif
    ;
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(),
              ElementsAre(diag("recursively when building a preamble")));
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #ifndef GUARD
    #define GUARD
    ;
    #include "self.h"
    #endif
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  // Guarded too late...
  TU.Code = R"cpp(
    #include "self.h" // error-ok
    #ifndef GUARD
    #define GUARD
    ;
    #endif
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(),
              ElementsAre(diag("recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #include "self.h" // error-ok
    ;
    #ifndef GUARD
    #define GUARD
    #endif
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(),
              ElementsAre(diag("recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    ;
    #ifndef GUARD
    #define GUARD
    #include "self.h"
    #endif
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), IsEmpty());
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #include "self.h" // error-ok
    #pragma once
    ;
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(),
              ElementsAre(diag("recursively when building a preamble")));
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #include "self.h" // error-ok
    ;
    #pragma once
  )cpp";
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(),
              ElementsAre(diag("recursively when building a preamble")));
  EXPECT_TRUE(mainIsGuarded(AST));
}

// Tests how we handle common idioms for splitting a header-only library
// into interface and implementation files (e.g. *.h vs *.inl).
// These files mutually include each other, and need careful handling of include
// guards (which interact with preambles).
TEST(ParsedASTTest, HeaderGuardsImplIface) {
  std::string Interface = R"cpp(
    // error-ok: we assert on diagnostics explicitly
    template <class T> struct Traits {
      unsigned size();
    };
    #include "impl.h"
  )cpp";
  std::string Implementation = R"cpp(
    // error-ok: we assert on diagnostics explicitly
    #include "iface.h"
    template <class T> unsigned Traits<T>::size() {
      return sizeof(T);
    }
  )cpp";

  TestTU TU;
  TU.ImplicitHeaderGuard = false; // We're testing include guard handling!
  TU.ExtraArgs.push_back("-xc++-header");

  // Editing the interface file, which is include guarded (easy case).
  // We mostly get this right via PP if we don't recognize the include guard.
  TU.Filename = "iface.h";
  TU.Code = guard(Interface);
  TU.AdditionalFiles = {{"impl.h", Implementation}};
  auto AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));
  // Slightly harder: the `#pragma once` is part of the preamble, and we
  // need to transfer it to the main file's HeaderFileInfo.
  TU.Code = once(Interface);
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  // Editing the implementation file, which is not include guarded.
  TU.Filename = "impl.h";
  TU.Code = Implementation;
  TU.AdditionalFiles = {{"iface.h", guard(Interface)}};
  AST = TU.build();
  // The diagnostic is unfortunate in this case, but correct per our model.
  // Ultimately the include is skipped and the code is parsed correctly though.
  EXPECT_THAT(AST.getDiagnostics(),
              ElementsAre(diag("in included file: main file cannot be included "
                               "recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));
  // Interface is pragma once guarded, same thing.
  TU.AdditionalFiles = {{"iface.h", once(Interface)}};
  AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(),
              ElementsAre(diag("in included file: main file cannot be included "
                               "recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));
}

TEST(ParsedASTTest, DiscoversPragmaMarks) {
  TestTU TU;
  TU.AdditionalFiles["Header.h"] = R"(
    #pragma mark - Something API
    int something();
    #pragma mark Something else
  )";
  TU.Code = R"cpp(
    #include "Header.h"
    #pragma mark In Preamble
    #pragma mark - Something Impl
    int something() { return 1; }
    #pragma mark End
  )cpp";
  auto AST = TU.build();

  EXPECT_THAT(AST.getMarks(), ElementsAre(pragmaTrivia(" In Preamble"),
                                          pragmaTrivia(" - Something Impl"),
                                          pragmaTrivia(" End")));
}

TEST(ParsedASTTest, GracefulFailureOnAssemblyFile) {
  std::string Filename = "TestTU.S";
  std::string Code = R"S(
main:
    # test comment
    bx lr
  )S";

  // The rest is a simplified version of TestTU::build().
  // Don't call TestTU::build() itself because it would assert on
  // failure to build an AST.
  MockFS FS;
  std::string FullFilename = testPath(Filename);
  FS.Files[FullFilename] = Code;
  ParseInputs Inputs;
  auto &Argv = Inputs.CompileCommand.CommandLine;
  Argv = {"clang"};
  Argv.push_back(FullFilename);
  Inputs.CompileCommand.Filename = FullFilename;
  Inputs.CompileCommand.Directory = testRoot();
  Inputs.Contents = Code;
  Inputs.TFS = &FS;
  StoreDiags Diags;
  auto CI = buildCompilerInvocation(Inputs, Diags);
  assert(CI && "Failed to build compilation invocation.");
  auto AST = ParsedAST::build(FullFilename, Inputs, std::move(CI), {}, nullptr);

  EXPECT_FALSE(AST.has_value())
      << "Should not try to build AST for assembly source file";
}

} // namespace
} // namespace clangd
} // namespace clang
