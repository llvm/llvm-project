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

#include "AST.h"
#include "Annotations.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "ParsedAST.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Syntax/Tokens.h"
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
      for (const auto Arg : Args->asArray()) {
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

TEST(ParsedASTTest, TokensAfterPreamble) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"(
    int foo();
  )";
  TU.Code = R"cpp(
      #include "foo.h"
      first_token;
      void test() {
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
    ^MACRO_ARGS(^MACRO_ARGS(^MACRO_EXP(int), A), ^ID(= 2));

    // Macro names inside other macros not included.
    #define ^MACRO_ARGS2(X, Y) X Y
    #define ^FOO BAR
    #define ^BAR 1
    int A = ^FOO;

    // Macros from token concatenations not included.
    #define ^CONCAT(X) X##A()
    #define ^PREPEND(X) MACRO##X()
    #define ^MACROA() 123
    int B = ^CONCAT(MACRO);
    int D = ^PREPEND(A)

    // Macros included not from preamble not included.
    #include "foo.inc"

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

} // namespace
} // namespace clangd
} // namespace clang
