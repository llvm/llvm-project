//===-- ClangdUnitTests.cpp - ClangdUnit tests ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Annotations.h"
#include "ClangdUnit.h"
#include "Compiler.h"
#include "Diagnostics.h"
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

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::AllOf;

TEST(ClangdUnitTest, GetBeginningOfIdentifier) {
  std::string Preamble = R"cpp(
struct Bar { int func(); };
#define MACRO(X) void f() { X; }
Bar* bar;
  )cpp";
  // First ^ is the expected beginning, last is the search position.
  for (std::string Text : std::vector<std::string>{
           "int ^f^oo();", // inside identifier
           "int ^foo();",  // beginning of identifier
           "int ^foo^();", // end of identifier
           "int foo(^);",  // non-identifier
           "^int foo();",  // beginning of file (can't back up)
           "int ^f0^0();", // after a digit (lexing at N-1 is wrong)
           "int ^λλ^λ();", // UTF-8 handled properly when backing up

           // identifier in macro arg
           "MACRO(bar->^func())",  // beginning of identifier
           "MACRO(bar->^fun^c())", // inside identifier
           "MACRO(bar->^func^())", // end of identifier
           "MACRO(^bar->func())",  // begin identifier
           "MACRO(^bar^->func())", // end identifier
           "^MACRO(bar->func())",  // beginning of macro name
           "^MAC^RO(bar->func())", // inside macro name
           "^MACRO^(bar->func())", // end of macro name
       }) {
    std::string WithPreamble = Preamble + Text;
    Annotations TestCase(WithPreamble);
    auto AST = TestTU::withCode(TestCase.code()).build();
    const auto &SourceMgr = AST.getSourceManager();
    SourceLocation Actual = getBeginningOfIdentifier(
        AST, TestCase.points().back(), SourceMgr.getMainFileID());
    Position ActualPos = offsetToPosition(
        TestCase.code(),
        SourceMgr.getFileOffset(SourceMgr.getSpellingLoc(Actual)));
    EXPECT_EQ(TestCase.points().front(), ActualPos) << Text;
  }
}

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

TEST(ClangdUnitTest, TopLevelDecls) {
  TestTU TU;
  TU.HeaderCode = R"(
    int header1();
    int header2;
  )";
  TU.Code = "int main();";
  auto AST = TU.build();
  EXPECT_THAT(AST.getLocalTopLevelDecls(), ElementsAre(DeclNamed("main")));
}

TEST(ClangdUnitTest, DoesNotGetIncludedTopDecls) {
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

TEST(ClangdUnitTest, DoesNotGetImplicitTemplateTopDecls) {
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

TEST(ClangdUnitTest,
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

TEST(ClangdUnitTest, TokensAfterPreamble) {
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


TEST(ClangdUnitTest, NoCrashOnTokensWithTidyCheck) {
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

TEST(ClangdUnitTest, CanBuildInvocationWithUnknownArgs) {
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

} // namespace
} // namespace clangd
} // namespace clang
