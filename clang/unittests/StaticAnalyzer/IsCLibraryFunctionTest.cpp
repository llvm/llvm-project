#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

#include <memory>

using namespace clang;
using namespace ento;
using namespace ast_matchers;

testing::AssertionResult extractFunctionDecl(StringRef Code,
                                             const FunctionDecl *&Result) {
  auto ASTUnit = tooling::buildASTFromCode(Code);
  if (!ASTUnit)
    return testing::AssertionFailure() << "AST construction failed";

  ASTContext &Context = ASTUnit->getASTContext();
  if (Context.getDiagnostics().hasErrorOccurred())
    return testing::AssertionFailure() << "Compilation error";

  auto Matches = ast_matchers::match(functionDecl().bind("fn"), Context);
  if (Matches.empty())
    return testing::AssertionFailure() << "No function declaration found";

  if (Matches.size() > 1)
    return testing::AssertionFailure()
           << "Multiple function declarations found";

  Result = Matches[0].getNodeAs<FunctionDecl>("fn");
  return testing::AssertionSuccess();
}

TEST(IsCLibraryFunctionTest, AcceptsGlobal) {
  const FunctionDecl *Result;
  ASSERT_TRUE(extractFunctionDecl(R"cpp(void fun();)cpp", Result));
  EXPECT_TRUE(CheckerContext::isCLibraryFunction(Result));
}

TEST(IsCLibraryFunctionTest, AcceptsExternCGlobal) {
  const FunctionDecl *Result;
  ASSERT_TRUE(
      extractFunctionDecl(R"cpp(extern "C" { void fun(); })cpp", Result));
  EXPECT_TRUE(CheckerContext::isCLibraryFunction(Result));
}

TEST(IsCLibraryFunctionTest, RejectsNoInlineNoExternalLinkage) {
  // Functions that are neither inlined nor externally visible cannot be C library functions.
  const FunctionDecl *Result;
  ASSERT_TRUE(extractFunctionDecl(R"cpp(static void fun();)cpp", Result));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(Result));
}

TEST(IsCLibraryFunctionTest, RejectsAnonymousNamespace) {
  const FunctionDecl *Result;
  ASSERT_TRUE(
      extractFunctionDecl(R"cpp(namespace { void fun(); })cpp", Result));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(Result));
}

TEST(IsCLibraryFunctionTest, AcceptsStdNamespace) {
  const FunctionDecl *Result;
  ASSERT_TRUE(
      extractFunctionDecl(R"cpp(namespace std { void fun(); })cpp", Result));
  EXPECT_TRUE(CheckerContext::isCLibraryFunction(Result));
}

TEST(IsCLibraryFunctionTest, RejectsOtherNamespaces) {
  const FunctionDecl *Result;
  ASSERT_TRUE(
      extractFunctionDecl(R"cpp(namespace stdx { void fun(); })cpp", Result));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(Result));
}

TEST(IsCLibraryFunctionTest, RejectsClassStatic) {
  const FunctionDecl *Result;
  ASSERT_TRUE(
      extractFunctionDecl(R"cpp(class A { static void fun(); };)cpp", Result));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(Result));
}

TEST(IsCLibraryFunctionTest, RejectsClassMember) {
  const FunctionDecl *Result;
  ASSERT_TRUE(extractFunctionDecl(R"cpp(class A { void fun(); };)cpp", Result));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(Result));
}
