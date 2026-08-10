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

class IsCLibraryFunctionTest : public testing::Test {
  std::unique_ptr<ASTUnit> ASTUnitP;
  const FunctionDecl *Result = nullptr;

public:
  const FunctionDecl *getFunctionDecl() const { return Result; }

  testing::AssertionResult buildAST(StringRef Code) {
    ASTUnitP = tooling::buildASTFromCode(Code);
    if (!ASTUnitP)
      return testing::AssertionFailure() << "AST construction failed";

    ASTContext &Context = ASTUnitP->getASTContext();
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
};

TEST_F(IsCLibraryFunctionTest, AcceptsGlobal) {
  ASSERT_TRUE(buildAST(R"cpp(void fun();)cpp"));
  EXPECT_TRUE(CheckerContext::isCLibraryFunction(getFunctionDecl()));
}

TEST_F(IsCLibraryFunctionTest, AcceptsExternCGlobal) {
  ASSERT_TRUE(buildAST(R"cpp(extern "C" { void fun(); })cpp"));
  EXPECT_TRUE(CheckerContext::isCLibraryFunction(getFunctionDecl()));
}

TEST_F(IsCLibraryFunctionTest, RejectsNoInlineNoExternalLinkage) {
  // Functions that are neither inlined nor externally visible cannot be C
  // library functions.
  ASSERT_TRUE(buildAST(R"cpp(static void fun();)cpp"));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(getFunctionDecl()));
}

TEST_F(IsCLibraryFunctionTest, RejectsAnonymousNamespace) {
  ASSERT_TRUE(buildAST(R"cpp(namespace { void fun(); })cpp"));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(getFunctionDecl()));
}

TEST_F(IsCLibraryFunctionTest, AcceptsStdNamespace) {
  ASSERT_TRUE(buildAST(R"cpp(namespace std { void fun(); })cpp"));
  EXPECT_TRUE(CheckerContext::isCLibraryFunction(getFunctionDecl()));
}

TEST_F(IsCLibraryFunctionTest, RejectsOtherNamespaces) {
  ASSERT_TRUE(buildAST(R"cpp(namespace stdx { void fun(); })cpp"));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(getFunctionDecl()));
}

TEST_F(IsCLibraryFunctionTest, RejectsClassStatic) {
  ASSERT_TRUE(buildAST(R"cpp(class A { static void fun(); };)cpp"));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(getFunctionDecl()));
}

TEST_F(IsCLibraryFunctionTest, RejectsClassMember) {
  ASSERT_TRUE(buildAST(R"cpp(class A { void fun(); };)cpp"));
  EXPECT_FALSE(CheckerContext::isCLibraryFunction(getFunctionDecl()));
}
