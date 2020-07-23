#include "ClangTidy.h"
#include "ClangTidyTest.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

class TestCheck : public ClangTidyCheck {
public:
  TestCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    Finder->addMatcher(ast_matchers::varDecl().bind("var"), this);
  }
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var");
    // Add diagnostics in the wrong order.
    diag(Var->getLocation(), "variable");
    diag(Var->getTypeSpecStartLoc(), "type specifier");
  }
};

// FIXME: This test seems to cause a strange linking interference
// with the ValidConfiguration.ValidEnumOptions test on macOS.
// If both tests are enabled, this test will fail as if
// runCheckOnCode() is not invoked at all. Looks like a linker bug.
// For now both tests are disabled on macOS. It is not sufficient
// to only disable the other test because this test keeps failing
// under Address Sanitizer, which may be an indication of more
// such linking interference with other tests and this test
// seems to be in the center of it.
#ifndef __APPLE__
TEST(ClangTidyDiagnosticConsumer, SortsErrors) {
  std::vector<ClangTidyError> Errors;
  runCheckOnCode<TestCheck>("int a;", &Errors);
  EXPECT_EQ(2ul, Errors.size());
  EXPECT_EQ("type specifier", Errors[0].Message.Message);
  EXPECT_EQ("variable", Errors[1].Message.Message);
}
#endif

} // namespace test
} // namespace tidy
} // namespace clang
