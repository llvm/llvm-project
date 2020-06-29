#include "ClangTidy.h"
#include "ClangTidyTest.h"
#include "gtest/gtest.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace test {

class TestCheck : public ClangTidyCheck {
public:
  TestCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(MatchFinder *Finder) override {
    Finder->addMatcher(varDecl().bind("var"), this);
    DidRegister = true;
  }
  void check(const MatchFinder::MatchResult &Result) override {
    const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var");

    assert(Var && "Node isn't bound");
    DidFire = true;

    // Add diagnostics in the wrong order.
    diag(Var->getLocation(), "variable");
    diag(Var->getTypeSpecStartLoc(), "type specifier");
  }

  ~TestCheck() {
    assert(DidRegister && "Check never registered");
    assert(DidFire && "Check never fired");
  }

private:
  bool DidRegister = false;
  bool DidFire = false;
};

TEST(ClangTidyDiagnosticConsumer, SortsErrors) {
  std::vector<ClangTidyError> Errors;
  runCheckOnCode<TestCheck>("int a;", &Errors);
  EXPECT_EQ(2u, Errors.size());
  EXPECT_EQ("type specifier", Errors[0].Message.Message);
  EXPECT_EQ("variable", Errors[1].Message.Message);
}

} // namespace test
} // namespace tidy
} // namespace clang
