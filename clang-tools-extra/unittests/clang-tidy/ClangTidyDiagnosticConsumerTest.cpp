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

TEST(ClangTidyDiagnosticConsumer, SortsErrors) {
  std::vector<ClangTidyError> Errors;
  runCheckOnCode<TestCheck>("int a;", &Errors);
  EXPECT_EQ(2ul, Errors.size());
  EXPECT_EQ("type specifier", Errors[0].Message.Message);
  EXPECT_EQ("variable", Errors[1].Message.Message);
}

TEST(GlobList, Empty) {
  GlobList Filter("");

  EXPECT_TRUE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("aaa"));
}

TEST(GlobList, Nothing) {
  GlobList Filter("-*");

  EXPECT_FALSE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("a"));
  EXPECT_FALSE(Filter.contains("-*"));
  EXPECT_FALSE(Filter.contains("-"));
  EXPECT_FALSE(Filter.contains("*"));
}

TEST(GlobList, Everything) {
  GlobList Filter("*");

  EXPECT_TRUE(Filter.contains(""));
  EXPECT_TRUE(Filter.contains("aaaa"));
  EXPECT_TRUE(Filter.contains("-*"));
  EXPECT_TRUE(Filter.contains("-"));
  EXPECT_TRUE(Filter.contains("*"));
}

TEST(GlobList, Simple) {
  GlobList Filter("aaa");

  EXPECT_TRUE(Filter.contains("aaa"));
  EXPECT_FALSE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("aa"));
  EXPECT_FALSE(Filter.contains("aaaa"));
  EXPECT_FALSE(Filter.contains("bbb"));
}

TEST(GlobList, WhitespacesAtBegin) {
  GlobList Filter("-*,   a.b.*");

  EXPECT_TRUE(Filter.contains("a.b.c"));
  EXPECT_FALSE(Filter.contains("b.c"));
}

TEST(GlobList, Complex) {
  GlobList Filter("*,-a.*, -b.*,   a.1.* ,-a.1.A.*,-..,-...,-..+,-*$, -*qwe* ");

  EXPECT_TRUE(Filter.contains("aaa"));
  EXPECT_TRUE(Filter.contains("qqq"));
  EXPECT_FALSE(Filter.contains("a."));
  EXPECT_FALSE(Filter.contains("a.b"));
  EXPECT_FALSE(Filter.contains("b."));
  EXPECT_FALSE(Filter.contains("b.b"));
  EXPECT_TRUE(Filter.contains("a.1.b"));
  EXPECT_FALSE(Filter.contains("a.1.A.a"));
  EXPECT_FALSE(Filter.contains("qwe"));
  EXPECT_FALSE(Filter.contains("asdfqweasdf"));
  EXPECT_TRUE(Filter.contains("asdfqwEasdf"));
}

} // namespace test
} // namespace tidy
} // namespace clang
