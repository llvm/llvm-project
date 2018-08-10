//===---- UsingInserterTest.cpp - clang-tidy ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/UsingInserter.h"

#include "ClangTidyTest.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace utils {

// Replace all function calls with calls to foo::func. Inserts using
// declarations as necessary. This checker is for testing only. It
// can only run on one test case (e.g. wih one SourceManager).
class InsertUsingCheck : public clang::tidy::ClangTidyCheck {
public:
  InsertUsingCheck(StringRef Name, ClangTidyContext *Context)
      :ClangTidyCheck(Name, Context) {}
  void registerMatchers(clang::ast_matchers::MatchFinder *Finder) override {
    Finder->addMatcher(clang::ast_matchers::callExpr().bind("foo"), this);
  }
  void
  check(const clang::ast_matchers::MatchFinder::MatchResult &Result) override {
    if (!Inserter)
      Inserter.reset(new UsingInserter(*Result.SourceManager));

    const auto *Call = Result.Nodes.getNodeAs<clang::CallExpr>("foo");
    assert(Call != nullptr && "Did not find node \"foo\"");
    auto Hint =
        Inserter->createUsingDeclaration(*Result.Context, *Call, "::foo::func");

    if (Hint.hasValue())
      diag(Call->getBeginLoc(), "Fix for testing") << Hint.getValue();

    diag(Call->getBeginLoc(), "insert call")
        << clang::FixItHint::CreateReplacement(
               Call->getCallee()->getSourceRange(),
               Inserter->getShortName(*Result.Context, *Call, "::foo::func"));
  }

private:
  std::unique_ptr<UsingInserter> Inserter;
};

template <typename Check>
std::string runChecker(StringRef Code, unsigned ExpectedWarningCount) {
  std::map<StringRef, StringRef> AdditionalFileContents = {{"foo.h",
                                                            "namespace foo {\n"
                                                            "namespace bar {\n"
                                                            "}\n"
                                                            "void func() { }\n"
                                                            "}"}};
  std::vector<ClangTidyError> errors;

  std::string result =
      test::runCheckOnCode<Check>(Code, &errors, "foo.cc", None,
                                  ClangTidyOptions(), AdditionalFileContents);

  EXPECT_EQ(ExpectedWarningCount, errors.size());
  return result;
}

TEST(UsingInserterTest, ReusesExisting) {
  EXPECT_EQ("#include \"foo.h\"\n"
            "namespace {"
            "using ::foo::func;\n"
            "void f() { func(); }"
            "}",
            runChecker<InsertUsingCheck>("#include \"foo.h\"\n"
                                         "namespace {"
                                         "using ::foo::func;\n"
                                         "void f() { f(); }"
                                         "}",
                                         1));
}

TEST(UsingInserterTest, ReusesExistingGlobal) {
  EXPECT_EQ("#include \"foo.h\"\n"
            "using ::foo::func;\n"
            "namespace {"
            "void f() { func(); }"
            "}",
            runChecker<InsertUsingCheck>("#include \"foo.h\"\n"
                                         "using ::foo::func;\n"
                                         "namespace {"
                                         "void f() { f(); }"
                                         "}",
                                         1));
}

TEST(UsingInserterTest, AvoidsConflict) {
  EXPECT_EQ("#include \"foo.h\"\n"
            "namespace {"
            "void f() { int func; ::foo::func(); }"
            "}",
            runChecker<InsertUsingCheck>("#include \"foo.h\"\n"
                                         "namespace {"
                                         "void f() { int func; f(); }"
                                         "}",
                                         1));
}

} // namespace utils
} // namespace tidy
} // namespace clang
