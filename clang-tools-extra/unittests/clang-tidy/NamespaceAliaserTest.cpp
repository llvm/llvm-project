//===---- NamespaceAliaserTest.cpp - clang-tidy
//----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/NamespaceAliaser.h"

#include "ClangTidyTest.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace utils {
// This checker is for testing only. It can only run on one test case
// (e.g. with one SourceManager).
class InsertAliasCheck : public ClangTidyCheck {
public:
  InsertAliasCheck(StringRef Name, ClangTidyContext *Context)
      :ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    Finder->addMatcher(ast_matchers::callExpr().bind("foo"), this);
  }
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    if (!Aliaser)
      Aliaser.reset(new NamespaceAliaser(*Result.SourceManager));

    const auto *Call = Result.Nodes.getNodeAs<CallExpr>("foo");
    assert(Call != nullptr && "Did not find node \"foo\"");
    auto Hint = Aliaser->createAlias(*Result.Context, *Call, "::foo::bar",
                                     {"b", "some_alias"});
    if (Hint.hasValue())
      diag(Call->getBeginLoc(), "Fix for testing") << Hint.getValue();

    diag(Call->getBeginLoc(), "insert call") << FixItHint::CreateInsertion(
        Call->getBeginLoc(),
        Aliaser->getNamespaceName(*Result.Context, *Call, "::foo::bar") + "::");
  }

private:
  std::unique_ptr<NamespaceAliaser> Aliaser;
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

TEST(NamespaceAliaserTest, AddNewAlias) {
  EXPECT_EQ("#include \"foo.h\"\n"
            "void f() {\n"
            "namespace b = ::foo::bar;"
            " b::f(); }",
            runChecker<InsertAliasCheck>("#include \"foo.h\"\n"
                                         "void f() { f(); }",
                                         2));
}

TEST(NamespaceAliaserTest, ReuseAlias) {
  EXPECT_EQ(
      "#include \"foo.h\"\n"
      "void f() { namespace x = foo::bar; x::f(); }",
      runChecker<InsertAliasCheck>("#include \"foo.h\"\n"
                                   "void f() { namespace x = foo::bar; f(); }",
                                   1));
}

TEST(NamespaceAliaserTest, AddsOnlyOneAlias) {
  EXPECT_EQ("#include \"foo.h\"\n"
            "void f() {\n"
            "namespace b = ::foo::bar;"
            " b::f(); b::f(); }",
            runChecker<InsertAliasCheck>("#include \"foo.h\"\n"
                                         "void f() { f(); f(); }",
                                         3));
}

TEST(NamespaceAliaserTest, LocalConflict) {
  EXPECT_EQ("#include \"foo.h\"\n"
            "void f() {\n"
            "namespace some_alias = ::foo::bar;"
            " namespace b = foo; some_alias::f(); }",
            runChecker<InsertAliasCheck>("#include \"foo.h\"\n"
                                         "void f() { namespace b = foo; f(); }",
                                         2));
}

TEST(NamespaceAliaserTest, GlobalConflict) {
  EXPECT_EQ("#include \"foo.h\"\n"
            "namespace b = foo;\n"
            "void f() {\n"
            "namespace some_alias = ::foo::bar;"
            " some_alias::f(); }",
            runChecker<InsertAliasCheck>("#include \"foo.h\"\n"
                                         "namespace b = foo;\n"
                                         "void f() { f(); }",
                                         2));
}

} // namespace utils
} // namespace tidy
} // namespace clang
