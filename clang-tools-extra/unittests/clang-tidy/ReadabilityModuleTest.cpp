#include "../../clang/unittests/ASTMatchers/ASTMatchersTest.h"
#include "ClangTidyTest.h"
#include "readability/BracesAroundStatementsCheck.h"
#include "readability/NamespaceCommentCheck.h"
#include "readability/SimplifyBooleanExprMatchers.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

using readability::BracesAroundStatementsCheck;
using readability::NamespaceCommentCheck;
using namespace ast_matchers;

TEST_P(ASTMatchersTest, HasCaseSubstatement) {
  EXPECT_TRUE(matches(
      "void f() { switch (1) { case 1: return; break; default: break; } }",
      traverse(TK_AsIs, caseStmt(hasSubstatement(returnStmt())))));
}

TEST_P(ASTMatchersTest, HasDefaultSubstatement) {
  EXPECT_TRUE(matches(
      "void f() { switch (1) { case 1: return; break; default: break; } }",
      traverse(TK_AsIs, defaultStmt(hasSubstatement(breakStmt())))));
}

TEST_P(ASTMatchersTest, HasLabelSubstatement) {
  EXPECT_TRUE(
      matches("void f() { while (1) { bar: break; foo: return; } }",
              traverse(TK_AsIs, labelStmt(hasSubstatement(breakStmt())))));
}

TEST_P(ASTMatchersTest, HasSubstatementSequenceSimple) {
  const char *Text = "int f() { int x = 5; if (x < 0) return 1; return 0; }";
  EXPECT_TRUE(matches(
      Text, compoundStmt(hasSubstatementSequence(ifStmt(), returnStmt()))));
  EXPECT_FALSE(matches(
      Text, compoundStmt(hasSubstatementSequence(ifStmt(), labelStmt()))));
  EXPECT_FALSE(matches(
      Text, compoundStmt(hasSubstatementSequence(returnStmt(), ifStmt()))));
  EXPECT_FALSE(matches(
      Text, compoundStmt(hasSubstatementSequence(switchStmt(), labelStmt()))));
}

TEST_P(ASTMatchersTest, HasSubstatementSequenceAlmost) {
  const char *Text = R"code(
int f() {
  int x = 5;
  if (x < 10)
    ;
  if (x < 0)
    return 1;
  return 0;
}
)code";
  EXPECT_TRUE(matches(
      Text, compoundStmt(hasSubstatementSequence(ifStmt(), returnStmt()))));
  EXPECT_TRUE(
      matches(Text, compoundStmt(hasSubstatementSequence(ifStmt(), ifStmt()))));
}

TEST_P(ASTMatchersTest, HasSubstatementSequenceComplex) {
  const char *Text = R"code(
int f() {
  int x = 5;
  if (x < 10)
    x -= 10;
  if (x < 0)
    return 1;
  return 0;
}
)code";
  EXPECT_TRUE(matches(
      Text, compoundStmt(hasSubstatementSequence(ifStmt(), returnStmt()))));
  EXPECT_FALSE(
      matches(Text, compoundStmt(hasSubstatementSequence(ifStmt(), expr()))));
}

TEST_P(ASTMatchersTest, HasSubstatementSequenceExpression) {
  const char *Text = R"code(
int f() {
  return ({ int x = 5;
      int result;
      if (x < 10)
        x -= 10;
      if (x < 0)
        result = 1;
      else
        result = 0;
      result;
    });
  }
)code";
  EXPECT_TRUE(
      matches(Text, stmtExpr(hasSubstatementSequence(ifStmt(), expr()))));
  EXPECT_FALSE(
      matches(Text, stmtExpr(hasSubstatementSequence(ifStmt(), returnStmt()))));
}

// Copied from ASTMatchersTests
static std::vector<TestClangConfig> allTestClangConfigs() {
  std::vector<TestClangConfig> all_configs;
  for (TestLanguage lang : {Lang_C89, Lang_C99, Lang_CXX03, Lang_CXX11,
                            Lang_CXX14, Lang_CXX17, Lang_CXX20}) {
    TestClangConfig config;
    config.Language = lang;

    // Use an unknown-unknown triple so we don't instantiate the full system
    // toolchain.  On Linux, instantiating the toolchain involves stat'ing
    // large portions of /usr/lib, and this slows down not only this test, but
    // all other tests, via contention in the kernel.
    //
    // FIXME: This is a hack to work around the fact that there's no way to do
    // the equivalent of runToolOnCodeWithArgs without instantiating a full
    // Driver.  We should consider having a function, at least for tests, that
    // invokes cc1.
    config.Target = "i386-unknown-unknown";
    all_configs.push_back(config);

    // Windows target is interesting to test because it enables
    // `-fdelayed-template-parsing`.
    config.Target = "x86_64-pc-win32-msvc";
    all_configs.push_back(config);
  }
  return all_configs;
}

INSTANTIATE_TEST_SUITE_P(ASTMatchersTests, ASTMatchersTest,
                         testing::ValuesIn(allTestClangConfigs()));

TEST(NamespaceCommentCheckTest, Basic) {
  EXPECT_EQ("namespace i {\n} // namespace i",
            runCheckOnCode<NamespaceCommentCheck>("namespace i {\n}"));
  EXPECT_EQ("namespace {\n} // namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n}"));
  EXPECT_EQ("namespace i { namespace j {\n} // namespace j\n } // namespace i",
            runCheckOnCode<NamespaceCommentCheck>(
                "namespace i { namespace j {\n} }"));
}

TEST(NamespaceCommentCheckTest, SingleLineNamespaces) {
  EXPECT_EQ(
      "namespace i { namespace j { } }",
      runCheckOnCode<NamespaceCommentCheck>("namespace i { namespace j { } }"));
}

TEST(NamespaceCommentCheckTest, CheckExistingComments) {
  EXPECT_EQ("namespace i { namespace j {\n"
            "} /* namespace j */ } // namespace i\n"
            " /* random comment */",
            runCheckOnCode<NamespaceCommentCheck>(
                "namespace i { namespace j {\n"
                "} /* namespace j */ } /* random comment */"));
  EXPECT_EQ("namespace {\n"
            "} // namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // namespace"));
  EXPECT_EQ("namespace {\n"
            "} //namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} //namespace"));
  EXPECT_EQ("namespace {\n"
            "} // anonymous namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // anonymous namespace"));
  EXPECT_EQ("namespace {\n"
            "} // Anonymous namespace.",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // Anonymous namespace."));
  EXPECT_EQ(
      "namespace q {\n"
      "} // namespace q",
      runCheckOnCode<NamespaceCommentCheck>("namespace q {\n"
                                            "} // anonymous namespace q"));
  EXPECT_EQ(
      "namespace My_NameSpace123 {\n"
      "} // namespace My_NameSpace123",
      runCheckOnCode<NamespaceCommentCheck>("namespace My_NameSpace123 {\n"
                                            "} // namespace My_NameSpace123"));
  EXPECT_EQ(
      "namespace My_NameSpace123 {\n"
      "} //namespace My_NameSpace123",
      runCheckOnCode<NamespaceCommentCheck>("namespace My_NameSpace123 {\n"
                                            "} //namespace My_NameSpace123"));
  EXPECT_EQ("namespace My_NameSpace123 {\n"
            "} //  end namespace   My_NameSpace123",
            runCheckOnCode<NamespaceCommentCheck>(
                "namespace My_NameSpace123 {\n"
                "} //  end namespace   My_NameSpace123"));
  // Understand comments only on the same line.
  EXPECT_EQ("namespace {\n"
            "} // namespace\n"
            "// namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "}\n"
                                                  "// namespace"));
}

TEST(NamespaceCommentCheckTest, FixWrongComments) {
  EXPECT_EQ("namespace i { namespace jJ0_ {\n"
            "} // namespace jJ0_\n"
            " } // namespace i\n"
            " /* random comment */",
            runCheckOnCode<NamespaceCommentCheck>(
                "namespace i { namespace jJ0_ {\n"
                "} /* namespace qqq */ } /* random comment */"));
  EXPECT_EQ("namespace {\n"
            "} // namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // namespace asdf"));
  // Remove unknown line comments. These are likely to be an unrecognized form
  // of a namespace ending comment.
  EXPECT_EQ("namespace {\n"
            "} // namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // random text"));
}

TEST(BracesAroundStatementsCheckTest, IfWithComments) {
  EXPECT_EQ("int main() {\n"
            "  if (false /*dummy token*/) {\n"
            "    // comment\n"
            "    return -1; /**/\n"
            "}\n"
            "  if (false) {\n"
            "    return -1; // comment\n"
            "}\n"
            "  if (false) {\n"
            "    return -1; \n"
            "}/* multi-line \n comment */\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (false /*dummy token*/)\n"
                "    // comment\n"
                "    return -1; /**/\n"
                "  if (false)\n"
                "    return -1; // comment\n"
                "  if (false)\n"
                "    return -1; /* multi-line \n comment */\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  if (false /*dummy token*/) {\n"
            "    // comment\n"
            "    return -1 /**/ ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (false /*dummy token*/)\n"
                "    // comment\n"
                "    return -1 /**/ ;\n"
                "}"));
}

TEST(BracesAroundStatementsCheckTest, If) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  if (false) {\n"
                                                 "    return -1;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  if (auto Cond = false) {\n"
                                                 "    return -1;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  if (false) {\n"
                                                 "    return -1;\n"
                                                 "  } else {\n"
                                                 "    return -2;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  if (false) {\n"
            "    return -1;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  if (false)\n"
                                                        "    return -1;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  if (auto Cond = false /**/ ) {\n"
            "    return -1;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (auto Cond = false /**/ )\n"
                "    return -1;\n"
                "}"));
  // FIXME: Consider adding braces before EMPTY_MACRO and after the statement.
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck,
                    "#define EMPTY_MACRO\n"
                    "int main() {\n"
                    "  if (auto Cond = false EMPTY_MACRO /**/ ) EMPTY_MACRO\n"
                    "    return -1;\n"
                    "}");
  EXPECT_EQ("int main() {\n"
            "  if (true) { return -1/**/ ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (true) return -1/**/ ;\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  if (false) {\n"
            "    return -1;\n"
            "  } else {\n"
            "    return -2;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  if (false)\n"
                                                        "    return -1;\n"
                                                        "  else\n"
                                                        "    return -2;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  if (false) {\n"
            "    return -1;\n"
            "  } else if (1 == 2) {\n"
            "    return -2;\n"
            "  } else {\n"
            "    return -3;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  if (false)\n"
                                                        "    return -1;\n"
                                                        "  else if (1 == 2)\n"
                                                        "    return -2;\n"
                                                        "  else\n"
                                                        "    return -3;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  if (false) {\n"
            "    return -1;\n"
            "  } else if (1 == 2) {\n"
            "    return -2;\n"
            "  } else {\n"
            "    return -3;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  if (false)\n"
                                                        "    return -1;\n"
                                                        "  else if (1 == 2) {\n"
                                                        "    return -2;\n"
                                                        "  } else\n"
                                                        "    return -3;\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheckTest, IfElseWithShortStatements) {
  ClangTidyOptions Options;
  Options.CheckOptions["test-check-0.ShortStatementLines"] = "1";

  EXPECT_EQ("int main() {\n"
            "  if (true) return 1;\n"
            "  if (false) { return -1;\n"
            "  } else if (1 == 2) { return -2;\n"
            "  } else { return -3;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (true) return 1;\n"
                "  if (false) return -1;\n"
                "  else if (1 == 2) return -2;\n"
                "  else return -3;\n"
                "}",
                nullptr, "input.cc", None, Options));

  // If the last else is an else-if, we also force it.
  EXPECT_EQ("int main() {\n"
            "  if (false) { return -1;\n"
            "  } else if (1 == 2) { return -2;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (false) return -1;\n"
                "  else if (1 == 2) return -2;\n"
                "}",
                nullptr, "input.cc", None, Options));
}

TEST(BracesAroundStatementsCheckTest, For) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  for (;;) {\n"
                                                 "    ;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  for (;;) {\n"
            "    ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  for (;;)\n"
                                                        "    ;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  for (;;) {\n"
            "    /**/ ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  for (;;)\n"
                                                        "    /**/ ;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  for (;;) {\n"
            "    return -1 /**/ ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  for (;;)\n"
                                                        "    return -1 /**/ ;\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheckTest, ForRange) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  int arr[4];\n"
                                                 "  for (int i : arr) {\n"
                                                 "    ;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  int arr[4];\n"
            "  for (int i : arr) {\n"
            "    ;\n"
            "}\n"
            "  for (int i : arr) {\n"
            "    return -1 ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  int arr[4];\n"
                                                        "  for (int i : arr)\n"
                                                        "    ;\n"
                                                        "  for (int i : arr)\n"
                                                        "    return -1 ;\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheckTest, DoWhile) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  do {\n"
                                                 "    ;\n"
                                                 "  } while (false);\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  do {\n"
            "    ;\n"
            "  } while (false);\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  do\n"
                                                        "    ;\n"
                                                        "  while (false);\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheckTest, While) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  while (false) {\n"
                                                 "    ;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  while (false) {\n"
            "    ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  while (false)\n"
                                                        "    ;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  while (auto Cond = false) {\n"
            "    ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  while (auto Cond = false)\n"
                "    ;\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  while (false /*dummy token*/) {\n"
            "    ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  while (false /*dummy token*/)\n"
                "    ;\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  while (false) {\n"
            "    break;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  while (false)\n"
                                                        "    break;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  while (false) {\n"
            "    break /**/;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  while (false)\n"
                                                        "    break /**/;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  while (false) {\n"
            "    /**/;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  while (false)\n"
                                                        "    /**/;\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheckTest, Nested) {
  EXPECT_EQ("int main() {\n"
            "  do { if (true) {}} while (false);\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  do if (true) {}while (false);\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  do { if (true) {}} while (false);\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  do if (true) {}while (false);\n"
                "}"));
  EXPECT_EQ(
      "int main() {\n"
      "  if (true) {\n"
      "    // comment\n"
      "    if (false) {\n"
      "      // comment\n"
      "      /**/ ; // comment\n"
      "    }\n"
      "}\n"
      "}",
      runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                  "  if (true)\n"
                                                  "    // comment\n"
                                                  "    if (false) {\n"
                                                  "      // comment\n"
                                                  "      /**/ ; // comment\n"
                                                  "    }\n"
                                                  "}"));
}

TEST(BracesAroundStatementsCheckTest, Macros) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck,
                    "#define IF(COND) if (COND) return -1;\n"
                    "int main() {\n"
                    "  IF(false)\n"
                    "}");
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck,
                    "#define FOR(COND) for (COND) return -1;\n"
                    "int main() {\n"
                    "  FOR(;;)\n"
                    "}");
  EXPECT_EQ("#define DO_IT ++i\n"
            "int i = 0;\n"
            "int main() {\n"
            "  if (false) {\n"
            "    DO_IT;\n"
            "  } else if (1 == 2) {\n"
            "    DO_IT;\n"
            "  } else {\n"
            "    DO_IT;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("#define DO_IT ++i\n"
                                                        "int i = 0;\n"
                                                        "int main() {\n"
                                                        "  if (false)\n"
                                                        "    DO_IT;\n"
                                                        "  else if (1 == 2)\n"
                                                        "    DO_IT;\n"
                                                        "  else\n"
                                                        "    DO_IT;\n"
                                                        "}"));
}

#define EXPECT_NO_CHANGES_WITH_OPTS(Check, Opts, Code)                         \
  EXPECT_EQ(Code, runCheckOnCode<Check>(Code, nullptr, "input.cc", None, Opts))
TEST(BracesAroundStatementsCheckTest, ImplicitCastInReturn) {
  ClangTidyOptions Opts;
  Opts.CheckOptions["test-check-0.ShortStatementLines"] = "1";

  StringRef Input = "const char *f() {\n"
                    "  if (true) return \"\";\n"
                    "  return \"abc\";\n"
                    "}\n";
  EXPECT_NO_CHANGES_WITH_OPTS(BracesAroundStatementsCheck, Opts, Input);
  EXPECT_EQ("const char *f() {\n"
            "  if (true) { return \"\";\n"
            "}\n"
            "  return \"abc\";\n"
            "}\n",
            runCheckOnCode<BracesAroundStatementsCheck>(Input));
}

} // namespace test
} // namespace tidy
} // namespace clang
