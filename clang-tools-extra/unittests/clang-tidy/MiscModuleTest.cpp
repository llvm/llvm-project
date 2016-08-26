#include "ClangTidyTest.h"
#include "misc/ArgumentCommentCheck.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

using misc::ArgumentCommentCheck;

TEST(ArgumentCommentCheckTest, CorrectComments) {
  EXPECT_NO_CHANGES(ArgumentCommentCheck,
                    "void f(int x, int y); void g() { f(/*x=*/0, /*y=*/0); }");
  EXPECT_NO_CHANGES(ArgumentCommentCheck,
                    "struct C { C(int x, int y); }; C c(/*x=*/0, /*y=*/0);");
}

TEST(ArgumentCommentCheckTest, ThisEditDistanceAboveThreshold) {
  EXPECT_NO_CHANGES(ArgumentCommentCheck,
                    "void f(int xxx); void g() { f(/*xyz=*/0); }");
}

TEST(ArgumentCommentCheckTest, OtherEditDistanceAboveThreshold) {
  EXPECT_EQ("void f(int xxx, int yyy); void g() { f(/*xxx=*/0, 0); }",
            runCheckOnCode<ArgumentCommentCheck>(
                "void f(int xxx, int yyy); void g() { f(/*Zxx=*/0, 0); }"));
  EXPECT_EQ("struct C { C(int xxx, int yyy); }; C c(/*xxx=*/0, 0);",
            runCheckOnCode<ArgumentCommentCheck>(
                "struct C { C(int xxx, int yyy); }; C c(/*Zxx=*/0, 0);"));
}

TEST(ArgumentCommentCheckTest, OtherEditDistanceBelowThreshold) {
  EXPECT_NO_CHANGES(ArgumentCommentCheck,
                    "void f(int xxx, int yyy); void g() { f(/*xxy=*/0, 0); }");
}

} // namespace test
} // namespace tidy
} // namespace clang
