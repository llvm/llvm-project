#include "GlobList.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {

template <typename GlobListT> struct GlobListTest : public ::testing::Test {};

using GlobListTypes = ::testing::Types<GlobList, CachedGlobList>;
TYPED_TEST_SUITE(GlobListTest, GlobListTypes);

TYPED_TEST(GlobListTest, Empty) {
  TypeParam Filter("");

  EXPECT_TRUE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("aaa"));
}

TYPED_TEST(GlobListTest, Nothing) {
  TypeParam Filter("-*");

  EXPECT_FALSE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("a"));
  EXPECT_FALSE(Filter.contains("-*"));
  EXPECT_FALSE(Filter.contains("-"));
  EXPECT_FALSE(Filter.contains("*"));
}

TYPED_TEST(GlobListTest, Everything) {
  TypeParam Filter("*");

  EXPECT_TRUE(Filter.contains(""));
  EXPECT_TRUE(Filter.contains("aaaa"));
  EXPECT_TRUE(Filter.contains("-*"));
  EXPECT_TRUE(Filter.contains("-"));
  EXPECT_TRUE(Filter.contains("*"));
}

TYPED_TEST(GlobListTest, OneSimplePattern) {
  TypeParam Filter("aaa");

  EXPECT_TRUE(Filter.contains("aaa"));
  EXPECT_FALSE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("aa"));
  EXPECT_FALSE(Filter.contains("aaaa"));
  EXPECT_FALSE(Filter.contains("bbb"));
}

TYPED_TEST(GlobListTest, TwoSimplePatterns) {
  TypeParam Filter("aaa,bbb");

  EXPECT_TRUE(Filter.contains("aaa"));
  EXPECT_TRUE(Filter.contains("bbb"));
  EXPECT_FALSE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("aa"));
  EXPECT_FALSE(Filter.contains("aaaa"));
  EXPECT_FALSE(Filter.contains("bbbb"));
}

TYPED_TEST(GlobListTest, PatternPriority) {
  // The last glob that matches the string decides whether that string is
  // included or excluded.
  {
    TypeParam Filter("a*,-aaa");

    EXPECT_FALSE(Filter.contains(""));
    EXPECT_TRUE(Filter.contains("a"));
    EXPECT_TRUE(Filter.contains("aa"));
    EXPECT_FALSE(Filter.contains("aaa"));
    EXPECT_TRUE(Filter.contains("aaaa"));
  }
  {
    TypeParam Filter("-aaa,a*");

    EXPECT_FALSE(Filter.contains(""));
    EXPECT_TRUE(Filter.contains("a"));
    EXPECT_TRUE(Filter.contains("aa"));
    EXPECT_TRUE(Filter.contains("aaa"));
    EXPECT_TRUE(Filter.contains("aaaa"));
  }
}

TYPED_TEST(GlobListTest, WhitespacesAtBegin) {
  TypeParam Filter("-*,   a.b.*");

  EXPECT_TRUE(Filter.contains("a.b.c"));
  EXPECT_FALSE(Filter.contains("b.c"));
}

TYPED_TEST(GlobListTest, Complex) {
  TypeParam Filter(
      "*,-a.*, -b.*, \r  \n  a.1.* ,-a.1.A.*,-..,-...,-..+,-*$, -*qwe* ");

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

} // namespace tidy
} // namespace clang
