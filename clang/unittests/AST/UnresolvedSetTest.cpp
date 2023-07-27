#include "clang/AST/UnresolvedSet.h"
#include "clang/AST/Decl.h"
#include "gtest/gtest.h"

using namespace clang;

class UnresolvedSetTest : public ::testing::Test {
protected:
  // For this test we don't care about the contents of the NamedDecl,
  // so just use a dummy struct of appropriate size and alignment.
  struct alignas(NamedDecl) DummyDecl {
    char contents[sizeof(NamedDecl)];
  };
  DummyDecl d0, d1, d2, d3;
  NamedDecl *n0, *n1, *n2, *n3;
  UnresolvedSet<2> set;

  void SetUp() override {
    n0 = reinterpret_cast<NamedDecl*>(&d0);
    n1 = reinterpret_cast<NamedDecl*>(&d1);
    n2 = reinterpret_cast<NamedDecl*>(&d2);
    n3 = reinterpret_cast<NamedDecl*>(&d3);
    set.addDecl(n0);
    set.addDecl(n1);
    set.addDecl(n2);
    set.addDecl(n3);
  }
};

TEST_F(UnresolvedSetTest, Size) { EXPECT_EQ(set.size(), 4u); }

TEST_F(UnresolvedSetTest, ArrayOperator) {
  EXPECT_EQ(set[0].getDecl(), n0);
  EXPECT_EQ(set[1].getDecl(), n1);
  EXPECT_EQ(set[2].getDecl(), n2);
  EXPECT_EQ(set[3].getDecl(), n3);
}

TEST_F(UnresolvedSetTest, EraseIntegerFromStart) {
  set.erase(0);
  EXPECT_EQ(set.size(), 3u);
  EXPECT_EQ(set[0].getDecl(), n3);
  EXPECT_EQ(set[1].getDecl(), n1);
  EXPECT_EQ(set[2].getDecl(), n2);

  set.erase(0);
  EXPECT_EQ(set.size(), 2u);
  EXPECT_EQ(set[0].getDecl(), n2);
  EXPECT_EQ(set[1].getDecl(), n1);

  set.erase(0);
  EXPECT_EQ(set.size(), 1u);
  EXPECT_EQ(set[0].getDecl(), n1);

  set.erase(0);
  EXPECT_EQ(set.size(), 0u);
}

TEST_F(UnresolvedSetTest, EraseIntegerFromEnd) {
  set.erase(3);
  EXPECT_EQ(set.size(), 3u);
  EXPECT_EQ(set[0].getDecl(), n0);
  EXPECT_EQ(set[1].getDecl(), n1);
  EXPECT_EQ(set[2].getDecl(), n2);

  set.erase(2);
  EXPECT_EQ(set.size(), 2u);
  EXPECT_EQ(set[0].getDecl(), n0);
  EXPECT_EQ(set[1].getDecl(), n1);

  set.erase(1);
  EXPECT_EQ(set.size(), 1u);
  EXPECT_EQ(set[0].getDecl(), n0);

  set.erase(0);
  EXPECT_EQ(set.size(), 0u);
}

TEST_F(UnresolvedSetTest, EraseIteratorFromStart) {
  set.erase(set.begin());
  EXPECT_EQ(set.size(), 3u);
  EXPECT_EQ(set[0].getDecl(), n3);
  EXPECT_EQ(set[1].getDecl(), n1);
  EXPECT_EQ(set[2].getDecl(), n2);

  set.erase(set.begin());
  EXPECT_EQ(set.size(), 2u);
  EXPECT_EQ(set[0].getDecl(), n2);
  EXPECT_EQ(set[1].getDecl(), n1);

  set.erase(set.begin());
  EXPECT_EQ(set.size(), 1u);
  EXPECT_EQ(set[0].getDecl(), n1);

  set.erase(set.begin());
  EXPECT_EQ(set.size(), 0u);
}

TEST_F(UnresolvedSetTest, EraseIteratorFromEnd) {
  set.erase(--set.end());
  EXPECT_EQ(set.size(), 3u);
  EXPECT_EQ(set[0].getDecl(), n0);
  EXPECT_EQ(set[1].getDecl(), n1);
  EXPECT_EQ(set[2].getDecl(), n2);

  set.erase(--set.end());
  EXPECT_EQ(set.size(), 2u);
  EXPECT_EQ(set[0].getDecl(), n0);
  EXPECT_EQ(set[1].getDecl(), n1);

  set.erase(--set.end());
  EXPECT_EQ(set.size(), 1u);
  EXPECT_EQ(set[0].getDecl(), n0);

  set.erase(--set.end());
  EXPECT_EQ(set.size(), 0u);
}
