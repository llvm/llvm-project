//===-- list_test.cc --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo/standalone/list.h"
#include "gtest/gtest.h"

struct ListItem {
  ListItem *Next;
};

typedef scudo::IntrusiveList<ListItem> List;

static List StaticList;

static void setList(List *L, ListItem *X = nullptr, ListItem *Y = nullptr,
                    ListItem *Z = nullptr) {
  L->clear();
  if (X)
    L->push_back(X);
  if (Y)
    L->push_back(Y);
  if (Z)
    L->push_back(Z);
}

static void checkList(List *L, ListItem *I1, ListItem *I2 = nullptr,
                      ListItem *I3 = nullptr, ListItem *I4 = nullptr,
                      ListItem *I5 = nullptr, ListItem *I6 = nullptr) {
  if (I1) {
    EXPECT_EQ(L->front(), I1);
    L->pop_front();
  }
  if (I2) {
    EXPECT_EQ(L->front(), I2);
    L->pop_front();
  }
  if (I3) {
    EXPECT_EQ(L->front(), I3);
    L->pop_front();
  }
  if (I4) {
    EXPECT_EQ(L->front(), I4);
    L->pop_front();
  }
  if (I5) {
    EXPECT_EQ(L->front(), I5);
    L->pop_front();
  }
  if (I6) {
    EXPECT_EQ(L->front(), I6);
    L->pop_front();
  }
  EXPECT_TRUE(L->empty());
}

TEST(ScudoSandalone, IntrusiveList) {
  ListItem Items[6];
  EXPECT_EQ(StaticList.size(), 0U);

  List L;
  L.clear();

  ListItem *X = &Items[0];
  ListItem *Y = &Items[1];
  ListItem *Z = &Items[2];
  ListItem *A = &Items[3];
  ListItem *B = &Items[4];
  ListItem *C = &Items[5];

  EXPECT_EQ(L.size(), 0U);
  L.push_back(X);
  EXPECT_EQ(L.size(), 1U);
  EXPECT_EQ(L.back(), X);
  EXPECT_EQ(L.front(), X);
  L.pop_front();
  EXPECT_TRUE(L.empty());
  L.checkConsistency();

  L.push_front(X);
  EXPECT_EQ(L.size(), 1U);
  EXPECT_EQ(L.back(), X);
  EXPECT_EQ(L.front(), X);
  L.pop_front();
  EXPECT_TRUE(L.empty());
  L.checkConsistency();

  L.push_front(X);
  L.push_front(Y);
  L.push_front(Z);
  EXPECT_EQ(L.size(), 3U);
  EXPECT_EQ(L.front(), Z);
  EXPECT_EQ(L.back(), X);
  L.checkConsistency();

  L.pop_front();
  EXPECT_EQ(L.size(), 2U);
  EXPECT_EQ(L.front(), Y);
  EXPECT_EQ(L.back(), X);
  L.pop_front();
  L.pop_front();
  EXPECT_TRUE(L.empty());
  L.checkConsistency();

  L.push_back(X);
  L.push_back(Y);
  L.push_back(Z);
  EXPECT_EQ(L.size(), 3U);
  EXPECT_EQ(L.front(), X);
  EXPECT_EQ(L.back(), Z);
  L.checkConsistency();

  L.pop_front();
  EXPECT_EQ(L.size(), 2U);
  EXPECT_EQ(L.front(), Y);
  EXPECT_EQ(L.back(), Z);
  L.pop_front();
  L.pop_front();
  EXPECT_TRUE(L.empty());
  L.checkConsistency();

  L.push_back(X);
  L.push_back(Y);
  L.push_back(Z);
  L.extract(X, Y);
  EXPECT_EQ(L.size(), 2U);
  EXPECT_EQ(L.front(), X);
  EXPECT_EQ(L.back(), Z);
  L.checkConsistency();
  L.extract(X, Z);
  EXPECT_EQ(L.size(), 1U);
  EXPECT_EQ(L.front(), X);
  EXPECT_EQ(L.back(), X);
  L.checkConsistency();
  L.pop_front();
  EXPECT_TRUE(L.empty());

  List L1, L2;
  L1.clear();
  L2.clear();

  L1.append_front(&L2);
  EXPECT_TRUE(L1.empty());
  EXPECT_TRUE(L2.empty());

  L1.append_back(&L2);
  EXPECT_TRUE(L1.empty());
  EXPECT_TRUE(L2.empty());

  setList(&L1, X);
  checkList(&L1, X);

  setList(&L1, X, Y, Z);
  setList(&L2, A, B, C);
  L1.append_back(&L2);
  checkList(&L1, X, Y, Z, A, B, C);
  EXPECT_TRUE(L2.empty());

  setList(&L1, X, Y);
  setList(&L2);
  L1.append_front(&L2);
  checkList(&L1, X, Y);
  EXPECT_TRUE(L2.empty());
}

TEST(ScudoStandalone, IntrusiveListAppendEmpty) {
  ListItem I;
  List L;
  L.clear();
  L.push_back(&I);
  List L2;
  L2.clear();
  L.append_back(&L2);
  EXPECT_EQ(L.back(), &I);
  EXPECT_EQ(L.front(), &I);
  EXPECT_EQ(L.size(), 1U);
  L.append_front(&L2);
  EXPECT_EQ(L.back(), &I);
  EXPECT_EQ(L.front(), &I);
  EXPECT_EQ(L.size(), 1U);
}
