//===-- list_test.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "list.h"

#include <array>

struct ListItemLinkedWithPtr {
  ListItemLinkedWithPtr *Next;
  ListItemLinkedWithPtr *Prev;
};

struct ListItemLinkedWithIndex {
  scudo::uptr Next;
  scudo::uptr Prev;
  static constexpr scudo::uptr EndOfListVal = 1ULL << 30;
};

template <typename ListT, typename ListItemTy>
static void setList(ListT *L, ListItemTy *I1 = nullptr,
                    ListItemTy *I2 = nullptr, ListItemTy *I3 = nullptr) {
  L->clear();
  if (I1)
    L->push_back(I1);
  if (I2)
    L->push_back(I2);
  if (I3)
    L->push_back(I3);
}

template <typename ListT, typename ListItemTy>
static void checkList(ListT *L, ListItemTy *I1, ListItemTy *I2 = nullptr,
                      ListItemTy *I3 = nullptr, ListItemTy *I4 = nullptr,
                      ListItemTy *I5 = nullptr, ListItemTy *I6 = nullptr) {
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

template <template <typename> class ListTy, typename ListItemTy>
static void testListCommon(void) {
  ListItemTy Items[3];
  ListItemTy *X = &Items[0];
  ListItemTy *Y = &Items[1];
  ListItemTy *Z = &Items[2];

  ListTy<ListItemTy> L;
  L.clear();
  L.init(Items, sizeof(Items));

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

  // Verify the iterator
  std::array<ListItemTy *, 3> visitOrder{X, Y, Z};
  auto Iter = visitOrder.begin();
  for (const auto &Item : L) {
    EXPECT_EQ(&Item, *Iter);
    ++Iter;
  }
}

TEST(ScudoListTest, LinkedListCommon) {
  testListCommon<scudo::SinglyLinkedList, ListItemLinkedWithPtr>();
  testListCommon<scudo::SinglyLinkedList, ListItemLinkedWithIndex>();
  testListCommon<scudo::DoublyLinkedList, ListItemLinkedWithPtr>();
  testListCommon<scudo::DoublyLinkedList, ListItemLinkedWithIndex>();
}

template <template <typename> class ListTy, typename ListItemTy>
static void testSinglyLinkedList() {
  ListItemTy Items[6];
  ListItemTy *X = &Items[0];
  ListItemTy *Y = &Items[1];
  ListItemTy *Z = &Items[2];
  ListItemTy *A = &Items[3];
  ListItemTy *B = &Items[4];
  ListItemTy *C = &Items[5];

  ListTy<ListItemTy> L;
  L.clear();
  L.init(Items, sizeof(Items));

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

  ListTy<ListItemTy> L1, L2;
  L1.clear();
  L2.clear();
  L1.init(Items, sizeof(Items));
  L2.init(Items, sizeof(Items));

  L1.append_back(&L2);
  EXPECT_TRUE(L1.empty());
  EXPECT_TRUE(L2.empty());

  setList(&L1, X);
  checkList(&L1, X);

  setList(&L1, X, Y);
  L1.insert(X, Z);
  checkList(&L1, X, Z, Y);

  setList(&L1, X, Y, Z);
  setList(&L2, A, B, C);
  L1.append_back(&L2);
  checkList(&L1, X, Y, Z, A, B, C);
  EXPECT_TRUE(L2.empty());

  L1.clear();
  L2.clear();
  L1.push_back(X);
  L1.append_back(&L2);
  EXPECT_EQ(L1.back(), X);
  EXPECT_EQ(L1.front(), X);
  EXPECT_EQ(L1.size(), 1U);
}

TEST(ScudoListTest, SinglyLinkedList) {
  testSinglyLinkedList<scudo::SinglyLinkedList, ListItemLinkedWithPtr>();
  testSinglyLinkedList<scudo::SinglyLinkedList, ListItemLinkedWithIndex>();
}

template <template <typename> class ListTy, typename ListItemTy>
static void testDoublyLinkedList() {
  ListItemTy Items[3];
  ListItemTy *X = &Items[0];
  ListItemTy *Y = &Items[1];
  ListItemTy *Z = &Items[2];

  ListTy<ListItemTy> L;
  L.clear();
  L.init(Items, sizeof(Items));

  L.push_back(X);
  L.push_back(Y);
  L.push_back(Z);
  L.remove(Y);
  EXPECT_EQ(L.size(), 2U);
  EXPECT_EQ(L.front(), X);
  EXPECT_EQ(L.back(), Z);
  L.checkConsistency();
  L.remove(Z);
  EXPECT_EQ(L.size(), 1U);
  EXPECT_EQ(L.front(), X);
  EXPECT_EQ(L.back(), X);
  L.checkConsistency();
  L.pop_front();
  EXPECT_TRUE(L.empty());

  L.push_back(X);
  L.insert(Y, X);
  EXPECT_EQ(L.size(), 2U);
  EXPECT_EQ(L.front(), Y);
  EXPECT_EQ(L.back(), X);
  L.checkConsistency();
  L.remove(Y);
  EXPECT_EQ(L.size(), 1U);
  EXPECT_EQ(L.front(), X);
  EXPECT_EQ(L.back(), X);
  L.checkConsistency();
  L.pop_front();
  EXPECT_TRUE(L.empty());
}

TEST(ScudoListTest, DoublyLinkedList) {
  testDoublyLinkedList<scudo::DoublyLinkedList, ListItemLinkedWithPtr>();
  testDoublyLinkedList<scudo::DoublyLinkedList, ListItemLinkedWithIndex>();
}
