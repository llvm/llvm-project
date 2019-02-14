//===-- list.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_LIST_H_
#define SCUDO_LIST_H_

#include "internal_defs.h"

namespace scudo {

// Intrusive POD singly-linked list.
// An object with all zero fields should represent a valid empty list. clear()
// should be called on all non-zero-initialized objects before using.
template <class Item> struct IntrusiveList {
  friend class Iterator;

  void clear() {
    First = Last = nullptr;
    Size = 0;
  }

  bool empty() const { return Size == 0; }
  uptr size() const { return Size; }

  void push_back(Item *X) {
    if (empty()) {
      X->Next = nullptr;
      First = Last = X;
      Size = 1;
    } else {
      X->Next = nullptr;
      Last->Next = X;
      Last = X;
      Size++;
    }
  }

  void push_front(Item *X) {
    if (empty()) {
      X->Next = nullptr;
      First = Last = X;
      Size = 1;
    } else {
      X->Next = First;
      First = X;
      Size++;
    }
  }

  void pop_front() {
    DCHECK(!empty());
    First = First->Next;
    if (!First)
      Last = nullptr;
    Size--;
  }

  void extract(Item *Prev, Item *X) {
    DCHECK(!empty());
    DCHECK_NE(Prev, nullptr);
    DCHECK_NE(X, nullptr);
    DCHECK_EQ(Prev->Next, X);
    Prev->Next = X->Next;
    if (Last == X)
      Last = Prev;
    Size--;
  }

  Item *front() { return First; }
  const Item *front() const { return First; }
  Item *back() { return Last; }
  const Item *back() const { return Last; }

  void append_front(IntrusiveList<Item> *L) {
    DCHECK_NE(this, L);
    if (L->empty())
      return;
    if (empty()) {
      *this = *L;
    } else if (!L->empty()) {
      L->Last->Next = First;
      First = L->First;
      Size += L->size();
    }
    L->clear();
  }

  void append_back(IntrusiveList<Item> *L) {
    DCHECK_NE(this, L);
    if (L->empty())
      return;
    if (empty()) {
      *this = *L;
    } else {
      Last->Next = L->First;
      Last = L->Last;
      Size += L->size();
    }
    L->clear();
  }

  void checkConsistency() {
    if (Size == 0) {
      CHECK_EQ(First, 0);
      CHECK_EQ(Last, 0);
    } else {
      uptr count = 0;
      for (Item *i = First;; i = i->Next) {
        count++;
        if (i == Last)
          break;
      }
      CHECK_EQ(size(), count);
      CHECK_EQ(Last->Next, 0);
    }
  }

  template <class ItemT> class IteratorBase {
  public:
    explicit IteratorBase(ItemT *CurrentItem) : Current(CurrentItem) {}
    IteratorBase &operator++() {
      Current = Current->Next;
      return *this;
    }
    bool operator!=(IteratorBase Other) const {
      return Current != Other.Current;
    }
    ItemT &operator*() { return *Current; }

  private:
    ItemT *Current;
  };

  typedef IteratorBase<Item> Iterator;
  typedef IteratorBase<const Item> ConstIterator;

  Iterator begin() { return Iterator(First); }
  Iterator end() { return Iterator(nullptr); }

  ConstIterator begin() const { return ConstIterator(First); }
  ConstIterator end() const { return ConstIterator(nullptr); }

private:
  uptr Size;
  Item *First;
  Item *Last;
};

} // namespace scudo

#endif // SCUDO_LIST_H_
