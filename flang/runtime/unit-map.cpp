//===-- runtime/unit-map.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "unit-map.h"

namespace Fortran::runtime::io {

void UnitMap::Initialize() {
  if (!isInitialized_) {
    freeNewUnits_.InitializeState();
    // Unit number -1 is reserved.
    // The unit numbers are pushed in reverse order so that the first
    // ones to be popped will be small and suitable for use as kind=1
    // integers.
    for (int j{freeNewUnits_.maxValue}; j > 1; --j) {
      freeNewUnits_.Add(j);
    }
    isInitialized_ = true;
  }
}

// See 12.5.6.12 in Fortran 2018.  NEWUNIT= unit numbers are negative,
// and not equal to -1 (or ERROR_UNIT, if it were negative, which it isn't.)
ExternalFileUnit &UnitMap::NewUnit(const Terminator &terminator) {
  CriticalSection critical{lock_};
  Initialize();
  std::optional<int> n{freeNewUnits_.PopValue()};
  if (!n) {
    n = emergencyNewUnit_++;
  }
  return Create(-*n, terminator);
}

ExternalFileUnit *UnitMap::LookUpForClose(int n) {
  CriticalSection critical{lock_};
  Chain *previous{nullptr};
  int hash{Hash(n)};
  for (Chain *p{bucket_[hash].get()}; p; previous = p, p = p->next.get()) {
    if (p->unit.unitNumber() == n) {
      if (previous) {
        previous->next.swap(p->next);
      } else {
        bucket_[hash].swap(p->next);
      }
      // p->next.get() == p at this point; the next swap pushes p on closing_
      closing_.swap(p->next);
      return &p->unit;
    }
  }
  return nullptr;
}

void UnitMap::DestroyClosed(ExternalFileUnit &unit) {
  Chain *p{nullptr};
  {
    CriticalSection critical{lock_};
    Chain *previous{nullptr};
    for (p = closing_.get(); p; previous = p, p = p->next.get()) {
      if (&p->unit == &unit) {
        int n{unit.unitNumber()};
        if (n <= -2) {
          freeNewUnits_.Add(-n);
        }
        if (previous) {
          previous->next.swap(p->next);
        } else {
          closing_.swap(p->next);
        }
        break;
      }
    }
  }
  if (p) {
    p->unit.~ExternalFileUnit();
    FreeMemory(p);
  }
}

void UnitMap::CloseAll(IoErrorHandler &handler) {
  // Extract units from the map so they can be closed
  // without holding lock_.
  OwningPtr<Chain> closeList;
  {
    CriticalSection critical{lock_};
    for (int j{0}; j < buckets_; ++j) {
      while (Chain * p{bucket_[j].get()}) {
        bucket_[j].swap(p->next); // pops p from head of bucket list
        closeList.swap(p->next); // pushes p to closeList
      }
    }
  }
  while (Chain * p{closeList.get()}) {
    closeList.swap(p->next); // pops p from head of closeList
    p->unit.CloseUnit(CloseStatus::Keep, handler);
    p->unit.~ExternalFileUnit();
    FreeMemory(p);
  }
}

void UnitMap::FlushAll(IoErrorHandler &handler) {
  CriticalSection critical{lock_};
  for (int j{0}; j < buckets_; ++j) {
    for (Chain *p{bucket_[j].get()}; p; p = p->next.get()) {
      p->unit.FlushOutput(handler);
    }
  }
}

ExternalFileUnit *UnitMap::Find(const char *path) {
  if (path) {
    // TODO: Faster data structure
    for (int j{0}; j < buckets_; ++j) {
      for (Chain *p{bucket_[j].get()}; p; p = p->next.get()) {
        if (p->unit.path() && std::strcmp(p->unit.path(), path) == 0) {
          return &p->unit;
        }
      }
    }
  }
  return nullptr;
}

ExternalFileUnit &UnitMap::Create(int n, const Terminator &terminator) {
  Chain &chain{*New<Chain>{terminator}(n).release()};
  chain.next.reset(&chain);
  bucket_[Hash(n)].swap(chain.next); // pushes new node as list head
  return chain.unit;
}

} // namespace Fortran::runtime::io
