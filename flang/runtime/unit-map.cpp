//===-- runtime/unit-map.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "unit-map.h"

namespace Fortran::runtime::io {

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
  CriticalSection critical{lock_};
  for (int j{0}; j < buckets_; ++j) {
    while (Chain * p{bucket_[j].get()}) {
      bucket_[j].swap(p->next); // pops p from head of list
      p->unit.CloseUnit(CloseStatus::Keep, handler);
      p->unit.~ExternalFileUnit();
      FreeMemory(p);
    }
  }
}

void UnitMap::FlushAll(IoErrorHandler &handler) {
  CriticalSection critical{lock_};
  for (int j{0}; j < buckets_; ++j) {
    for (Chain *p{bucket_[j].get()}; p; p = p->next.get()) {
      p->unit.Flush(handler);
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
