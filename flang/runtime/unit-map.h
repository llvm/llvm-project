//===-- runtime/unit-map.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Maps Fortran unit numbers to their ExternalFileUnit instances.
// A simple hash table with forward-linked chains per bucket.

#ifndef FORTRAN_RUNTIME_UNIT_MAP_H_
#define FORTRAN_RUNTIME_UNIT_MAP_H_

#include "lock.h"
#include "memory.h"
#include "unit.h"
#include <cstdlib>

namespace Fortran::runtime::io {

class UnitMap {
public:
  ExternalFileUnit *LookUp(int n) {
    CriticalSection critical{lock_};
    return Find(n);
  }

  ExternalFileUnit &LookUpOrCreate(
      int n, const Terminator &terminator, bool &wasExtant) {
    CriticalSection critical{lock_};
    auto *p{Find(n)};
    wasExtant = p != nullptr;
    return p ? *p : Create(n, terminator);
  }

  // Unit look-up by name is needed for INQUIRE(FILE="...")
  ExternalFileUnit *LookUp(const char *path) {
    CriticalSection critical{lock_};
    return Find(path);
  }

  ExternalFileUnit &NewUnit(const Terminator &terminator) {
    CriticalSection critical{lock_};
    return Create(nextNewUnit_--, terminator);
  }

  // To prevent races, the unit is removed from the map if it exists,
  // and put on the closing_ list until DestroyClosed() is called.
  ExternalFileUnit *LookUpForClose(int);

  void DestroyClosed(ExternalFileUnit &);
  void CloseAll(IoErrorHandler &);
  void FlushAll(IoErrorHandler &);

private:
  struct Chain {
    explicit Chain(int n) : unit{n} {}
    ExternalFileUnit unit;
    OwningPtr<Chain> next{nullptr};
  };

  static constexpr int buckets_{1031}; // must be prime
  int Hash(int n) { return std::abs(n) % buckets_; }

  ExternalFileUnit *Find(int n) {
    Chain *previous{nullptr};
    int hash{Hash(n)};
    for (Chain *p{bucket_[hash].get()}; p; previous = p, p = p->next.get()) {
      if (p->unit.unitNumber() == n) {
        if (previous) {
          // Move found unit to front of chain for quicker lookup next time
          previous->next.swap(p->next); // now p->next.get() == p
          bucket_[hash].swap(p->next); // now bucket_[hash].get() == p
        }
        return &p->unit;
      }
    }
    return nullptr;
  }
  ExternalFileUnit *Find(const char *path);

  ExternalFileUnit &Create(int, const Terminator &);

  Lock lock_;
  OwningPtr<Chain> bucket_[buckets_]{}; // all owned by *this
  int nextNewUnit_{-1000}; // see 12.5.6.12 in Fortran 2018
  OwningPtr<Chain> closing_{nullptr}; // units during CLOSE statement
};
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_UNIT_MAP_H_
