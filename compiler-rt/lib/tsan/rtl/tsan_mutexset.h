//===-- tsan_mutexset.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// MutexSet holds the set of mutexes currently held by a thread.
//===----------------------------------------------------------------------===//
#ifndef TSAN_MUTEXSET_H
#define TSAN_MUTEXSET_H

#include "tsan_defs.h"

namespace __tsan {

class MutexSet {
 public:
  // Holds limited number of mutexes.
  // The oldest mutexes are discarded on overflow.
  static const uptr kMaxSize = 16;
  struct Desc {
    uptr addr;
    StackID stack_id;
    u32 seq;
    u32 count;
    bool write;
  };

  MutexSet();
  void Add(uptr addr, StackID stack_id, bool write);
  void Del(uptr addr, bool destroy = false);
  uptr Size() const;
  Desc Get(uptr i) const;

  MutexSet(const MutexSet& other) {
    *this = other;
  }
  void operator=(const MutexSet &other) {
    internal_memcpy(this, &other, sizeof(*this));
  }

 private:
#if !SANITIZER_GO
   u32 seq_;
   uptr size_;
   Desc descs_[kMaxSize];
#endif

  void RemovePos(uptr i);
};

// Go does not have mutexes, so do not spend memory and time.
// (Go sync.Mutex is actually a semaphore -- can be unlocked
// in different goroutine).
#if SANITIZER_GO
MutexSet::MutexSet() {}
void MutexSet::Add(uptr addr, StackID stack_id, bool write) {
}
void MutexSet::Del(uptr addr, bool destroy) {
}
void MutexSet::RemovePos(uptr i) {}
uptr MutexSet::Size() const { return 0; }
MutexSet::Desc MutexSet::Get(uptr i) const { return Desc(); }
#endif

}  // namespace __tsan

#endif  // TSAN_MUTEXSET_H
