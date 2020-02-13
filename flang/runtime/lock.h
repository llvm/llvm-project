//===-- runtime/lock.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Wraps a mutex

#ifndef FORTRAN_RUNTIME_LOCK_H_
#define FORTRAN_RUNTIME_LOCK_H_

#include "terminator.h"
#include <mutex>

namespace Fortran::runtime {

class Lock {
public:
  void Take() { mutex_.lock(); }
  bool Try() { return mutex_.try_lock(); }
  void Drop() { mutex_.unlock(); }
  void CheckLocked(const Terminator &terminator) {
    if (Try()) {
      Drop();
      terminator.Crash("Lock::CheckLocked() failed");
    }
  }
private:
  std::mutex mutex_;
};

class CriticalSection {
public:
  explicit CriticalSection(Lock &lock) : lock_{lock} { lock_.Take(); }
  ~CriticalSection() { lock_.Drop(); }

private:
  Lock &lock_;
};
}

#endif  // FORTRAN_RUNTIME_LOCK_H_
