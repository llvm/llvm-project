//===-- lib/openmp/omp_util.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of PointerDeviceMap -- a thread-safe pointer-to-device-ID
// map used by the OpenMP allocator runtime to track allocation origins.
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/OpenMP/omp_util.h"
#include "flang-rt/runtime/lock.h"
#include "flang-rt/runtime/terminator.h"
#include <cstdlib>
#include <cstring>

namespace Fortran::runtime::omp {

static constexpr std::size_t initialCapacity{256};

static Lock pointerDeviceMapLock;

/// Double the capacity of the entries array (or set it to initialCapacity if
/// empty).  Crashes on allocation failure.  Must be called under the lock.
void PointerDeviceMap::grow() {
  std::size_t newCapacity = capacity_ ? capacity_ * 2 : initialCapacity;
  Entry *newEntries =
      static_cast<Entry *>(std::realloc(entries_, newCapacity * sizeof(Entry)));
  if (!newEntries) {
    Terminator{__FILE__, __LINE__}.Crash(
        "PointerDeviceMap: realloc failed (capacity %zu)", newCapacity);
  }
  entries_ = newEntries;
  capacity_ = newCapacity;
}

/// Record that \p pointer was allocated on \p device.  Thread-safe.
void PointerDeviceMap::insert(void *pointer, int device) {
  CriticalSection guard(pointerDeviceMapLock);
  if (count_ == capacity_) {
    grow();
  }
  entries_[count_++] = {pointer, device};
}

/// Remove \p pointer from the map and return its device ID, or -1 if not
/// found.  Uses swap-with-last for O(1) removal.  Thread-safe.
int PointerDeviceMap::removeAndGet(void *pointer) {
  CriticalSection guard(pointerDeviceMapLock);
  for (std::size_t i = 0; i < count_; ++i) {
    if (entries_[i].pointer == pointer) {
      int device = entries_[i].device;
      // Swap with last entry and shrink.
      entries_[i] = entries_[--count_];
      return device;
    }
  }
  return -1;
}

/// Print all (pointer, device) entries to stderr.  Thread-safe.
/// Can be used for debugging purposes.
void PointerDeviceMap::dump() const {
  CriticalSection guard(pointerDeviceMapLock);
  for (std::size_t i = 0; i < count_; ++i) {
    std::fprintf(stderr, "%p -> %d\n", entries_[i].pointer, entries_[i].device);
  }
}

} // namespace Fortran::runtime::omp
