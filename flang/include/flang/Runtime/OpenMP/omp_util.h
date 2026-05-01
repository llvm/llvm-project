//===-- include/flang/Runtime/OpenMP/omp_util.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_OMP_UTIL_H_
#define FORTRAN_RUNTIME_OMP_UTIL_H_

#include <cstddef>

namespace Fortran::runtime::omp {

/// A thread-safe map from allocation pointer to device ID.
///
/// Used to remember which OpenMP device each pointer was allocated on,
/// so that deallocation can target the correct device even if
/// omp_set_default_device() was called in between.
///
/// Implemented as a dynamically-grown flat array with linear search and
/// a global lock, to avoid pulling in C++ runtime dependencies (e.g.
/// std::unordered_map).  This is adequate for the expected allocation
/// counts in typical Fortran programs.
class PointerDeviceMap {
public:
  /// Record that \p pointer was allocated on \p device.
  void insert(void *pointer, int device);

  /// Remove the entry for \p pointer and return the device ID it was
  /// allocated on.  Returns -1 if \p pointer is not in the map.
  int removeAndGet(void *pointer);

  /// Print all entries to stderr (for debugging).
  void dump() const;

private:
  struct Entry {
    void *pointer;
    int device;
  };

  void grow();

  Entry *entries_{nullptr};
  std::size_t count_{0};
  std::size_t capacity_{0};
};

} // namespace Fortran::runtime::omp

#endif // FORTRAN_RUNTIME_OMP_UTIL_H_
