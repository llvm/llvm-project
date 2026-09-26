//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.2)
// [stacktrace.basic.cons], creation and assignment
//
//   static basic_stacktrace current(size_type skip, size_type max_depth,
//                                 const allocator_type& alloc = allocator_type()) noexcept;
//
// `current()` is `noexcept`, so an allocator that throws while capturing must not escape as an
// exception (which would call std::terminate); it must instead produce a trace populated with
// whatever entries were successfully captured before the failure, possibly empty.

#include <cassert>
#include <cstddef>
#include <new>
#include <stacktrace>

#include "test_macros.h"

// Throws std::bad_alloc from the `n`-th call to allocate() onward (1-indexed); a negative count
// never throws. Shared (not per-instance) so every rebind/copy of the allocator sees the same
// countdown, matching how vector's growth reallocation uses its allocator.
template <class T>
struct throwing_allocator {
  using value_type = T;

  static inline long throw_after_n_allocs = -1;

  throwing_allocator() = default;
  template <class U>
  throwing_allocator(throwing_allocator<U> const&) noexcept {}

  T* allocate(std::size_t n) {
    if (throw_after_n_allocs == 0) {
      throw std::bad_alloc();
    }
    if (throw_after_n_allocs > 0) {
      --throw_after_n_allocs;
    }
    return std::allocator<T>().allocate(n);
  }
  void deallocate(T* p, std::size_t n) { std::allocator<T>().deallocate(p, n); }

  template <class U>
  bool operator==(throwing_allocator<U> const&) const {
    return true;
  }
};

using throwing_stacktrace = std::basic_stacktrace<throwing_allocator<std::stacktrace_entry>>;

TEST_NO_TAIL_CALLS TEST_NOINLINE throwing_stacktrace f() { return throwing_stacktrace::current(); }

int main(int, char**) {
  // Every allocation fails: current() must come back empty, not terminate.
  throwing_allocator<std::stacktrace_entry>::throw_after_n_allocs = 0;
  throwing_stacktrace empty                                       = f();
  assert(empty.empty());

  // The first allocation (room for the first captured frame) succeeds, everything after fails:
  // current() must retain that one entry rather than discard everything.
  throwing_allocator<std::stacktrace_entry>::throw_after_n_allocs = 1;
  throwing_stacktrace partial                                     = f();
  assert(partial.size() >= 1);

  throwing_allocator<std::stacktrace_entry>::throw_after_n_allocs = -1;

  return 0;
}
