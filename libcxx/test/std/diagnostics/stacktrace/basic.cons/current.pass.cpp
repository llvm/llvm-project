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
//   static basic_stacktrace current(const allocator_type& alloc = allocator_type()) noexcept;
//
//   static basic_stacktrace current(size_type skip,
//                                 const allocator_type& alloc = allocator_type()) noexcept;
//
//   static basic_stacktrace current(size_type skip, size_type max_depth,
//                                 const allocator_type& alloc = allocator_type()) noexcept;
//
// Hardened requirements for the `current` call with given `skip` and `max_depth` amounts:
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3697r0.html#basic_stacktrace
// Specifically: "Hardened preconditions: skip <= skip + max_depth is true."

#include <cassert>
#include <cstdint>
#include <stacktrace>
#include <vector>

#include "test_macros.h"

// These let us produce dummy functions with distinct addresses to build test stacks.
// `main` calls `c`, which calls `b`, which calls `a`, which calls `current`;
// therefore the stacktrace built in `a` should contain, starting at the top,
// `a`, `b`, `c` (followed by `main`, `_start` on some platforms, and so on).

TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace a(size_t skip = 0, size_t max_depth = 99) {
  return std::stacktrace::current(skip, max_depth);
}
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace b(size_t skip = 0, size_t max_depth = 99) {
  return a(skip, max_depth);
}
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace c(size_t skip = 0, size_t max_depth = 99) {
  return b(skip, max_depth);
}

// We cannot guarantee that we can resolve symbols at runtime in the test environment,
// so we can't simply check the stacktrace entries' descriptions.  But we can at least get
// the lower bounds of these functions.
uintptr_t _a = reinterpret_cast<uintptr_t>(&a);
uintptr_t _b = reinterpret_cast<uintptr_t>(&b);
uintptr_t _c = reinterpret_cast<uintptr_t>(&c);

void expect_trace(const std::stacktrace& st, std::vector<uintptr_t> const& expect_chain) {
  auto trace_it = st.begin();
  for (uintptr_t expect_addr : expect_chain) {
    std::stacktrace_entry const& entry = *trace_it++;
    assert(entry.native_handle() >= expect_addr);
  }
}

int main(int, char**) {
  // All overloads are noexcept
  static_assert(noexcept(std::stacktrace::current()));
  static_assert(noexcept(std::stacktrace::current({})));
  static_assert(noexcept(std::stacktrace::current({}, {})));

  // Build stacktraces with different skip and max_depth values.
  // Call chain: main -> c -> b -> a -> current.
  // skip == 0 yields top frames a, b, c (we ignore other frames).
  // skip removes that many top frames; max_depth caps the number of frames returned.

  expect_trace(c(0, 3), {_a, _b, _c});
  expect_trace(c(0, 2), {_a, _b});
  expect_trace(c(0, 1), {_a});
  expect_trace(c(0, 0), {});

  expect_trace(c(1, 2), {_b, _c});
  expect_trace(c(1, 1), {_b});
  expect_trace(c(1, 0), {});

  expect_trace(c(2, 1), {_c});
  expect_trace(c(2, 0), {});

  return 0;
}
