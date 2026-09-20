//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.4) Comparisons [stacktrace.basic.cmp]
//
//   template<class Allocator2>
//   friend bool operator==(const basic_stacktrace& x,
//                          const basic_stacktrace<Allocator2>& y) noexcept;

#include <cassert>
#include <memory_resource>
#include <stacktrace>
#include <utility>
#include <vector>

#include "test_macros.h"

// Call chain is: main -> c -> b -> a -> stacktrace::current;
// we're only checking a, b, c in the returned stacktrace, so use max_depth of 3.
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace a(size_t skip = 0) { return std::stacktrace::current(skip, 3); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace b(size_t skip = 0) { return a(skip); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace c(size_t skip = 0) { return b(skip); }

// Two independent captures of a real stacktrace (even at the "same" call site) needn't compare
// equal across two different allocator template instantiations, since each instantiation of a
// templated call chain is its own distinct function with its own addresses. So for a
// deterministic, allocator-independent cross-allocator equality check, synthesize traces with
// specific addresses instead, using the same internal-access technique as
// basic.cmp/strong_ordering.pass.cpp's `fake_trace`.
namespace {
template <class Alloc>
std::basic_stacktrace<Alloc> fake_trace(std::vector<uintptr_t> const& addrs, Alloc const& alloc = Alloc()) {
  std::basic_stacktrace<Alloc> ret(alloc);
  auto& base = *reinterpret_cast<std::__stacktrace::_Trace*>(&ret);
  for (uintptr_t addr : addrs) {
    auto& entry = base.__entry_append_();
    auto& eb    = *reinterpret_cast<std::__stacktrace::_Entry*>(&entry);
    eb.__addr_  = addr;
  }
  return ret;
}
} // namespace

int main(int, char**) {
  std::stacktrace st0;

  static_assert(noexcept(st0 == st0));
  static_assert(noexcept(st0 != st0));
  assert(st0 == st0);

  std::stacktrace st1 = a();
  assert(st1 != st0);

  std::stacktrace st2 = b();
  assert(st2 != st1);
  assert(st2 != st0);

  std::stacktrace st3 = c();
  assert(st3 != st0);
  assert(st3 != st1);
  assert(st3 != st2);
  assert(c() == st3);

  // Cross-allocator comparisons: `operator==` is a template over `Allocator2`, so a
  // `std::stacktrace` (std::allocator) must compare equal/unequal correctly against a
  // `std::pmr::stacktrace` (std::pmr::polymorphic_allocator) with matching/differing content.
  {
    std::allocator<std::stacktrace_entry> std_alloc;
    std::pmr::polymorphic_allocator<std::stacktrace_entry> pmr_alloc;

    static_assert(noexcept(std::declval<std::stacktrace const&>() == std::declval<std::pmr::stacktrace const&>()));

    std::stacktrace same_content     = fake_trace({100, 200, 300}, std_alloc);
    std::pmr::stacktrace pmr_content = fake_trace({100, 200, 300}, pmr_alloc);
    assert(same_content == pmr_content);
    assert(pmr_content == same_content);
    assert(!(same_content != pmr_content));

    std::pmr::stacktrace pmr_empty;
    assert(same_content != pmr_empty);
    assert(pmr_empty != same_content);
    assert(st0 == pmr_empty);
  }

  return 0;
}
