//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.5) Modifiers [stacktrace.basic.mod]
//
//   template<class Allocator>
//   void swap(basic_stacktrace& other)
//       noexcept(allocator_traits<Allocator>::propagate_on_container_swap::value ||
//                allocator_traits<Allocator>::is_always_equal::value);

#include <cassert>
#include <stacktrace>

namespace {

template <typename T>
struct AllocNoPropagate : std::allocator<T> {
  using propagate_on_container_swap = std::false_type;
  using is_always_equal             = std::false_type;
};

template <typename T>
struct AllocPropagate : std::allocator<T> {
  using propagate_on_container_swap = std::true_type;
  using is_always_equal             = std::false_type;
};

template <typename T>
struct AllocAlwaysEqual : std::allocator<T> {
  using propagate_on_container_swap = std::false_type;
  using is_always_equal             = std::true_type;
};

} // namespace

int main(int, char**) {
  std::stacktrace trace1;
  assert(trace1.empty());

  std::stacktrace trace2 = std::stacktrace::current();
  assert(!trace2.empty());

  trace1.swap(trace2);
  assert(!trace1.empty());
  assert(trace2.empty());

  // Check `noexcept`: `swap` is noexcept if either:
  //   (1) the allocator propagates on swap
  //   (2) if instances of that allocator type are always equal.

  {
    // `AllocPropagate` satisfies the first (but not the second); stacktrace swap should be noexcept
    using A       = AllocPropagate<std::stacktrace_entry>;
    auto prop_st1 = std::basic_stacktrace<A>();
    auto prop_st2 = std::basic_stacktrace<A>();
    static_assert(noexcept(prop_st1.swap(prop_st2)));
  }

  {
    // `AllocAlwaysEqual` satisfies second; stacktrace swap should be noexcept
    using A            = AllocAlwaysEqual<std::stacktrace_entry>;
    auto always_eq_st1 = std::basic_stacktrace<A>();
    auto always_eq_st2 = std::basic_stacktrace<A>();
    static_assert(noexcept(always_eq_st1.swap(always_eq_st2)));
  }

  {
    // `AllocNoPropagate` satisfies neither; stacktrace swap should *not* be noexcept
    using A          = AllocNoPropagate<std::stacktrace_entry>;
    auto no_prop_st1 = std::basic_stacktrace<A>();
    auto no_prop_st2 = std::basic_stacktrace<A>();
    static_assert(!noexcept(no_prop_st1.swap(no_prop_st2)));
  }

  return 0;
}
