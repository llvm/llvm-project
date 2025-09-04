//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// XFAIL: availability-stacktrace-missing

/*
  (19.6.3.5) Comparison [stacktrace.entry.cmp]

namespace std {
  class stacktrace_entry {
  public:
    // [stacktrace.entry.cmp], comparison
    friend constexpr strong_ordering operator<=>(const stacktrace_entry& x,
                                                 const stacktrace_entry& y) noexcept;
*/

#include <cassert>
#include <cstdint>
#include <stacktrace>
#include <utility>

namespace {
int func1() { return 41; }
int func2() { return 42; }
} // namespace

int main(int, char**) {
  auto addr1 = uintptr_t(&func1);
  auto addr2 = uintptr_t(&func2);
  assert(addr1 != addr2);
  if (addr1 > addr2) {
    std::swap(addr1, addr2);
  }

  std::stacktrace_entry a;
  std::stacktrace_entry b;
  std::stacktrace_entry c;

  *(uintptr_t*)(&a) = uintptr_t(addr1);
  *(uintptr_t*)(&b) = uintptr_t(addr1);
  *(uintptr_t*)(&c) = uintptr_t(addr2);

  static_assert(noexcept(a <=> b));

  assert(std::strong_ordering::equal == (a <=> b));
  assert(std::strong_ordering::equivalent == (a <=> b));
  assert(std::strong_ordering::greater == (c <=> a));

  return 0;
}
