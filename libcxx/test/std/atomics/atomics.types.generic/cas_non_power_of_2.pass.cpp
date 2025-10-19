//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// https://github.com/llvm/llvm-project/issues/30023
// compare exchange does not work with types of which the size is not a power of 2

// XFAIL: clang-19, clang-20, clang-21, apple-clang-15, apple-clang-16, apple-clang-17
// UNSUPPORTED: c++03

// TODO: remove the UNSUPPORTED clang-22 once libc++ CI's clang is updated to include
// the fix https://github.com/llvm/llvm-project/pull/78707
// UNSUPPORTED: clang-22

#include <atomic>
#include <cstring>
#include <cassert>

template <int Size>
struct S {
  char data[Size];

  explicit S(char v = 0) noexcept { memset(&data[0], v, sizeof(data)); }

  // only used in the test to check the results. Not used in atomic operations.
  friend bool operator==(const S& lhs, const S& rhs) noexcept {
    return memcmp(&lhs.data[0], &rhs.data[0], sizeof(lhs.data)) == 0;
  }
  friend bool operator!=(const S& lhs, const S& rhs) noexcept { return !(lhs == rhs); }
};

template <int Size>
struct Expected {
  Expected(S<Size> ss) : s(ss) {}

  S<Size> s;
  bool b = true; // used to validate that s's operation won't overwrite the memory next to it
};

template <int Size>
void test() {
  using T = S<Size>;
  std::atomic<T> a(T(0));
  Expected<Size> expected{T(17)};

  assert(a.load() != expected.s);
  assert(expected.b);

  auto r1 = a.compare_exchange_strong(expected.s, T(18), std::memory_order_relaxed);

  assert(!r1);
  assert(expected.s == T(0)); // expected.s is modified by compare_exchange_strong
  assert(expected.b);
  assert(a.load() == T(0));

  auto r2 = a.compare_exchange_strong(expected.s, T(18), std::memory_order_relaxed);
  assert(r2);
  assert(a.load() == T(18));
  assert(expected.s == T(0));
  assert(expected.b);
}

int main(int, char**) {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();
  test<6>();

  return 0;
}
