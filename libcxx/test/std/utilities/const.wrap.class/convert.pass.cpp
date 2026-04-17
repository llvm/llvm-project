//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// constexpr operator decltype(value)() const noexcept { return value; }

#include <cassert>
#include <utility>

struct S {
  int value;

  constexpr S(int v) : value(v) {}
};

constexpr void f1(const S&) {}

constexpr bool test() {
  {
    // int conversion
    std::constant_wrapper<6> cw6;
    const int& result = cw6;
    assert(result == 6);
    assert(&result == &cw6.value);

    static_assert(noexcept(static_cast<const int&>(cw6)));
  }

  {
    // struct conversion
    constexpr S s{42};
    std::constant_wrapper<s> cws;
    const S& result = cws;
    assert(result.value == 42);
    assert(&result == &cws.value);

    static_assert(noexcept(static_cast<const S&>(cws)));
  }

  {
    // array conversion
    constexpr int arr[] = {1, 2, 3};
    std::constant_wrapper<arr> cwArr;
    const int (&result)[3] = cwArr;
    assert(result[0] == 1);
    assert(result[1] == 2);
    assert(result[2] == 3);
    assert(&result == &cwArr.value);

    static_assert(noexcept(static_cast<const int (&)[3]>(cwArr)));
  }

  {
    // function pointer conversion
    constexpr int (*fptr)(int) = [](int x) constexpr { return x * 2; };
    std::constant_wrapper<fptr> cwFptr;
    int (*result)(int) = cwFptr;
    assert(result(5) == 10);

    static_assert(noexcept(static_cast<int (*)(int)>(cwFptr)));
  }

  {
    // conversion is implicit
    std::constant_wrapper<S{42}> cws;
    f1(cws);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
