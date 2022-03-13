//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// T* <copyable-box>::operator->()
// const T* <copyable-box>::operator->() const

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility> // in_place_t

#include "types.h"

template<class T>
constexpr void check() {
  // non-const version
  {
    std::ranges::__copyable_box<T> x(std::in_place, 10);
    T* result = x.operator->();
    static_assert(noexcept(x.operator->()));
    assert(result->value == 10);
    assert(x->value == 10);
  }

  // const version
  {
    std::ranges::__copyable_box<T> const x(std::in_place, 10);
    const T* result = x.operator->();
    static_assert(noexcept(x.operator->()));
    assert(result->value == 10);
    assert(x->value == 10);
  }
}

constexpr bool test() {
  check<CopyConstructible>(); // primary template
  check<Copyable>(); // optimization #1
  check<NothrowCopyConstructible>(); // optimization #2
  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
