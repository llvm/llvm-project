//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <copyable-box>::<copyable-box>()

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility> // in_place_t

#include "types.h"

template<class T>
using Box = std::ranges::__copyable_box<T>;

struct NoDefault {
  NoDefault() = delete;
};
static_assert(!std::is_default_constructible_v<Box<NoDefault>>);

template<bool Noexcept>
struct DefaultNoexcept {
  DefaultNoexcept() noexcept(Noexcept);
};
static_assert( std::is_nothrow_default_constructible_v<Box<DefaultNoexcept<true>>>);
static_assert(!std::is_nothrow_default_constructible_v<Box<DefaultNoexcept<false>>>);

constexpr bool test() {
  // check primary template
  {
    Box<CopyConstructible> box;
    assert(box.__has_value());
    assert((*box).value == CopyConstructible().value);
  }

  // check optimization #1
  {
    Box<Copyable> box;
    assert(box.__has_value());
    assert((*box).value == Copyable().value);
  }

  // check optimization #2
  {
    Box<NothrowCopyConstructible> box;
    assert(box.__has_value());
    assert((*box).value == NothrowCopyConstructible().value);
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
