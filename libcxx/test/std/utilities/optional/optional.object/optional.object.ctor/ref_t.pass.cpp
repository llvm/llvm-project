//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

#include <cassert>
#include <optional>
#include <type_traits>
#include <utility>

template <typename RefType, std::remove_reference_t<RefType> _Val>
constexpr bool test() {
  std::remove_reference_t<RefType> item{_Val};
  std::optional<RefType> opt{item};

  {
    assert(*opt == item);
    assert(&(*opt) == &item);
  }
  {
    assert(*std::as_const(opt) == item);
    assert(&(*std::as_const(opt)) == &item);
  }

  return true;
}

template <typename T>
constexpr T foo(T val) {
  return val;
}

template <typename T, T _Val>
constexpr bool fn_ref_test() {
  std::optional<T (&)(T)> opt{foo<T>};
  assert(opt.has_value());
  assert((*opt)(_Val) == _Val);

  return true;
}

template <typename T, T _Val>
constexpr bool array_ref_test() {
  T arr[5]{};
  std::optional<T(&)[5]> opt{arr};

  assert(opt.has_value());
  (*opt)[0] = _Val;
  assert((*opt)[0] == _Val);
  assert(arr[0] == _Val);

  return true;
}

constexpr bool tests() {
  assert((test<int&, 1>()));
  assert((test<double&, 1.0>()));
  assert((fn_ref_test<int, 1>()));
  assert((array_ref_test<int, 1>()));
  assert((fn_ref_test<double, 1.0>()));
  assert((array_ref_test<double, 1.0>()));
  return true;
}

int main(int, char**) {
  static_assert(tests());
  tests();
}
