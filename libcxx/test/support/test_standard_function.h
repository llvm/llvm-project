//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBCXX_TEST_SUPPORT_TEST_STANDARD_FUNCTION_H
#define LIBCXX_TEST_SUPPORT_TEST_STANDARD_FUNCTION_H

#include "test_macros.h"

#if TEST_STD_VER >= 20
template <class T>
constexpr bool is_addressable = requires(T t) {
  &t;
};

template <class T>
[[nodiscard]] constexpr bool is_function_like() {
  using X = std::remove_cvref_t<T>;
  static_assert(!is_addressable<X>);
  static_assert(!is_addressable<X const>);

  static_assert(std::destructible<X> && !std::default_initializable<X>);

  static_assert(!std::move_constructible<X>);
  static_assert(!std::assignable_from<X&, X>);

  static_assert(!std::copy_constructible<X>);
  static_assert(!std::assignable_from<X&, X const>);
  static_assert(!std::assignable_from<X&, X&>);
  static_assert(!std::assignable_from<X&, X const&>);
  static_assert(std::is_final_v<X>);
  return true;
}
#endif

#endif // LIBCXX_TEST_SUPPORT_TEST_STANDARD_FUNCTION_H
