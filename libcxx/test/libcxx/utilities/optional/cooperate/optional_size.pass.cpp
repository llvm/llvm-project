//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// template <class T> class optional;

#include <optional>
#include <functional>

template <class T>
struct type_with_bool {
  T value;
  bool has_value;
};

template <class T>
using cooperative_type =
#ifdef _LIBCPP_ABI_COOPERATIVE_OPTIONAL
    T;
#else
    type_with_bool<T>;
#endif

int main(int, char**) {
  static_assert(
      sizeof(std::optional<std::reference_wrapper<int>>) == sizeof(cooperative_type<std::reference_wrapper<int>>));
  static_assert(sizeof(std::optional<const std::reference_wrapper<int>>) ==
                sizeof(type_with_bool<const std::reference_wrapper<int>>));
  static_assert(
      sizeof(std::optional<std::reference_wrapper<int()>>) == sizeof(cooperative_type<std::reference_wrapper<int()>>));
  static_assert(sizeof(std::optional<const std::reference_wrapper<int()>>) ==
                sizeof(type_with_bool<const std::reference_wrapper<int()>>));

  return 0;
}
