//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

//  template<class ElementType>
//  struct default_accessor {
//    using offset_policy = default_accessor;
//    using element_type = ElementType;
//    using reference = ElementType&;
//    using data_handle_type = ElementType*;
//    ...
//  };
//
//  Each specialization of default_accessor is a trivially copyable type that models semiregular.


#include <mdspan>
#include <type_traits>
#include <concepts>
#include <cassert>

#include "test_macros.h"

#include "../MinimalElementType.h"

template <class T>
void test() {
  using A = std::default_accessor<T>;
  ASSERT_SAME_TYPE(typename A::offset_policy, A);
  ASSERT_SAME_TYPE(typename A::element_type, T);
  ASSERT_SAME_TYPE(typename A::reference, T&);
  ASSERT_SAME_TYPE(typename A::data_handle_type, T*);

  static_assert(std::semiregular<A>);
  static_assert(std::is_trivially_copyable_v<A>);

  LIBCPP_STATIC_ASSERT(std::is_empty_v<A>);
}

int main(int, char**) {
  test<int>();
  test<const int>();
  test<MinimalElementType>();
  test<const MinimalElementType>();
  return 0;
}
