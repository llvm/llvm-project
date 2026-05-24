//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>

//  template<class ElementType, size_t ByteAlignment>
//  struct aligned_accessor {
//    using offset_policy = default_accessor<ElementType>;
//    using element_type = ElementType;
//    using reference = ElementType&;
//    using data_handle_type = ElementType*;
//
//    static constexpr size_t byte_alignment = ByteAlignment;
//
//    ...
//  };
//
//  Each specialization of aligned_accessor is a trivially copyable type that models semiregular.

#include <mdspan>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

template <class T, std::size_t N>
void test_types() {
  using A = std::aligned_accessor<T, N>;
  ASSERT_SAME_TYPE(typename A::offset_policy, std::default_accessor<T>);
  ASSERT_SAME_TYPE(typename A::element_type, T);
  ASSERT_SAME_TYPE(typename A::reference, T&);
  ASSERT_SAME_TYPE(typename A::data_handle_type, T*);

  ASSERT_SAME_TYPE(decltype(A::byte_alignment), const std::size_t);
  static_assert(A::byte_alignment == N);

  static_assert(std::semiregular<A>);
  static_assert(std::is_trivially_copyable_v<A>);

  LIBCPP_STATIC_ASSERT(std::is_empty_v<A>);
}

template <class T>
void test() {
  constexpr std::size_t N = alignof(T);
  test_types<T, N>();
  test_types<T, 2 * N>();
  test_types<T, 4 * N>();
  test_types<T, 8 * N>();
  test_types<T, 16 * N>();
  test_types<T, 32 * N>();
}

int main(int, char**) {
  test<int>();
  test<const int>();
  test<MinimalElementType>();
  test<const MinimalElementType>();
  return 0;
}
