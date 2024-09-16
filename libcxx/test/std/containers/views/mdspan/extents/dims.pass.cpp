//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>

// template<size_t Rank, class IndexType = size_t>
//     using dims = see below;
//
// Result: A type E that is a specialization of extents such that
//         E::rank() == Rank && E::rank() == E::rank_dynamic() is true,
//         and E::index_type denotes IndexType.

#include <mdspan>
#include <cstddef>
#include <span> // dynamic_extent

#include "test_macros.h"

template <class IndexType>
void test_alias_template_dims() {
  constexpr size_t D = std::dynamic_extent;
  ASSERT_SAME_TYPE(std::dims<0, IndexType>, std::extents<IndexType>);
  ASSERT_SAME_TYPE(std::dims<1, IndexType>, std::extents<IndexType, D>);
  ASSERT_SAME_TYPE(std::dims<2, IndexType>, std::extents<IndexType, D, D>);
  ASSERT_SAME_TYPE(std::dims<3, IndexType>, std::extents<IndexType, D, D, D>);
  ASSERT_SAME_TYPE(std::dims<9, IndexType>, std::extents<IndexType, D, D, D, D, D, D, D, D, D>);
}

template <>
void test_alias_template_dims<size_t>() {
  constexpr size_t D = std::dynamic_extent;
  ASSERT_SAME_TYPE(std::dims<0>, std::extents<size_t>);
  ASSERT_SAME_TYPE(std::dims<1>, std::extents<size_t, D>);
  ASSERT_SAME_TYPE(std::dims<2>, std::extents<size_t, D, D>);
  ASSERT_SAME_TYPE(std::dims<3>, std::extents<size_t, D, D, D>);
  ASSERT_SAME_TYPE(std::dims<9>, std::extents<size_t, D, D, D, D, D, D, D, D, D>);
}

int main(int, char**) {
  test_alias_template_dims<int>();
  test_alias_template_dims<unsigned int>();
  test_alias_template_dims<size_t>();
  return 0;
}
