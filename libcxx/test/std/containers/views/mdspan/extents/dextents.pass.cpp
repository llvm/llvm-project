//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class IndexType, size_t Rank>
//     using dextents = see below;
//
// Result: A type E that is a specialization of extents such that
//         E::rank() == Rank && E::rank() == E::rank_dynamic() is true,
//         and E::index_type denotes IndexType.

#include <mdspan>
#include <cstddef>
#include <span> // dynamic_extent

#include "test_macros.h"

template <class IndexType>
void test_alias_template_dextents() {
  constexpr size_t D = std::dynamic_extent;
  ASSERT_SAME_TYPE(std::dextents<IndexType, 0>, std::extents<IndexType>);
  ASSERT_SAME_TYPE(std::dextents<IndexType, 1>, std::extents<IndexType, D>);
  ASSERT_SAME_TYPE(std::dextents<IndexType, 2>, std::extents<IndexType, D, D>);
  ASSERT_SAME_TYPE(std::dextents<IndexType, 3>, std::extents<IndexType, D, D, D>);
  ASSERT_SAME_TYPE(std::dextents<IndexType, 9>, std::extents<IndexType, D, D, D, D, D, D, D, D, D>);
}

int main(int, char**) {
  test_alias_template_dextents<int>();
  test_alias_template_dextents<unsigned int>();
  test_alias_template_dextents<size_t>();
  return 0;
}
