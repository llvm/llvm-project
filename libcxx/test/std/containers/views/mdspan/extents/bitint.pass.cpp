//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class IndexType, size_t... Extents>
//  class extents;
//
// After [libc++] recognized _BitInt as an integer type in
// __type_traits/integer_traits.h, extents silently starts accepting
// _BitInt(N) as IndexType because __signed_or_unsigned_integer is now
// satisfied. This test pins that behavior: construction, extent() and
// static_extent() all work, and the representability static_assert on
// static extents fires when the extent does not fit in the index type.

#include <cassert>
#include <cstddef>
#include <mdspan>
#include <type_traits>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int)

template <class IndexType>
constexpr bool test_extents_with_index_type() {
  using Ext = std::extents<IndexType, 3, std::dynamic_extent, 7>;
  static_assert(std::is_same_v<typename Ext::index_type, IndexType>);
  static_assert(Ext::rank() == 3);
  static_assert(Ext::rank_dynamic() == 1);

  Ext e(IndexType{5});
  assert(e.extent(0) == IndexType{3});
  assert(e.extent(1) == IndexType{5});
  assert(e.extent(2) == IndexType{7});
  assert(Ext::static_extent(0) == 3);
  assert(Ext::static_extent(1) == std::dynamic_extent);
  assert(Ext::static_extent(2) == 7);

  // All-dynamic form.
  using DynExt = std::dextents<IndexType, 2>;
  static_assert(std::is_same_v<typename DynExt::index_type, IndexType>);
  DynExt d(IndexType{4}, IndexType{6});
  assert(d.extent(0) == IndexType{4});
  assert(d.extent(1) == IndexType{6});
  return true;
}

constexpr bool test() {
  // Signed _BitInt index types across the width tiers.
  test_extents_with_index_type<_BitInt(13)>();
  test_extents_with_index_type<_BitInt(32)>();
  test_extents_with_index_type<_BitInt(64)>();

  // Unsigned _BitInt index types.
  test_extents_with_index_type<unsigned _BitInt(13)>();
  test_extents_with_index_type<unsigned _BitInt(32)>();
  test_extents_with_index_type<unsigned _BitInt(64)>();

#  if __BITINT_MAXWIDTH__ >= 128
  test_extents_with_index_type<_BitInt(128)>();
  test_extents_with_index_type<unsigned _BitInt(128)>();
#  endif

  return true;
}

#endif // TEST_HAS_EXTENSION(bit_int)

int main(int, char**) {
#if TEST_HAS_EXTENSION(bit_int)
  test();
  static_assert(test());
#endif
  return 0;
}
