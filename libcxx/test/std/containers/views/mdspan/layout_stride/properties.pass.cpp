//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// namespace std {
//   template<class Extents>
//   class layout_stride::mapping {
//
//     ...
//     static constexpr bool is_always_unique() noexcept { return true; }
//     static constexpr bool is_always_exhaustive() noexcept { return false; }
//     static constexpr bool is_always_strided() noexcept { return true; }
//
//     static constexpr bool is_unique() noexcept { return true; }
//     static constexpr bool is_exhaustive() noexcept;
//     static constexpr bool is_strided() noexcept { return true; }
//     ...
//   };
// }
//
//
// layout_stride::mapping<E> is a trivially copyable type that models regular for each E.
//
// constexpr bool is_exhaustive() const noexcept;
//
// Returns:
//   - true if rank_ is 0.
//   - Otherwise, true if there is a permutation P of the integers in the range [0, rank_) such that
//     stride(p0) equals 1, and stride(pi) equals stride(pi_1) * extents().extent(pi_1) for i in the
//     range [1, rank_), where pi is the ith element of P.
//   - Otherwise, false.

#include <mdspan>
#include <type_traits>
#include <concepts>
#include <cassert>

#include "test_macros.h"

template <class E>
constexpr void
test_layout_mapping_stride(E ext, std::array<typename E::index_type, E::rank()> strides, bool exhaustive) {
  using M = std::layout_stride::mapping<E>;
  M m(ext, strides);
  const M c_m = m;
  assert(m.strides() == strides);
  assert(c_m.strides() == strides);
  assert(m.extents() == ext);
  assert(c_m.extents() == ext);
  assert(M::is_unique() == true);
  assert(m.is_exhaustive() == exhaustive);
  assert(c_m.is_exhaustive() == exhaustive);
  assert(M::is_strided() == true);
  assert(M::is_always_unique() == true);
  assert(M::is_always_exhaustive() == false);
  assert(M::is_always_strided() == true);

  ASSERT_NOEXCEPT(m.strides());
  ASSERT_NOEXCEPT(c_m.strides());
  ASSERT_NOEXCEPT(m.extents());
  ASSERT_NOEXCEPT(c_m.extents());
  ASSERT_NOEXCEPT(M::is_unique());
  ASSERT_NOEXCEPT(m.is_exhaustive());
  ASSERT_NOEXCEPT(c_m.is_exhaustive());
  ASSERT_NOEXCEPT(M::is_strided());
  ASSERT_NOEXCEPT(M::is_always_unique());
  ASSERT_NOEXCEPT(M::is_always_exhaustive());
  ASSERT_NOEXCEPT(M::is_always_strided());

  for (typename E::rank_type r = 0; r < E::rank(); r++) {
    assert(m.stride(r) == strides[r]);
    assert(c_m.stride(r) == strides[r]);
    ASSERT_NOEXCEPT(m.stride(r));
    ASSERT_NOEXCEPT(c_m.stride(r));
  }

  typename E::index_type expected_size = 1;
  for (typename E::rank_type r = 0; r < E::rank(); r++) {
    if (ext.extent(r) == 0) {
      expected_size = 0;
      break;
    }
    expected_size += (ext.extent(r) - 1) * static_cast<typename E::index_type>(strides[r]);
  }
  assert(m.required_span_size() == expected_size);
  assert(c_m.required_span_size() == expected_size);
  ASSERT_NOEXCEPT(m.required_span_size());
  ASSERT_NOEXCEPT(c_m.required_span_size());

  static_assert(std::is_trivially_copyable_v<M>);
  static_assert(std::regular<M>);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;
  test_layout_mapping_stride(std::extents<int>(), std::array<int, 0>{}, true);
  test_layout_mapping_stride(std::extents<signed char, 4, 5>(), std::array<signed char, 2>{1, 4}, true);
  test_layout_mapping_stride(std::extents<signed char, 4, 5>(), std::array<signed char, 2>{1, 5}, false);
  test_layout_mapping_stride(std::extents<unsigned, D, 4>(7), std::array<unsigned, 2>{20, 2}, false);
  test_layout_mapping_stride(std::extents<size_t, D, D, D, D>(3, 3, 3, 3), std::array<size_t, 4>{3, 1, 9, 27}, true);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
