//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class OtherIndexType>
//  constexpr mapping(const extents_type& e, array<OtherIndexType, rank_> s) noexcept;
//
// Constraints:
//    - is_convertible_v<const OtherIndexType&, index_type> is true, and
//    - is_nothrow_constructible_v<index_type, const OtherIndexType&> is true.
//
// Preconditions:
//    - s[i] > 0 is true for all i in the range [0, rank_).
//    - REQUIRED-SPAN-SIZE(e, s) is representable as a value of type index_type ([basic.fundamental]).
//    - If rank_ is greater than 0, then there exists a permutation P of the integers in the range [0, rank_),
//      such that s[pi] >= s[pi_1] * e.extent(pi_1) is true for all i in the range [1, rank_), where pi is the ith element of P.
//     Note 1: For layout_stride, this condition is necessary and sufficient for is_unique() to be true.
//
// Effects: Direct-non-list-initializes extents_ with e, and for all d in the range [0, rank_),
//         direct-non-list-initializes strides_[d] with as_const(s[d]).

#include <mdspan>
#include <cassert>
#include <cstdint>

#include "test_macros.h"
#include "../ConvertibleToIntegral.h"

template <class E, class S>
constexpr void test_construction(E e, S s) {
  using M = std::layout_stride::mapping<E>;
  ASSERT_NOEXCEPT(M{e, s});
  M m(e, s);

  // check correct extents are returned
  ASSERT_NOEXCEPT(m.extents());
  assert(m.extents() == e);

  // check required_span_size()
  typename E::index_type expected_size = 1;
  for (typename E::rank_type r = 0; r < E::rank(); r++) {
    if (e.extent(r) == 0) {
      expected_size = 0;
      break;
    }
    expected_size += (e.extent(r) - 1) * static_cast<typename E::index_type>(s[r]);
  }
  assert(m.required_span_size() == expected_size);

  // check strides: node stride function is constrained on rank>0, e.extent(r) is not
  auto strides = m.strides();
  ASSERT_NOEXCEPT(m.strides());
  if constexpr (E::rank() > 0) {
    for (typename E::rank_type r = 0; r < E::rank(); r++) {
      assert(m.stride(r) == static_cast<typename E::index_type>(s[r]));
      assert(strides[r] == m.stride(r));
    }
  }
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;
  {
    std::array<int, 0> s{};
    test_construction(std::extents<int>(), s);
  }
  {
    std::array<int, 1> s{1};
    test_construction(std::extents<unsigned, D>(7), s);
  }
  {
    std::array<int, 1> s{1};
    test_construction(std::extents<unsigned, D>(0), s);
  }
  {
    std::array<int, 1> s{2};
    test_construction(std::extents<unsigned, 7>(), s);
  }
  {
    std::array<IntType, 1> s{1};
    test_construction(std::extents<int, D>(7), s);
  }
  {
    std::array<int, 2> s{3, 30};
    test_construction(std::extents<unsigned, 7, 8>(), s);
  }
  {
    std::array<int, 4> s{20, 2, 200, 2000};
    test_construction(std::extents<int64_t, D, 8, D, D>(7, 9, 10), s);
    test_construction(std::extents<int64_t, D, 8, D, D>(0, 9, 10), s);
    test_construction(std::extents<int64_t, D, 8, D, D>(0, 8, 0), s);
  }
  {
    std::array<int, 4> s{200, 20, 20, 2000};
    test_construction(std::extents<int64_t, D, D, D, D>(7, 0, 8, 9), s);
    test_construction(std::extents<int64_t, D, D, D, D>(7, 8, 0, 9), s);
    test_construction(std::extents<int64_t, D, D, D, D>(7, 1, 8, 9), s);
    test_construction(std::extents<int64_t, D, D, D, D>(7, 8, 1, 9), s);
    test_construction(std::extents<int64_t, D, D, D, D>(7, 1, 1, 9), s);
    test_construction(std::extents<int64_t, D, D, D, D>(7, 0, 0, 9), s);
    test_construction(std::extents<int64_t, D, D, D, D>(7, 1, 1, 9), s);
    test_construction(std::extents<int64_t, D, D, D, D>(7, 1, 0, 9), s);
    test_construction(std::extents<int64_t, D, D, D, D>(7, 0, 1, 9), s);
  }

  {
    using mapping_t = std::layout_stride::mapping<std::dextents<unsigned, 2>>;
    // wrong strides size
    static_assert(!std::is_constructible_v<mapping_t, std::dextents<int, 2>, std::array<int, 3>>);
    static_assert(!std::is_constructible_v<mapping_t, std::dextents<int, 2>, std::array<int, 1>>);
    // wrong extents rank
    static_assert(!std::is_constructible_v<mapping_t, std::dextents<int, 3>, std::array<int, 2>>);
    // none-convertible strides
    static_assert(!std::is_constructible_v<mapping_t, std::dextents<int, 2>, std::array<IntType, 2>>);
  }
  {
    // not no-throw constructible index_type from stride
    using mapping_t = std::layout_stride::mapping<std::dextents<unsigned char, 2>>;
    static_assert(std::is_convertible_v<IntType, unsigned char>);
    static_assert(!std::is_constructible_v<mapping_t, std::dextents<int, 2>, std::array<IntType, 2>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
