//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// template<class OtherExtents>
//   constexpr explicit(see below)
//     mapping(const layout_stride::mapping<OtherExtents>&);

#include <array>
#include <cassert>
#include <cstddef>
#include <mdspan>
#include <type_traits>

template <class Dst, class Src>
constexpr void assert_same_mapping(const Dst& dst, const Src& src) {
  assert(dst.extents() == src.extents());

  if constexpr (Dst::extents_type::rank() > 0) {
    for (typename Dst::rank_type r = 0; r < Dst::extents_type::rank(); ++r)
      assert(dst.stride(r) == static_cast<Dst::index_type>(src.stride(r)));
  }
}

template <bool Implicit, class Dst, class Src>
constexpr void test_conversion(const Src& source) {
  static_assert(std::is_constructible_v<Dst, Src>);

  Dst direct(source);
  assert_same_mapping(direct, source);

  if constexpr (Implicit) {
    Dst implicit = source;
    assert_same_mapping(implicit, source);
  } else {
    static_assert(!std::is_convertible_v<Src, Dst>);
  }
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  {
    using Src = std::layout_stride::mapping<std::extents<size_t>>;
    using Dst = std::layout_left_padded<D>::mapping<std::extents<size_t>>;
    Src source(std::extents<size_t>(), std::array<size_t, 0>{});
    test_conversion<true, Dst>(source);
  }

  {
    using Src = std::layout_stride::mapping<std::extents<size_t, 3>>;
    using Dst = std::layout_left_padded<4>::mapping<std::extents<size_t, 3>>;
    Src source(std::extents<size_t, 3>(), std::array<size_t, 1>{1});
    test_conversion<false, Dst>(source);
  }

  {
    using Src = std::layout_stride::mapping<std::extents<size_t, 5, 7>>;
    using Dst = std::layout_left_padded<4>::mapping<std::extents<size_t, D, 7>>;
    Src source(std::extents<size_t, 5, 7>(), std::array<size_t, 2>{1, 8});
    test_conversion<false, Dst>(source);
  }

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
