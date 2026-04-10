//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// template<class LayoutRightPaddedMapping>
// constexpr mapping(const LayoutRightPaddedMapping& other) noexcept;

#include <cassert>
#include <cstddef>
#include <mdspan>
#include <type_traits>

#include "test_macros.h"

template <class Dst, class Src>
constexpr void assert_same_mapping(const Dst& dst, const Src& src) {
  assert(dst.extents() == src.extents());

  if constexpr (Dst::extents_type::rank() > 0) {
    for (typename Dst::rank_type r = 0; r < Dst::extents_type::rank(); ++r)
      assert(dst.stride(r) == static_cast<typename Dst::index_type>(src.stride(r)));
  }
}

template <bool Implicit, class Dst, class Src>
constexpr void test_conversion(const Src& source) {
  static_assert(std::is_constructible_v<Dst, Src>);

  ASSERT_NOEXCEPT(Dst(source));

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
    using Src = std::layout_right_padded<4>::mapping<std::extents<size_t>>;
    using Dst = std::layout_right::mapping<std::extents<size_t>>;
    Src source;
    test_conversion<true, Dst>(source);
  }

  {
    using Src = std::layout_right_padded<4>::mapping<std::extents<size_t, 5>>;
    using Dst = std::layout_right::mapping<std::extents<size_t, D>>;
    Src source;
    test_conversion<true, Dst>(source);
  }

  {
    using Src        = std::layout_right_padded<4>::mapping<std::extents<size_t, 4, 8>>;
    using DstStatic  = std::layout_right::mapping<std::extents<size_t, 4, 8>>;
    using DstDynamic = std::layout_right::mapping<std::extents<size_t, D, 8>>;
    Src source;
    test_conversion<true, DstStatic>(source);
    test_conversion<true, DstDynamic>(source);
  }

  {
    using Src = std::layout_right_padded<D>::mapping<std::extents<size_t, D, D>>;
    using Dst = std::layout_right::mapping<std::extents<size_t, D, D>>;
    Src source(std::extents<size_t, D, D>(4, 7), 7);
    test_conversion<true, Dst>(source);
  }

  {
    using Src = std::layout_right_padded<D>::mapping<std::extents<size_t, D, 8>>;
    using Dst = std::layout_right::mapping<std::extents<size_t, 4, 8>>;
    Src source(std::extents<size_t, D, 8>(4), 4);
    test_conversion<false, Dst>(source);
  }

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
