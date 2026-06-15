//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// constexpr mapping(const extents_type&);

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mdspan>

template <size_t PaddingValue, class Extents>
constexpr void
test_construction(Extents extents, std::array<typename Extents::index_type, Extents::rank()> expected_strides) {
  using Mapping = std::layout_right_padded<PaddingValue>::template mapping<Extents>;
  static_assert(Mapping::padding_value == PaddingValue);

  Mapping mapping(extents);
  assert(mapping.extents() == extents);

  for (typename Mapping::rank_type r = 0; r < Mapping::extents_type::rank(); ++r)
    assert(mapping.stride(r) == expected_strides[r]);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  // clang-format off
  test_construction<4>(std::extents<int32_t>(),                  {});
  test_construction<4>(std::extents<int32_t, 5>(),               {1});
  test_construction<4>(std::extents<uint32_t, D>(7),             {1});
  test_construction<4>(std::extents<uint32_t, 5, 7>(),           {8, 1});
  test_construction<4>(std::extents<uint64_t, D, 2, 3>(6),       {8, 4, 1});
  test_construction<D>(std::extents<int32_t>(),                  {});
  test_construction<D>(std::extents<int32_t, 5>(),               {1});
  test_construction<D>(std::extents<uint32_t, D>(7),             {1});
  test_construction<D>(std::extents<uint32_t, 0, 7>(),           {7, 1});
  test_construction<D>(std::extents<uint64_t, 5, 7>(),           {7, 1});
  test_construction<D>(std::extents<uint64_t, D, 7>(5),          {7, 1});
  test_construction<0>(std::extents<uint32_t, D, 13>(7),         {13, 1});
  test_construction<0>(std::extents<uint32_t, 0, 7>(),           {7, 1});
  //clang-format on

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
