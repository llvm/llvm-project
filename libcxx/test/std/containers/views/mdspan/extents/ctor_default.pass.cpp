//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// Test default construction:
//
// constexpr extents() noexcept = default;
//
// Remarks: since the standard uses an exposition only array member, dynamic extents
// need to be zero intialized!

#include <mdspan>
#include <cassert>
#include <array>

#include "../ConvertibleToIntegral.h"
#include "CtorTestCombinations.h"
#include "test_macros.h"

struct DefaultCtorTest {
  template <class E, class AllExtents, class Extents, size_t... Indices>
  static constexpr void test_construction(AllExtents all_ext, Extents, std::index_sequence<Indices...>) {
    // This function gets called twice: once with Extents being just the dynamic ones, and once with all the extents specified.
    // We only test during the all extent case, since then Indices is the correct number. This allows us to reuse the same
    // testing machinery used in other constructor tests.
    if constexpr (sizeof...(Indices) == E::rank()) {
      ASSERT_NOEXCEPT(E{});
      // Need to construct new expected values, replacing dynamic values with 0
      std::array<typename AllExtents::value_type, E::rank()> expected_exts{
          ((E::static_extent(Indices) == std::dynamic_extent)
               ? typename AllExtents::value_type(0)
               : all_ext[Indices])...};
      test_runtime_observers(E{}, expected_exts);
    }
  }
};

int main(int, char**) {
  test_index_type_combo<DefaultCtorTest>();
  static_assert(test_index_type_combo<DefaultCtorTest>());
  return 0;
}
