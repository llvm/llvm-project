//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-exceptions

// <mdspan>

// template<class... OtherIndexTypes>
//   constexpr reference at(OtherIndexTypes... indices) const;
//
// Constraints:
//   - (is_convertible_v<OtherIndexTypes, index_type> && ...) is true,
//   - (is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) is true, and
//   - sizeof...(OtherIndexTypes) == rank() is true.
//
// template<class OtherIndexType>
//   constexpr reference at(span<OtherIndexType, rank()> indices) const;
//
// template<class OtherIndexType>
//   constexpr reference at(const array<OtherIndexType, rank()>& indices) const;
//
// Constraints:
//   - is_convertible_v<const OtherIndexType&, index_type> is true, and
//   - is_nothrow_constructible_v<index_type, const OtherIndexType&> is true.
//
// Throws:
//   - std::out_of_range if extents_type::index-cast(indices) is not a multidimensional index in extents_.

#include <mdspan>
#include <string_view>
#include <cassert>

#include "test_macros.h"

template <typename F>
void test(F&& f) {
  try {
    f();
    assert(false && "Unexpected");
  } catch (const std::out_of_range& e) {
    LIBCPP_ASSERT(std::string_view(e.what()).contains("mdspan"));
  } catch (...) {
    assert(false && "Unexpected");
  }
}

int main() {
  float data[1024];
  // value out of range
  {
    std::mdspan m(data, std::extents<unsigned char, 5>());
    test([&] { TEST_IGNORE_NODISCARD m.at(-1); });
    test([&] { TEST_IGNORE_NODISCARD m.at(-130); });
    test([&] { TEST_IGNORE_NODISCARD m.at(5); });
    test([&] { TEST_IGNORE_NODISCARD m.at(256); });
    test([&] { TEST_IGNORE_NODISCARD m.at(1000); });
  }
  {
    std::mdspan m(data, std::extents<signed char, 5>());
    test([&] { TEST_IGNORE_NODISCARD m.at(-1); });
    test([&] { TEST_IGNORE_NODISCARD m.at(-130); });
    test([&] { TEST_IGNORE_NODISCARD m.at(5); });
    test([&] { TEST_IGNORE_NODISCARD m.at(128); });
    test([&] { TEST_IGNORE_NODISCARD m.at(1000); });
  }
  {
    std::mdspan m(data, std::dextents<unsigned char, 1>(5));
    test([&] { TEST_IGNORE_NODISCARD m.at(-1); });
    test([&] { TEST_IGNORE_NODISCARD m.at(-130); });
    test([&] { TEST_IGNORE_NODISCARD m.at(5); });
    test([&] { TEST_IGNORE_NODISCARD m.at(256); });
    test([&] { TEST_IGNORE_NODISCARD m.at(1000); });
  }
  {
    std::mdspan m(data, std::dextents<signed char, 1>(5));
    test([&] { TEST_IGNORE_NODISCARD m.at(-1); });
    test([&] { TEST_IGNORE_NODISCARD m.at(-130); });
    test([&] { TEST_IGNORE_NODISCARD m.at(5); });
    test([&] { TEST_IGNORE_NODISCARD m.at(128); });
    test([&] { TEST_IGNORE_NODISCARD m.at(1000); });
  }
  {
    std::mdspan m(data, std::dextents<int, 3>(5, 7, 9));
    test([&] { TEST_IGNORE_NODISCARD m.at(-1, -1, -1); });
    test([&] { TEST_IGNORE_NODISCARD m.at(-1, 0, 0); });
    test([&] { TEST_IGNORE_NODISCARD m.at(0, -1, 0); });
    test([&] { TEST_IGNORE_NODISCARD m.at(0, 0, -1); });
    test([&] { TEST_IGNORE_NODISCARD m.at(5, 3, 3); });
    test([&] { TEST_IGNORE_NODISCARD m.at(3, 7, 3); });
    test([&] { TEST_IGNORE_NODISCARD m.at(3, 3, 9); });
    test([&] { TEST_IGNORE_NODISCARD m.at(5, 7, 9); });
  }
  {
    std::mdspan m(data, std::dextents<unsigned, 3>(5, 7, 9));
    test([&] { TEST_IGNORE_NODISCARD m.at(-1, -1, -1); });
    test([&] { TEST_IGNORE_NODISCARD m.at(-1, 0, 0); });
    test([&] { TEST_IGNORE_NODISCARD m.at(0, -1, 0); });
    test([&] { TEST_IGNORE_NODISCARD m.at(0, 0, -1); });
    test([&] { TEST_IGNORE_NODISCARD m.at(5, 3, 3); });
    test([&] { TEST_IGNORE_NODISCARD m.at(3, 7, 3); });
    test([&] { TEST_IGNORE_NODISCARD m.at(3, 3, 9); });
    test([&] { TEST_IGNORE_NODISCARD m.at(5, 7, 9); });
  }
  return 0;
}
