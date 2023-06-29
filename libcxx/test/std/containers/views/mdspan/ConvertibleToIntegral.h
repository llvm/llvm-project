//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_LAYOUT_RIGHT_CONVERTIBLE_TO_INTEGRAL_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_LAYOUT_RIGHT_CONVERTIBLE_TO_INTEGRAL_H

struct IntType {
  int val;
  constexpr IntType() = default;
  constexpr IntType(int v) noexcept : val(v){};

  constexpr bool operator==(const IntType& rhs) const { return val == rhs.val; }
  constexpr operator int() const noexcept { return val; }
  constexpr operator unsigned char() const { return val; }
  constexpr operator char() const noexcept { return val; }
};

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_LAYOUT_RIGHT_CONVERTIBLE_TO_INTEGRAL_H
