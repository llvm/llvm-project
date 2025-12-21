//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <span>

// Check that functions are marked [[nodiscard]]

#include <array>
#include <mdspan>
#include <span>

void test() {
  // mdspan<>

  std::array<int, 4> data;
  std::mdspan<int, std::extents<std::size_t, 2, 2>> mdsp{data.data(), 2, 2};

  mdsp[0, 1]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::array arr{0, 1};
  mdsp[arr]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::span sp{arr};
  mdsp[sp]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mdsp.rank();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.rank_dynamic();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.static_extent(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.extent(0);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mdsp.size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mdsp.extents();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.data_handle(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.mapping();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.accessor();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mdsp.is_always_unique(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.is_always_exhaustive(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.is_always_strided(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.is_unique();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.is_exhaustive(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.is_strided();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mdsp.stride(0);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Helpers

  std::extents<int, 1, 2> ex;
  ex.rank();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ex.rank_dynamic();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ex.static_extent(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ex.extent(0);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::dextents<int, 2> dex;
  dex.rank();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  dex.rank_dynamic();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  dex.static_extent(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  dex.extent(0);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
