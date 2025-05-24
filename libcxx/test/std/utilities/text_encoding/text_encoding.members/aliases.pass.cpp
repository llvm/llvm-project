//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-localization

// class text_encoding

// text_encoding text_encoding::environment(); 

// Concerns:
// 1. Verify that text_encoding::aliases_view satisfies ranges::forward_range, copyable, view, 
//    ranges::random_access_range and ranges::borrowed_range

#include <concepts>
#include <ranges>
#include <text_encoding>
#include <type_traits>

#include "platform_support.h" 
#include "test_macros.h"
#include "test_text_encoding.h"

int main(){
  static_assert(std::ranges::forward_range<std::text_encoding::aliases_view>);
  static_assert(std::copyable<std::text_encoding::aliases_view>);
  static_assert(std::ranges::view<std::text_encoding::aliases_view>);
  static_assert(std::ranges::random_access_range<std::text_encoding::aliases_view>);
  static_assert(std::ranges::borrowed_range<std::text_encoding::aliases_view>);
}
