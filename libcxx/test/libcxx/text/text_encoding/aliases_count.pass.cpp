//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=40000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=1000000000

// We implement 882 aliases, test to make sure that number matches with the total alias count.
#include <cassert>
#include <ranges>
#include <text_encoding>

#include "test_text_encoding.h"

constexpr bool test() {
  long long sum = 0;
  for (auto& enc : unique_encoding_data) {
    std::text_encoding te{std::text_encoding::id(enc.mib)};

    sum += std::ranges::size(te.aliases());
  }

  // +2 reserved as sentinels for id::unknown and id::other
  // Meaning, our offset table actually contains 884 entries.
  assert(sum == std::text_encoding::__num_aliases - 2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
