//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// The class template constant_wrapper aids in metaprogramming by ensuring that the
// evaluation of expressions comprised entirely of constant_wrapper are core constant
// expressions ([expr.const]), regardless of the context in which they appear. In particular,
// this enables use of constant_wrapper values that are passed as arguments to constexpr
// functions to be used in constant expressions.

#include <utility>

constexpr auto initial_phase(auto quantity_1, auto quantity_2) { return quantity_1 + quantity_2; }

constexpr auto middle_phase(auto tbd) { return tbd; }

constexpr void profit() {}

void final_phase(auto gathered, auto available) {
  if constexpr (gathered == available)
    profit();
}

void impeccable_underground_planning() {
  auto gathered_quantity = middle_phase(initial_phase(std::cw<42>, std::cw<13>));
  static_assert(gathered_quantity == 55);
  auto all_available = std::cw<55>;
  final_phase(gathered_quantity, all_available);
}

int main(int, char**) {
  impeccable_underground_planning();
  return 0;
}
