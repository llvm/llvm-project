//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory>

// unique_ptr

// The following constructors should not be selected by class template argument
// deduction:
//
// explicit unique_ptr(pointer p)
// unique_ptr(pointer p, const D& d) noexcept
// unique_ptr(pointer p, remove_reference_t<D>&& d) noexcept

#include <memory>

#include "deduction_guides_sfinae_checks.h"

struct Deleter {
  void operator()(int* p) const { delete p; }
};

int main(int, char**) {
  // Cannot deduce from (ptr).
  static_assert(SFINAEs_away<std::unique_ptr, int*>);
  // Cannot deduce from (array).
  static_assert(SFINAEs_away<std::unique_ptr, int[]>);
  // Cannot deduce from (ptr, Deleter&&).
  static_assert(SFINAEs_away<std::unique_ptr, int*, Deleter&&>);
  // Cannot deduce from (array, Deleter&&).
  static_assert(SFINAEs_away<std::unique_ptr, int[], Deleter&&>);
  // Cannot deduce from (ptr, const Deleter&).
  static_assert(SFINAEs_away<std::unique_ptr, int*, const Deleter&>);
  // Cannot deduce from (array, const Deleter&).
  static_assert(SFINAEs_away<std::unique_ptr, int[], const Deleter&>);

  return 0;
}
