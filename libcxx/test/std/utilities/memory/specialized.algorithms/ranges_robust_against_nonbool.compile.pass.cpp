//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>
//
// Make sure we support iterators that return a BooleanTestable in the ranges:: algorithms
// defined in <memory>.

#include <memory>

#include <ranges>

#include "boolean_testable.h"

using Value    = StrictComparable<int>;
using Iterator = StrictBooleanIterator<Value*>;
using Range    = std::ranges::subrange<Iterator>;

void f(Iterator it, Range in, Range out, std::size_t n, Value const& val) {
  // uninitialized_copy
  {
    std::ranges::uninitialized_copy(in, out);
    std::ranges::uninitialized_copy(it, it, it, it);
  }
  // uninitialized_copy_n
  { std::ranges::uninitialized_copy_n(it, n, it, it); }
  // uninitialized_fill
  {
    std::ranges::uninitialized_fill(it, it, val);
    std::ranges::uninitialized_fill(in, val);
  }
  // uninitialized_fill_n
  { std::ranges::uninitialized_fill_n(it, n, val); }
  // uninitialized_move
  {
    std::ranges::uninitialized_move(it, it, it, it);
    std::ranges::uninitialized_move(in, out);
  }
  // uninitialized_move_n
  { std::ranges::uninitialized_move_n(it, n, it, it); }
  // uninitialized_default_construct
  {
    std::ranges::uninitialized_default_construct(it, it);
    std::ranges::uninitialized_default_construct(in);
  }
  // uninitialized_default_construct_n
  { std::ranges::uninitialized_default_construct_n(it, n); }
  // uninitialized_value_construct
  {
    std::ranges::uninitialized_value_construct(it, it);
    std::ranges::uninitialized_value_construct(in);
  }
  // uninitialized_value_construct_n
  { std::ranges::uninitialized_value_construct_n(it, n); }
  // destroy
  {
    std::ranges::destroy(it, it);
    std::ranges::destroy(in);
  }
  // destroy_n
  { std::ranges::destroy_n(it, n); }
}
