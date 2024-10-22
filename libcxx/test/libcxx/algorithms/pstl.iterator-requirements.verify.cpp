//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// REQUIRES: stdlib=libc++
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>
// <numeric>

// Make sure that all PSTL algorithms contain checks for iterator requirements.
// This is not a requirement from the Standard, but we strive to catch misuse in
// the PSTL both because we can, and because iterator category mistakes in the
// PSTL can lead to subtle bugs.

// Ignore spurious errors after the initial static_assert failure.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

// We only diagnose this in C++20 and above because we implement the checks with concepts.
// UNSUPPORTED: c++17

#include <algorithm>
#include <cstddef>
#include <execution>
#include <numeric>

#include "test_iterators.h"

using non_forward_iterator = cpp17_input_iterator<int*>;
struct non_output_iterator : forward_iterator<int*> {
  constexpr int const& operator*() const; // prevent it from being an output iterator
};

void f(non_forward_iterator non_fwd, non_output_iterator non_output, std::execution::sequenced_policy pol) {
  auto pred     = [](auto&&...) -> bool { return true; };
  auto func     = [](auto&&...) -> int { return 1; };
  int* it       = nullptr;
  int* out      = nullptr;
  std::size_t n = 0;
  int val       = 0;

  {
    (void)std::any_of(pol, non_fwd, non_fwd, pred);  // expected-error@*:* {{static assertion failed: any_of}}
    (void)std::all_of(pol, non_fwd, non_fwd, pred);  // expected-error@*:* {{static assertion failed: all_of}}
    (void)std::none_of(pol, non_fwd, non_fwd, pred); // expected-error@*:* {{static assertion failed: none_of}}
  }

  {
    (void)std::copy(pol, non_fwd, non_fwd, it); // expected-error@*:* {{static assertion failed: copy}}
    (void)std::copy(pol, it, it, non_fwd);      // expected-error@*:* {{static assertion failed: copy}}
    (void)std::copy(pol, it, it, non_output);   // expected-error@*:* {{static assertion failed: copy}}
  }
  {
    (void)std::copy_n(pol, non_fwd, n, it);    // expected-error@*:* {{static assertion failed: copy_n}}
    (void)std::copy_n(pol, it, n, non_fwd);    // expected-error@*:* {{static assertion failed: copy_n}}
    (void)std::copy_n(pol, it, n, non_output); // expected-error@*:* {{static assertion failed: copy_n}}
  }

  {
    (void)std::count(pol, non_fwd, non_fwd, val);     // expected-error@*:* {{static assertion failed: count}}
    (void)std::count_if(pol, non_fwd, non_fwd, pred); // expected-error@*:* {{static assertion failed: count_if}}
  }

  {
    (void)std::equal(pol, non_fwd, non_fwd, it);       // expected-error@*:* {{static assertion failed: equal}}
    (void)std::equal(pol, it, it, non_fwd);            // expected-error@*:* {{static assertion failed: equal}}
    (void)std::equal(pol, non_fwd, non_fwd, it, pred); // expected-error@*:* {{static assertion failed: equal}}
    (void)std::equal(pol, it, it, non_fwd, pred);      // expected-error@*:* {{static assertion failed: equal}}

    (void)std::equal(pol, non_fwd, non_fwd, it, it);       // expected-error@*:* {{static assertion failed: equal}}
    (void)std::equal(pol, it, it, non_fwd, non_fwd);       // expected-error@*:* {{static assertion failed: equal}}
    (void)std::equal(pol, non_fwd, non_fwd, it, it, pred); // expected-error@*:* {{static assertion failed: equal}}
    (void)std::equal(pol, it, it, non_fwd, non_fwd, pred); // expected-error@*:* {{static assertion failed: equal}}
  }

  {
    (void)std::fill(pol, non_fwd, non_fwd, val); // expected-error@*:* {{static assertion failed: fill}}
    (void)std::fill_n(pol, non_fwd, n, val);     // expected-error@*:* {{static assertion failed: fill_n}}
  }

  {
    (void)std::find(pol, non_fwd, non_fwd, val);         // expected-error@*:* {{static assertion failed: find}}
    (void)std::find_if(pol, non_fwd, non_fwd, pred);     // expected-error@*:* {{static assertion failed: find_if}}
    (void)std::find_if_not(pol, non_fwd, non_fwd, pred); // expected-error@*:* {{static assertion failed: find_if_not}}
  }

  {
    (void)std::for_each(pol, non_fwd, non_fwd, func); // expected-error@*:* {{static assertion failed: for_each}}
    (void)std::for_each_n(pol, non_fwd, n, func);     // expected-error@*:* {{static assertion failed: for_each_n}}
  }

  {
    (void)std::generate(pol, non_fwd, non_fwd, func); // expected-error@*:* {{static assertion failed: generate}}
    (void)std::generate_n(pol, non_fwd, n, func);     // expected-error@*:* {{static assertion failed: generate_n}}
  }

  {
    (void)std::is_partitioned(
        pol, non_fwd, non_fwd, pred); // expected-error@*:* {{static assertion failed: is_partitioned}}
  }

  {
    (void)std::merge(pol, non_fwd, non_fwd, it, it, out); // expected-error@*:* {{static assertion failed: merge}}
    (void)std::merge(pol, it, it, non_fwd, non_fwd, out); // expected-error@*:* {{static assertion failed: merge}}
    (void)std::merge(pol, it, it, it, it, non_output);    // expected-error@*:* {{static assertion failed: merge}}

    (void)std::merge(pol, non_fwd, non_fwd, it, it, out, pred); // expected-error@*:* {{static assertion failed: merge}}
    (void)std::merge(pol, it, it, non_fwd, non_fwd, out, pred); // expected-error@*:* {{static assertion failed: merge}}
    (void)std::merge(pol, it, it, it, it, non_output, pred);    // expected-error@*:* {{static assertion failed: merge}}
  }

  {
    (void)std::move(pol, non_fwd, non_fwd, out); // expected-error@*:* {{static assertion failed: move}}
    (void)std::move(pol, it, it, non_fwd);       // expected-error@*:* {{static assertion failed: move}}
    (void)std::move(pol, it, it, non_output);    // expected-error@*:* {{static assertion failed: move}}
  }

  {
    (void)std::replace_if(
        pol, non_fwd, non_fwd, pred, val);               // expected-error@*:* {{static assertion failed: replace_if}}
    (void)std::replace(pol, non_fwd, non_fwd, val, val); // expected-error@*:* {{static assertion failed: replace}}

    (void)std::replace_copy_if(
        pol, non_fwd, non_fwd, out, pred, val); // expected-error@*:* {{static assertion failed: replace_copy_if}}
    (void)std::replace_copy_if(
        pol, it, it, non_fwd, pred, val); // expected-error@*:* {{static assertion failed: replace_copy_if}}
    (void)std::replace_copy_if(
        pol, it, it, non_output, pred, val); // expected-error@*:* {{static assertion failed: replace_copy_if}}

    (void)std::replace_copy(
        pol, non_fwd, non_fwd, out, val, val); // expected-error@*:* {{static assertion failed: replace_copy}}
    (void)std::replace_copy(
        pol, it, it, non_fwd, val, val); // expected-error@*:* {{static assertion failed: replace_copy}}
    (void)std::replace_copy(
        pol, it, it, non_output, val, val); // expected-error@*:* {{static assertion failed: replace_copy}}
  }

  {
    (void)std::rotate_copy(
        pol, non_fwd, non_fwd, non_fwd, out);            // expected-error@*:* {{static assertion failed: rotate_copy}}
    (void)std::rotate_copy(pol, it, it, it, non_fwd);    // expected-error@*:* {{static assertion failed: rotate_copy}}
    (void)std::rotate_copy(pol, it, it, it, non_output); // expected-error@*:* {{static assertion failed: rotate_copy}}
  }

  {
    (void)std::sort(pol, non_fwd, non_fwd);       // expected-error@*:* {{static assertion failed: sort}}
    (void)std::sort(pol, non_fwd, non_fwd, pred); // expected-error@*:* {{static assertion failed: sort}}
  }

  {
    (void)std::stable_sort(pol, non_fwd, non_fwd);       // expected-error@*:* {{static assertion failed: stable_sort}}
    (void)std::stable_sort(pol, non_fwd, non_fwd, pred); // expected-error@*:* {{static assertion failed: stable_sort}}
  }

  {
    (void)std::transform(pol, non_fwd, non_fwd, out, func); // expected-error@*:* {{static assertion failed: transform}}
    (void)std::transform(pol, it, it, non_fwd, func);       // expected-error@*:* {{static assertion failed: transform}}
    (void)std::transform(pol, it, it, non_output, func);    // expected-error@*:* {{static assertion failed: transform}}

    (void)std::transform(
        pol, non_fwd, non_fwd, it, out, func);             // expected-error@*:* {{static assertion failed: transform}}
    (void)std::transform(pol, it, it, non_fwd, out, func); // expected-error@*:* {{static assertion failed: transform}}
    (void)std::transform(pol, it, it, it, non_fwd, func);  // expected-error@*:* {{static assertion failed: transform}}
    (void)std::transform(
        pol, it, it, it, non_output, func); // expected-error@*:* {{static assertion failed: transform}}
  }

  {
    (void)std::reduce(pol, non_fwd, non_fwd);            // expected-error@*:* {{static assertion failed: reduce}}
    (void)std::reduce(pol, non_fwd, non_fwd, val);       // expected-error@*:* {{static assertion failed: reduce}}
    (void)std::reduce(pol, non_fwd, non_fwd, val, func); // expected-error@*:* {{static assertion failed: reduce}}
  }

  {
    (void)std::transform_reduce(
        pol, non_fwd, non_fwd, it, val); // expected-error@*:* {{static assertion failed: transform_reduce}}
    (void)std::transform_reduce(
        pol, it, it, non_fwd, val); // expected-error@*:* {{static assertion failed: transform_reduce}}

    (void)std::transform_reduce(
        pol, non_fwd, non_fwd, it, val, func, func); // expected-error@*:* {{static assertion failed: transform_reduce}}
    (void)std::transform_reduce(
        pol, it, it, non_fwd, val, func, func); // expected-error@*:* {{static assertion failed: transform_reduce}}

    (void)std::transform_reduce(
        pol, non_fwd, non_fwd, val, func, func); // expected-error@*:* {{static assertion failed: transform_reduce}}
  }
}
