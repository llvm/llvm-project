//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <input_range _View> requires view<_View>

#include "__ranges/stride_view.h"
#include "almost_satisfies_types.h"
#include "test.h"
#include "test_iterators.h"
#include "test_range.h"

// Ensure that the InputRangeNotIndirectlyReadable is a valid range.
static_assert(std::ranges::range<InputRangeNotIndirectlyReadable>); // expected-no-diagnostics
// Ensure that the InputRangeNotIndirectlyReadable's is not an input range ...
static_assert(
    !std::ranges::input_range<std::ranges::iterator_t<InputRangeNotIndirectlyReadable>>); // expected-no-diagnostics
// Because CanStrideView requires that the range/view type be default constructible, let's double check that ...
static_assert(std::is_constructible_v<InputRangeNotIndirectlyReadable>); // expected-no-diagnostics
// And now, finally, let's make sure that we cannot stride over a range whose iterator is not an input iterator ...
static_assert(!CanStrideView<InputRangeNotIndirectlyReadable, 1>); // expected-no-diagnostics

// Ensure that a range that is not a view cannot be the subject of a stride_view.
static_assert(std::ranges::range<non_view_range>);       // expected-no-diagnostics
static_assert(std::ranges::input_range<non_view_range>); // expected-no-diagnostics
static_assert(std::movable<non_view_range>);             // expected-no-diagnostics
static_assert(!std::ranges::view<non_view_range>);       // expected-no-diagnostics
static_assert(!CanStrideView<non_view_range, 1>);        // expected-no-diagnostics

// And now, let's satisfy all the prerequisites and make sure that we can stride over a range (that is an input range and is a view!)
static_assert(std::ranges::range<test_view<cpp17_input_iterator>>);       // expected-no-diagnostics
static_assert(std::ranges::input_range<test_view<cpp17_input_iterator>>); // expected-no-diagnostics
static_assert(std::ranges::view<test_view<cpp17_input_iterator>>);        // expected-no-diagnostics
static_assert(CanStrideView<test_view<cpp17_input_iterator>, 1>);         // expected-no-diagnostics
