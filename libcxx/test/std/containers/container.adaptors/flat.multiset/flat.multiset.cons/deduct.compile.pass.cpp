//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// Test CTAD on cases where deduction should fail.

#include <flat_set>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

struct NotAnAllocator {
  friend bool operator<(NotAnAllocator, NotAnAllocator) { return false; }
};

template <class... Args>
concept CanDeductFlatMultiSet = requires { std::flat_multiset(std::declval<Args>()...); };

static_assert(CanDeductFlatMultiSet<std::vector<int>>);

// cannot deduce Key and T from nothing
static_assert(!CanDeductFlatMultiSet<>);

// cannot deduce Key and T from just (Compare)
static_assert(!CanDeductFlatMultiSet<std::less<int>>);

// cannot deduce Key and T from just (Compare, Allocator)
static_assert(!CanDeductFlatMultiSet<std::less<int>, std::allocator<int>>);

// cannot deduce Key and T from just (Allocator)
static_assert(!CanDeductFlatMultiSet<std::allocator<int>>);

// cannot convert from some arbitrary unrelated type
static_assert(!CanDeductFlatMultiSet<NotAnAllocator>);
