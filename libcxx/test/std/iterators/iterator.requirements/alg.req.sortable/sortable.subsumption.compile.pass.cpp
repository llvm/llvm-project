//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class I, class R = ranges::less, class P = identity>
//   concept sortable = see below;                            // since C++20

#include <iterator>

#include <functional>

template <class I, class R, class P> void test_subsumption() requires std::permutable<I>;

template <class I, class R, class P> void test_subsumption()
    requires std::indirect_strict_weak_order<R, std::projected<I, P>>;

template <class I, class R, class P> constexpr bool test_subsumption() requires std::sortable<I, R, P> { return true; }

static_assert(test_subsumption<int*, std::ranges::less, std::identity>());
