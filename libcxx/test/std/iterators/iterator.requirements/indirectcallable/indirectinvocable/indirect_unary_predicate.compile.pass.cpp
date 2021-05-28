//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// template<class F, class I>
// concept indirect_unary_predicate;

#include <iterator>
#include <type_traits>

#include "indirectly_readable.h"

using It = IndirectlyReadable<struct Token>;

template <class I>
struct GoodPredicate {
    bool operator()(std::iter_reference_t<I>) const;
    bool operator()(std::iter_value_t<I>&) const;
    bool operator()(std::iter_common_reference_t<I>) const;
};

// Should work when all constraints are satisfied
static_assert(std::indirect_unary_predicate<GoodPredicate<It>, It>);
static_assert(std::indirect_unary_predicate<bool(*)(int), int*>);
[[maybe_unused]] auto lambda = [](int i) { return i % 2 == 0; };
static_assert(std::indirect_unary_predicate<decltype(lambda), int*>);

// Should fail when the iterator is not indirectly_readable
struct NotIndirectlyReadable { };
static_assert(!std::indirect_unary_predicate<GoodPredicate<NotIndirectlyReadable>, NotIndirectlyReadable>);

// Should fail when the predicate is not copy constructible
struct BadPredicate1 {
    BadPredicate1(BadPredicate1 const&) = delete;
    template <class T> bool operator()(T const&) const;
};
static_assert(!std::indirect_unary_predicate<BadPredicate1, It>);

// Should fail when the predicate can't be called with std::iter_value_t<It>&
struct BadPredicate2 {
    template <class T> bool operator()(T const&) const;
    bool operator()(std::iter_value_t<It>&) const = delete;
};
static_assert(!std::indirect_unary_predicate<BadPredicate2, It>);

// Should fail when the predicate can't be called with std::iter_reference_t<It>
struct BadPredicate3 {
    template <class T> bool operator()(T const&) const;
    bool operator()(std::iter_reference_t<It>) const = delete;
};
static_assert(!std::indirect_unary_predicate<BadPredicate3, It>);

// Should fail when the predicate can't be called with std::iter_common_reference_t<It>
struct BadPredicate4 {
    template <class T> bool operator()(T const&) const;
    bool operator()(std::iter_common_reference_t<It>) const = delete;
};
static_assert(!std::indirect_unary_predicate<BadPredicate4, It>);
