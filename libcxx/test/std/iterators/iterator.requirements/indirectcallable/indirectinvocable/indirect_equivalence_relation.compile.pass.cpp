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

// template<class F, class I1, class I2 = I1>
// concept indirect_equivalence_relation;

#include <iterator>
#include <concepts>

#include "indirectly_readable.h"

using It1 = IndirectlyReadable<struct Token1>;
using It2 = IndirectlyReadable<struct Token2>;

template <class I1, class I2>
struct GoodRelation {
    bool operator()(std::iter_value_t<I1>&, std::iter_value_t<I1>&) const;
    bool operator()(std::iter_value_t<I2>&, std::iter_value_t<I2>&) const;
    bool operator()(std::iter_value_t<I1>&, std::iter_value_t<I2>&) const;
    bool operator()(std::iter_value_t<I2>&, std::iter_value_t<I1>&) const;

    bool operator()(std::iter_value_t<I1>&, std::iter_reference_t<I2>) const;
    bool operator()(std::iter_reference_t<I2>, std::iter_value_t<I1>&) const;
    bool operator()(std::iter_reference_t<I2>, std::iter_reference_t<I2>) const;

    bool operator()(std::iter_reference_t<I1>, std::iter_value_t<I2>&) const;
    bool operator()(std::iter_value_t<I2>&, std::iter_reference_t<I1>) const;
    bool operator()(std::iter_reference_t<I1>, std::iter_reference_t<I1>) const;

    bool operator()(std::iter_reference_t<I1>, std::iter_reference_t<I2>) const;
    bool operator()(std::iter_reference_t<I2>, std::iter_reference_t<I1>) const;

    bool operator()(std::iter_common_reference_t<I1>, std::iter_common_reference_t<I1>) const;
    bool operator()(std::iter_common_reference_t<I2>, std::iter_common_reference_t<I2>) const;
    bool operator()(std::iter_common_reference_t<I1>, std::iter_common_reference_t<I2>) const;
    bool operator()(std::iter_common_reference_t<I2>, std::iter_common_reference_t<I1>) const;
};

// Should work when all constraints are satisfied
static_assert(std::indirect_equivalence_relation<GoodRelation<It1, It2>, It1, It2>);
static_assert(std::indirect_equivalence_relation<bool(*)(int, long), int*, long*>);
[[maybe_unused]] auto lambda = [](int i, long j) { return i == j; };
static_assert(std::indirect_equivalence_relation<decltype(lambda), int*, long*>);

// Should fail when either of the iterators is not indirectly_readable
struct NotIndirectlyReadable { };
static_assert(!std::indirect_equivalence_relation<GoodRelation<It1, NotIndirectlyReadable>, It1, NotIndirectlyReadable>);
static_assert(!std::indirect_equivalence_relation<GoodRelation<NotIndirectlyReadable, It2>, NotIndirectlyReadable, It2>);

// Should fail when the function is not copy constructible
struct BadRelation1 {
    BadRelation1(BadRelation1 const&) = delete;
    template <class T, class U> bool operator()(T const&, U const&) const;
};
static_assert(!std::indirect_equivalence_relation<BadRelation1, It1, It2>);

// Should fail when the function can't be called with (iter_value_t&, iter_value_t&)
struct BadRelation2 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_value_t<It1>&, std::iter_value_t<It2>&) const = delete;
};
static_assert(!std::indirect_equivalence_relation<BadRelation2, It1, It2>);

// Should fail when the function can't be called with (iter_value_t&, iter_reference_t)
struct BadRelation3 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_value_t<It1>&, std::iter_reference_t<It2>) const = delete;
};
static_assert(!std::indirect_equivalence_relation<BadRelation3, It1, It2>);

// Should fail when the function can't be called with (iter_reference_t, iter_value_t&)
struct BadRelation4 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_reference_t<It1>, std::iter_value_t<It2>&) const = delete;
};
static_assert(!std::indirect_equivalence_relation<BadRelation4, It1, It2>);

// Should fail when the function can't be called with (iter_reference_t, iter_reference_t)
struct BadRelation5 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_reference_t<It1>, std::iter_reference_t<It2>) const = delete;
};
static_assert(!std::indirect_equivalence_relation<BadRelation5, It1, It2>);

// Should fail when the function can't be called with (iter_common_reference_t, iter_common_reference_t)
struct BadRelation6 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_common_reference_t<It1>, std::iter_common_reference_t<It2>) const = delete;
};
static_assert(!std::indirect_equivalence_relation<BadRelation6, It1, It2>);
