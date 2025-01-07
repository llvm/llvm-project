//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class F, class I1, class I2>
// concept indirect_binary_predicate;

#include <functional>
#include <iterator>
#include <type_traits>

#include "indirectly_readable.h"
#include "test_macros.h"

using It1 = IndirectlyReadable<struct Token1>;
using It2 = IndirectlyReadable<struct Token2>;

template <class I1, class I2>
struct GoodPredicate {
    bool operator()(std::iter_value_t<I1>&, std::iter_value_t<I2>&) const;
    bool operator()(std::iter_value_t<I1>&, std::iter_reference_t<I2>) const;
    bool operator()(std::iter_reference_t<I1>, std::iter_value_t<I2>&) const;
    bool operator()(std::iter_reference_t<I1>, std::iter_reference_t<I2>) const;
    bool operator()(std::iter_common_reference_t<I1>, std::iter_common_reference_t<I2>) const;
};

// Should work when all constraints are satisfied
static_assert(std::indirect_binary_predicate<GoodPredicate<It1, It2>, It1, It2>);
static_assert(std::indirect_binary_predicate<bool(*)(int, float), int*, float*>);
[[maybe_unused]] auto lambda = [](int i, int j) { return i < j; };
static_assert(std::indirect_binary_predicate<decltype(lambda), int*, int*>);

// Should fail when either of the iterators is not indirectly_readable
struct NotIndirectlyReadable { };
static_assert(!std::indirect_binary_predicate<GoodPredicate<It1, NotIndirectlyReadable>, It1, NotIndirectlyReadable>);
static_assert(!std::indirect_binary_predicate<GoodPredicate<NotIndirectlyReadable, It2>, NotIndirectlyReadable, It2>);

// Should fail when the predicate is not copy constructible
struct BadPredicate1 {
    BadPredicate1(BadPredicate1 const&) = delete;
    template <class T, class U> bool operator()(T const&, U const&) const;
};
static_assert(!std::indirect_binary_predicate<BadPredicate1, It1, It2>);

// Should fail when the predicate can't be called with (iter_value_t&, iter_value_t&)
struct BadPredicate2 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_value_t<It1>&, std::iter_value_t<It2>&) const = delete;
};
static_assert(!std::indirect_binary_predicate<BadPredicate2, It1, It2>);

// Should fail when the predicate can't be called with (iter_value_t&, iter_reference_t)
struct BadPredicate3 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_value_t<It1>&, std::iter_reference_t<It2>) const = delete;
};
static_assert(!std::indirect_binary_predicate<BadPredicate3, It1, It2>);

// Should fail when the predicate can't be called with (iter_reference_t, iter_value_t&)
struct BadPredicate4 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_reference_t<It1>, std::iter_value_t<It2>&) const = delete;
};
static_assert(!std::indirect_binary_predicate<BadPredicate4, It1, It2>);

// Should fail when the predicate can't be called with (iter_reference_t, iter_reference_t)
struct BadPredicate5 {
    template <class T, class U> bool operator()(T const&, U const&) const;
    bool operator()(std::iter_reference_t<It1>, std::iter_reference_t<It2>) const = delete;
};
static_assert(!std::indirect_binary_predicate<BadPredicate5, It1, It2>);

// This case was made valid by P2997R1.
struct GoodPredicate6 {
  template <class T, class U>
  bool operator()(T const&, U const&) const;
  bool operator()(std::iter_common_reference_t<It1>, std::iter_common_reference_t<It2>) const = delete;
};
static_assert(std::indirect_binary_predicate<GoodPredicate6, It1, It2>);

// Test ADL-proofing (P2538R1)
#if TEST_STD_VER >= 26 || defined(_LIBCPP_VERSION)
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(std::indirect_binary_predicate<std::less<Holder<Incomplete>*>, Holder<Incomplete>**, Holder<Incomplete>**>);
static_assert(!std::indirect_binary_predicate<Holder<Incomplete>*, Holder<Incomplete>**, Holder<Incomplete>**>);
#endif
