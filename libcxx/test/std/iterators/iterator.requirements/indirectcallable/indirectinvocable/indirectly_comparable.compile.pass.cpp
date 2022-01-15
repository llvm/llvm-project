//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class I1, class I2, class R, class P1, class P2>
// concept indirectly_­comparable;

#include <functional>
#include <iterator>
#include <type_traits>

struct Deref {
    int operator()(int*) const;
};

static_assert(!std::indirectly_comparable<int, int, std::less<int>>);  // not dereferenceable
static_assert(!std::indirectly_comparable<int*, int*, int>);  // not a predicate
static_assert( std::indirectly_comparable<int*, int*, std::less<int>>);
static_assert(!std::indirectly_comparable<int**, int*, std::less<int>>);
static_assert( std::indirectly_comparable<int**, int*, std::less<int>, Deref>);
static_assert(!std::indirectly_comparable<int**, int*, std::less<int>, Deref, Deref>);
static_assert(!std::indirectly_comparable<int**, int*, std::less<int>, std::identity, Deref>);
static_assert( std::indirectly_comparable<int*, int**, std::less<int>, std::identity, Deref>);

template<class F>
  requires std::indirectly_comparable<int*, char*, F>
           && true // This true is an additional atomic constraint as a tie breaker
constexpr bool subsumes(F) { return true; }

template<class F>
  requires std::indirect_binary_predicate<F, std::projected<int*, std::identity>, std::projected<char*, std::identity>>
void subsumes(F);

template<class F>
  requires std::indirect_binary_predicate<F, std::projected<int*, std::identity>, std::projected<char*, std::identity>>
           && true // This true is an additional atomic constraint as a tie breaker
constexpr bool is_subsumed(F) { return true; }

template<class F>
  requires std::indirectly_comparable<int*, char*, F>
void is_subsumed(F);

static_assert(subsumes(std::less<int>()));
static_assert(is_subsumed(std::less<int>()));
