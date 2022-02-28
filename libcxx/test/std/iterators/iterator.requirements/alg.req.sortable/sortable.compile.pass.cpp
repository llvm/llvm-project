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

using CompInt = bool(*)(int, int);
using CompDefault = std::ranges::less;

using AllConstraintsSatisfied = int*;
static_assert( std::permutable<AllConstraintsSatisfied>);
static_assert( std::indirect_strict_weak_order<CompDefault, AllConstraintsSatisfied>);
static_assert( std::sortable<AllConstraintsSatisfied>);
static_assert( std::indirect_strict_weak_order<CompInt, AllConstraintsSatisfied>);
static_assert( std::sortable<AllConstraintsSatisfied, CompInt>);

struct Foo {};
using Proj = int(*)(Foo);
static_assert( std::permutable<Foo*>);
static_assert(!std::indirect_strict_weak_order<CompDefault, Foo*>);
static_assert( std::indirect_strict_weak_order<CompDefault, std::projected<Foo*, Proj>>);
static_assert(!std::sortable<Foo*, CompDefault>);
static_assert( std::sortable<Foo*, CompDefault, Proj>);
static_assert(!std::indirect_strict_weak_order<CompInt, Foo*>);
static_assert( std::indirect_strict_weak_order<CompInt, std::projected<Foo*, Proj>>);
static_assert(!std::sortable<Foo*, CompInt>);
static_assert( std::sortable<Foo*, CompInt, Proj>);

using NotPermutable = const int*;
static_assert(!std::permutable<NotPermutable>);
static_assert( std::indirect_strict_weak_order<CompInt, NotPermutable>);
static_assert(!std::sortable<NotPermutable, CompInt>);

struct Empty {};
using NoIndirectStrictWeakOrder = Empty*;
static_assert( std::permutable<NoIndirectStrictWeakOrder>);
static_assert(!std::indirect_strict_weak_order<CompInt, NoIndirectStrictWeakOrder>);
static_assert(!std::sortable<NoIndirectStrictWeakOrder, CompInt>);
