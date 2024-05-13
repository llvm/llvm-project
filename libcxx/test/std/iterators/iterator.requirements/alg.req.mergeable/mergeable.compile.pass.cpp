//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class I1, class I2, class Out,
//     class R = ranges::less, class P1 = identity, class P2 = identity>
//   concept mergeable = see below;                           // since C++20

#include <iterator>

#include <functional>

#include "test_iterators.h"
#include "test_macros.h"

using CompDefault = std::ranges::less;
using CompInt = bool(*)(int, int);
using ProjDefault = std::identity;

using Input = cpp20_input_iterator<int*>;
static_assert( std::input_iterator<Input>);
using InputLong = cpp20_input_iterator<long*>;
static_assert( std::input_iterator<InputLong>);

using Output = cpp17_output_iterator<int*>;
static_assert( std::weakly_incrementable<Output>);

static_assert( std::indirectly_copyable<Input, Output>);
static_assert( std::indirectly_copyable<InputLong, Output>);
static_assert( std::indirect_strict_weak_order<CompDefault, Input, Input>);
static_assert( std::indirect_strict_weak_order<CompInt, Input, Input>);
static_assert( std::indirect_strict_weak_order<CompDefault, Input, InputLong>);
static_assert( std::indirect_strict_weak_order<CompInt, Input, InputLong>);

// All requirements satisfied.
static_assert( std::mergeable<Input, Input, Output>);
static_assert( std::mergeable<Input, Input, Output, CompInt>);
static_assert( std::mergeable<Input, Input, Output, CompInt, ProjDefault>);

// Non-default projections.
struct Foo {};
using ProjFooToInt = int(*)(Foo);
using ProjFooToLong = long(*)(Foo);
static_assert( std::indirect_strict_weak_order<CompDefault,
    std::projected<Foo*, ProjFooToInt>, std::projected<Foo*, ProjFooToLong>>);
static_assert( std::mergeable<Foo*, Foo*, Foo*, CompDefault, ProjFooToInt, ProjFooToLong>);
static_assert( std::indirect_strict_weak_order<CompInt,
    std::projected<Foo*, ProjFooToInt>, std::projected<Foo*, ProjFooToLong>>);
static_assert( std::mergeable<Foo*, Foo*, Foo*, CompInt, ProjFooToInt, ProjFooToLong>);

// I1 or I2 is not an input iterator.
static_assert(!std::input_iterator<Output>);
static_assert(!std::mergeable<Output, Input, Output>);
static_assert(!std::mergeable<Input, Output, Output>);

// O is not weakly incrementable.
struct NotWeaklyIncrementable {
  int& operator*() const;
};

static_assert(!std::weakly_incrementable<NotWeaklyIncrementable>);
static_assert( std::indirectly_copyable<Input, NotWeaklyIncrementable>);
static_assert( std::indirect_strict_weak_order<CompDefault, Input, Input>);
static_assert(!std::mergeable<Input, Input, NotWeaklyIncrementable>);

// I1 or I2 is not indirectly copyable into O.
struct AssignableOnlyFromInt {
  AssignableOnlyFromInt& operator=(int);
  template <class T>
  AssignableOnlyFromInt& operator=(T) = delete;
};
using OutputOnlyInt = cpp17_output_iterator<AssignableOnlyFromInt*>;
static_assert( std::weakly_incrementable<OutputOnlyInt>);

static_assert( std::indirectly_copyable<Input, OutputOnlyInt>);
static_assert(!std::indirectly_copyable<InputLong, OutputOnlyInt>);
static_assert( std::indirect_strict_weak_order<CompDefault, Input, InputLong>);
static_assert( std::mergeable<Input, Input, OutputOnlyInt>);
static_assert(!std::mergeable<Input, InputLong, OutputOnlyInt>);
static_assert(!std::mergeable<InputLong, Input, OutputOnlyInt>);

// No indirect strict weak order between I1 and I2 (bad comparison functor).
using GoodComp = bool(*)(int, int);
static_assert( std::indirect_strict_weak_order<GoodComp, Input, Input>);
static_assert( std::mergeable<Input, Input, Output, GoodComp>);
using BadComp = bool(*)(int*, int*);
static_assert(!std::indirect_strict_weak_order<BadComp, Input, Input>);
static_assert(!std::mergeable<Input, Input, Output, BadComp>);

// No indirect strict weak order between I1 and I2 (bad projection).
using ToInt = int(*)(int);
using ToPtr = int*(*)(int);
static_assert( std::mergeable<Input, Input, Output, GoodComp, std::identity, std::identity>);
static_assert( std::mergeable<Input, Input, Output, GoodComp, ToInt, ToInt>);
static_assert(!std::mergeable<Input, Input, Output, GoodComp, ToPtr, ToInt>);
static_assert(!std::mergeable<Input, Input, Output, GoodComp, ToInt, ToPtr>);
static_assert(!std::mergeable<Input, Input, Output, bool(*)(int*, int), ToPtr, ToInt>);
static_assert(!std::mergeable<Input, Input, Output, bool(*)(int, int*), ToInt, ToPtr>);

// A projection that only supports non-const references and has a non-const `operator()` still has to work.
struct ProjectionOnlyMutable {
  int operator()(int&);
  int operator()(int&&) const = delete;
};
static_assert( std::mergeable<Input, Input, Output, CompDefault, ProjectionOnlyMutable, ProjectionOnlyMutable>);

// The output is weakly incrementable but not an output iterator.
struct WeaklyIncrementable {
  using value_type = int;
  using difference_type = int;

  int& operator*() const;
  WeaklyIncrementable& operator++();
  // `output_iterator` requires `i++` to return an iterator,
  // while `weakly_incrementable` requires only that `i++` be well-formed.
  void operator++(int);
};
static_assert( std::weakly_incrementable<WeaklyIncrementable>);
static_assert( std::indirectly_copyable<int*, WeaklyIncrementable>);
static_assert(!std::output_iterator<WeaklyIncrementable, int>);
static_assert( std::mergeable<Input, Input, WeaklyIncrementable>);
