//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class It, class T>
// concept output_iterator;

#include <iterator>

#include <cstddef>
#include "test_iterators.h"

struct T { };
struct DerivedFromT : T { };

static_assert( std::output_iterator<cpp17_output_iterator<int*>, int>);
static_assert( std::output_iterator<cpp17_output_iterator<int*>, short>);
static_assert( std::output_iterator<cpp17_output_iterator<int*>, long>);
static_assert( std::output_iterator<cpp17_output_iterator<T*>, T>);
static_assert(!std::output_iterator<cpp17_output_iterator<T const*>, T>);
static_assert( std::output_iterator<cpp17_output_iterator<T*>, T const>);
static_assert( std::output_iterator<cpp17_output_iterator<T*>, DerivedFromT>);
static_assert(!std::output_iterator<cpp17_output_iterator<DerivedFromT*>, T>);

// Not satisfied when the iterator is not an input_or_output_iterator
static_assert(!std::output_iterator<void, int>);
static_assert(!std::output_iterator<void (*)(), int>);
static_assert(!std::output_iterator<int&, int>);
static_assert(!std::output_iterator<T, int>);

// Not satisfied when we can't assign a T to the result of *it++
struct WrongPostIncrement {
    using difference_type = std::ptrdiff_t;
    T const* operator++(int);
    WrongPostIncrement& operator++();
    T& operator*();
};
static_assert( std::input_or_output_iterator<WrongPostIncrement>);
static_assert( std::indirectly_writable<WrongPostIncrement, T>);
static_assert(!std::output_iterator<WrongPostIncrement, T>);

// Not satisfied when we can't assign a T to the result of *it (i.e. not indirectly_writable)
struct NotIndirectlyWritable {
    using difference_type = std::ptrdiff_t;
    T* operator++(int);
    NotIndirectlyWritable& operator++();
    T const& operator*(); // const so we can't write to it
};
static_assert( std::input_or_output_iterator<NotIndirectlyWritable>);
static_assert(!std::indirectly_writable<NotIndirectlyWritable, T>);
static_assert(!std::output_iterator<NotIndirectlyWritable, T>);
