//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <memory>

// template <nothrow-forward-iterator ForwardIterator, nothrow-sentinel-for<ForwardIterator> Sentinel, class T>
//   requires constructible_from<iter_value_t<ForwardIterator>, const T&>
// ForwardIterator ranges::uninitialized_fill(ForwardIterator first, Sentinel last, const T& x);
//
// template <nothrow-forward-range ForwardRange, class T>
//   requires constructible_from<range_value_t<ForwardRange>, const T&>
// borrowed_iterator_t<ForwardRange> ranges::uninitialized_fill(ForwardRange&& range, const T& x);

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>

#include "../buffer.h"
#include "../counted.h"
#include "test_macros.h"
#include "test_iterators.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_fill)>);

struct NotConvertibleFromInt {};
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_fill), NotConvertibleFromInt*,
                                   NotConvertibleFromInt*, int>);

int main(int, char**) {
  constexpr int value = 42;
  Counted x(value);
  Counted::reset();
  auto pred = [](const Counted& e) { return e.value == value; };

  // An empty range -- no default constructors should be invoked.
  {
    Buffer<Counted, 1> buf;

    std::ranges::uninitialized_fill(buf.begin(), buf.begin(), x);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);

    std::ranges::uninitialized_fill(std::ranges::empty_view<Counted>(), x);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);

    forward_iterator<Counted*> it(buf.begin());
    auto range = std::ranges::subrange(it, sentinel_wrapper<forward_iterator<Counted*>>(it));
    std::ranges::uninitialized_fill(range.begin(), range.end(), x);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);
    Counted::reset();

    std::ranges::uninitialized_fill(range, x);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);
    Counted::reset();
  }

  // A range containing several objects, (iter, sentinel) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    std::ranges::uninitialized_fill(buf.begin(), buf.end(), x);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::all_of(buf.begin(), buf.end(), pred));

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // A range containing several objects, (range) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    auto range = std::ranges::subrange(buf.begin(), buf.end());
    std::ranges::uninitialized_fill(range, x);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::all_of(buf.begin(), buf.end(), pred));

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // Using `counted_iterator`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    std::ranges::uninitialized_fill(std::counted_iterator(buf.begin(), N), std::default_sentinel, x);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::all_of(buf.begin(), buf.begin() + N, pred));

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // Using `views::counted`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    std::ranges::uninitialized_fill(std::views::counted(buf.begin(), N), x);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::all_of(buf.begin(), buf.begin() + N, pred));

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // Using `reverse_view`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    auto range = std::ranges::subrange(buf.begin(), buf.begin() + N);
    std::ranges::uninitialized_fill(std::ranges::reverse_view(range), x);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::all_of(buf.begin(), buf.begin() + N, pred));

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // Any existing values should be overwritten by value constructors.
  {
    constexpr int N = 5;
    int buffer[N] = {value, value, value, value, value};

    std::ranges::uninitialized_fill(buffer, buffer + 1, 0);
    assert(buffer[0] == 0);
    assert(buffer[1] == value);

    std::ranges::uninitialized_fill(buffer, buffer + N, 0);
    assert(buffer[0] == 0);
    assert(buffer[1] == 0);
    assert(buffer[2] == 0);
    assert(buffer[3] == 0);
    assert(buffer[4] == 0);
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid. (iterator, sentinel) overload.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    Counted::throw_on = N; // When constructing the fourth object.
    try {
      std::ranges::uninitialized_fill(buf.begin(), buf.end(), x);
    } catch (...) {
    }
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid. (range) overload.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    Counted::throw_on = N; // When constructing the fourth object.
    try {
      std::ranges::uninitialized_fill(buf, x);
    } catch (...) {
    }
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  // Works with const iterators, (iter, sentinel) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    std::ranges::uninitialized_fill(buf.cbegin(), buf.cend(), x);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::all_of(buf.begin(), buf.end(), pred));

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // Works with const iterators, (range) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    auto range = std::ranges::subrange(buf.cbegin(), buf.cend());
    std::ranges::uninitialized_fill(range, x);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::all_of(buf.begin(), buf.end(), pred));

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  return 0;
}
