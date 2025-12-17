//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>
//
// template<input_iterator InputIterator, nothrow-forward-iterator OutputIterator, nothrow-sentinel-for<OutputIterator> Sentinel>
// requires constructible_from<iter_value_t<OutputIterator>, iter_rvalue_reference_t<InputIterator>>
// ranges::uninitialized_move_n_result<InputIterator, OutputIterator>
// ranges::uninitialized_move_n(InputIterator ifirst, iter_difference_t<InputIterator> n, OutputIterator ofirst, Sentinel olast); // since C++20


#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>

#include "../buffer.h"
#include "../counted.h"
#include "../overload_compare_iterator.h"
#include "MoveOnly.h"
#include "test_macros.h"
#include "test_iterators.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_move)>);

static_assert(std::is_invocable_v<decltype(std::ranges::uninitialized_move), int*, int*, long*, long*>);
struct NotConvertibleFromInt {};
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_move), int*, int*, NotConvertibleFromInt*,
                                   NotConvertibleFromInt*>);

int main(int, char**) {
  // An empty range -- no default constructors should be invoked.
  {
    Counted in[] = {Counted()};
    Buffer<Counted, 1> out;
    Counted::reset();

    {
      auto result = std::ranges::uninitialized_move(in, in, out.begin(), out.end());
      assert(Counted::current_objects == 0);
      assert(Counted::total_objects == 0);
      assert(Counted::total_copies == 0);
      assert(result.in == in);
      assert(result.out == out.begin());
    }

    {
      std::ranges::empty_view<Counted> view;
      auto result = std::ranges::uninitialized_move(view, out);
      assert(Counted::current_objects == 0);
      assert(Counted::total_objects == 0);
      assert(Counted::total_copies == 0);
      assert(result.in == view.begin());
      assert(result.out == out.begin());
    }

    {
      forward_iterator<Counted*> it(in);
      std::ranges::subrange range(it, sentinel_wrapper<forward_iterator<Counted*>>(it));

      auto result = std::ranges::uninitialized_move(range.begin(), range.end(), out.begin(), out.end());
      assert(Counted::current_objects == 0);
      assert(Counted::total_objects == 0);
      assert(Counted::total_copies == 0);
      assert(result.in == it);
      assert(result.out == out.begin());
    }

    {
      forward_iterator<Counted*> it(in);
      std::ranges::subrange range(it, sentinel_wrapper<forward_iterator<Counted*>>(it));

      auto result = std::ranges::uninitialized_move(range, out);
      assert(Counted::current_objects == 0);
      assert(Counted::total_objects == 0);
      assert(Counted::total_copies == 0);
      assert(result.in == it);
      assert(result.out == out.begin());
    }
    Counted::reset();
  }

  // A range containing several objects, (iter, sentinel) overload.
  {
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, N> out;
    Counted::reset();

    auto result = std::ranges::uninitialized_move(in, in + N, out.begin(), out.end());
    ASSERT_SAME_TYPE(decltype(result), std::ranges::uninitialized_move_result<Counted*, Counted*>);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_moves == N);
    assert(Counted::total_copies == 0);

    assert(std::equal(in, in + N, out.begin(), out.end()));
    assert(result.in == in + N);
    assert(result.out == out.end());

    std::destroy(out.begin(), out.end());
  }
  Counted::reset();

  // A range containing several objects, (range) overload.
  {
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, N> out;
    Counted::reset();

    std::ranges::subrange range(in, in + N);
    auto result = std::ranges::uninitialized_move(range, out);
    ASSERT_SAME_TYPE(decltype(result), std::ranges::uninitialized_move_result<Counted*, Counted*>);

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_moves == N);
    assert(Counted::total_copies == 0);
    assert(std::equal(in, in + N, out.begin(), out.end()));

    assert(result.in == in + N);
    assert(result.out == out.end());

    std::destroy(out.begin(), out.end());
  }
  Counted::reset();

  // Using `counted_iterator`.
  {
    constexpr int N = 3;
    Counted in[] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, 5> out;
    Counted::reset();

    std::counted_iterator iter(in, N);
    auto result = std::ranges::uninitialized_move(iter, std::default_sentinel, out.begin(), out.end());

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_moves == N);
    assert(Counted::total_copies == 0);
    assert(std::equal(in, in + N, out.begin(), out.begin() + N));

    assert(result.in == iter + N);
    assert(result.out == out.begin() + N);

    std::destroy(out.begin(), out.begin() + N);
  }
  Counted::reset();

  // Using `views::counted`.
  {
    constexpr int N = 3;
    Counted in[] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, 5> out;
    Counted::reset();

    auto view = std::views::counted(in, N);
    auto result = std::ranges::uninitialized_move(view, out);

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_moves == N);
    assert(Counted::total_copies == 0);
    assert(std::equal(in, in + N, out.begin(), out.begin() + N));

    assert(result.in == view.begin() + N);
    assert(result.out == out.begin() + N);

    std::destroy(out.begin(), out.begin() + N);
  }
  Counted::reset();

  // Using `reverse_view`.
  {
    constexpr int N = 3;
    Counted in[] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, 5> out;
    Counted::reset();

    std::ranges::subrange range(in, in + N);
    auto view = std::ranges::views::reverse(range);
    auto result = std::ranges::uninitialized_move(view, out);

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_moves == N);
    assert(Counted::total_copies == 0);

    Counted expected[N] = {Counted(3), Counted(2), Counted(1)};
    assert(std::equal(out.begin(), out.begin() + N, expected, expected + N));

    assert(result.in == view.begin() + N);
    assert(result.out == out.begin() + N);

    std::destroy(out.begin(), out.begin() + N);
  }
  Counted::reset();

  // Any existing values should be overwritten by move constructors.
  {
    constexpr int N = 5;
    int in[N] = {1, 2, 3, 4, 5};
    int out[N] = {6, 7, 8, 9, 10};
    assert(!std::equal(in, in + N, out, out + N));

    std::ranges::uninitialized_move(in, in + 1, out, out + N);
    assert(out[0] == 1);
    assert(out[1] == 7);

    std::ranges::uninitialized_move(in, in + N, out, out + N);
    assert(std::equal(in, in + N, out, out + N));
  }

  // An exception is thrown while objects are being created -- check that the objects in the source
  // range have been moved from. (iterator, sentinel) overload.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    constexpr int N = 3;
    Counted in[] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, 5> out;
    Counted::reset();

    Counted::throw_on = N; // When constructing out[3].
    try {
      std::ranges::uninitialized_move(in, in + 5, out.begin(), out.end());
      assert(false);
    } catch (...) {
    }
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == N);
    assert(Counted::total_moves == N);
    assert(Counted::total_copies == 0);

    assert(std::all_of(in, in + 1, [](const auto& e) { return e.moved_from; }));
    assert(std::none_of(in + N, in + 5, [](const auto& e) { return e.moved_from; }));

    std::destroy(out.begin(), out.begin() + N);
  }
  Counted::reset();

  // An exception is thrown while objects are being created -- check that the objects in the source
  // range have been moved from. (range) overload.
  {
    constexpr int N = 3;
    Counted in[] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, 5> out;
    Counted::reset();

    Counted::throw_on = N; // When constructing out[3].
    try {
      std::ranges::uninitialized_move(in, out);
      assert(false);
    } catch (...) {
    }
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == N);
    assert(Counted::total_moves == N);
    assert(Counted::total_copies == 0);

    assert(std::all_of(in, in + 1, [](const auto& e) { return e.moved_from; }));
    assert(std::none_of(in + N, in + 5, [](const auto& e) { return e.moved_from; }));

    std::destroy(out.begin(), out.begin() + N);
  }
  Counted::reset();
#endif // TEST_HAS_NO_EXCEPTIONS

  // Conversions, (iter, sentinel) overload.
  {
    constexpr int N = 3;
    int in[N] = {1, 2, 3};
    Buffer<double, N> out;

    std::ranges::uninitialized_move(in, in + N, out.begin(), out.end());
    assert(std::equal(in, in + N, out.begin(), out.end()));
  }

  // Conversions, (range) overload.
  {
    constexpr int N = 3;
    int in[N] = {1, 2, 3};
    Buffer<double, N> out;

    std::ranges::uninitialized_move(in, out);
    assert(std::equal(in, in + N, out.begin(), out.end()));
  }

  // Destination range is shorter than the source range, (iter, sentinel) overload.
  {
    constexpr int M = 3;
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, M> out;
    Counted::reset();

    auto result = std::ranges::uninitialized_move(in, in + N, out.begin(), out.end());
    assert(Counted::current_objects == M);
    assert(Counted::total_objects == M);
    assert(Counted::total_moves == M);
    assert(Counted::total_copies == 0);

    assert(std::equal(in, in + M, out.begin(), out.end()));
    assert(result.in == in + M);
    assert(result.out == out.end());
  }

  // Destination range is shorter than the source range, (range) overload.
  {
    constexpr int M = 3;
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, M> out;
    Counted::reset();

    std::ranges::subrange range(in, in + N);
    auto result = std::ranges::uninitialized_move(range, out);
    assert(Counted::current_objects == M);
    assert(Counted::total_objects == M);
    assert(Counted::total_moves == M);
    assert(Counted::total_copies == 0);

    assert(std::equal(in, in + M, out.begin(), out.end()));
    assert(result.in == in + M);
    assert(result.out == out.end());
  }

  // Ensure the `iter_move` customization point is being used.
  {
    constexpr int N = 3;
    int in[N] = {1, 2, 3};
    Buffer<int, N> out;
    int iter_moves = 0;
    adl::Iterator begin = adl::Iterator::TrackMoves(in, iter_moves);
    adl::Iterator end = adl::Iterator::TrackMoves(in + N, iter_moves);

    std::ranges::uninitialized_move(begin, end, out.begin(), out.end());
    assert(iter_moves == 3);
    iter_moves = 0;

    std::ranges::subrange range(begin, end);
    std::ranges::uninitialized_move(range, out);
    assert(iter_moves == 3);
    iter_moves = 0;
  }

  // Move-only iterators are supported.
  {
    using MoveOnlyIter = cpp20_input_iterator<const int*>;
    static_assert(!std::is_copy_constructible_v<MoveOnlyIter>);

    constexpr int N = 3;
    struct MoveOnlyRange {
      int buffer[N] = {1, 2, 3};
      auto begin() const { return MoveOnlyIter(buffer); }
      auto end() const { return sentinel_wrapper<MoveOnlyIter>(MoveOnlyIter(buffer)); }
    };
    static_assert(std::ranges::input_range<MoveOnlyRange>);
    MoveOnlyRange in;

    // (iter, sentinel) overload.
    {
      Buffer<int, N> out;
      std::ranges::uninitialized_move(in.begin(), in.end(), out.begin(), out.end());
    }

    // (range) overload.
    {
      Buffer<int, N> out;
      std::ranges::uninitialized_move(in, out);
    }
  }

  // MoveOnly types are supported
  {
    {
      MoveOnly a[] = {1, 2, 3, 4};
      Buffer<MoveOnly, 4> out;
      std::ranges::uninitialized_move(std::begin(a), std::end(a), std::begin(out), std::end(out));
      assert(std::ranges::equal(out, std::array<MoveOnly, 4>{1, 2, 3, 4}));
    }
    {
      MoveOnly a[] = {1, 2, 3, 4};
      Buffer<MoveOnly, 4> out;
      std::ranges::uninitialized_move(a, out);
      assert(std::ranges::equal(out, std::array<MoveOnly, 4>{1, 2, 3, 4}));
    }
  }

  // Test with an iterator that overloads operator== and operator!= as the input and output iterators
  {
    using T        = int;
    using Iterator = overload_compare_iterator<T*>;
    const int N    = 5;

    // input
    {
      char pool[sizeof(T) * N] = {0};
      T* p                     = reinterpret_cast<T*>(pool);
      T* p_end                 = reinterpret_cast<T*>(pool) + N;
      T array[N]               = {1, 2, 3, 4, 5};
      std::ranges::uninitialized_move(Iterator(array), Iterator(array + N), p, p_end);
      for (int i = 0; i != N; ++i) {
        assert(array[i] == p[i]);
      }
    }

    // output
    {
      char pool[sizeof(T) * N] = {0};
      T* p                     = reinterpret_cast<T*>(pool);
      T* p_end                 = reinterpret_cast<T*>(pool) + N;
      T array[N]               = {1, 2, 3, 4, 5};
      std::ranges::uninitialized_move(array, array + N, Iterator(p), Iterator(p_end));
      for (int i = 0; i != N; ++i) {
        assert(array[i] == p[i]);
      }
    }
  }

  return 0;
}
