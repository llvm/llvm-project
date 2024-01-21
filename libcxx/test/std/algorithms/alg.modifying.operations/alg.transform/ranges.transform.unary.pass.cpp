//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O,
//          copy_constructible F, class Proj = identity>
//   requires indirectly_writable<O, indirect_result_t<F&, projected<I, Proj>>>
//   constexpr ranges::unary_transform_result<I, O>
//     ranges::transform(I first1, S last1, O result, F op, Proj proj = {});
// template<input_range R, weakly_incrementable O, copy_constructible F,
//          class Proj = identity>
//   requires indirectly_writable<O, indirect_result_t<F&, projected<iterator_t<R>, Proj>>>
//   constexpr ranges::unary_transform_result<borrowed_iterator_t<R>, O>
//     ranges::transform(R&& r, O result, F op, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <ranges>

#include "test_iterators.h"
#include "almost_satisfies_types.h"

template <class Range>
concept HasTransformR = requires(Range r, int* out) { std::ranges::transform(r, out, std::identity{}); };

static_assert(HasTransformR<std::array<int, 1>>);
static_assert(!HasTransformR<int>);
static_assert(!HasTransformR<InputRangeNotDerivedFrom>);
static_assert(!HasTransformR<InputRangeNotIndirectlyReadable>);
static_assert(!HasTransformR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasTransformR<InputRangeNotSentinelSemiregular>);
static_assert(!HasTransformR<InputRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = It>
concept HasTransformIt =
    requires(It it, Sent sent, int* out) { std::ranges::transform(it, sent, out, std::identity{}); };

static_assert(HasTransformIt<int*>);
static_assert(!HasTransformIt<InputIteratorNotDerivedFrom>);
static_assert(!HasTransformIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasTransformIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasTransformIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasTransformIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

template <class It>
concept HasTransformOut = requires(int* it, int* sent, It out, std::array<int, 2> range) {
  std::ranges::transform(it, sent, out, std::identity{});
  std::ranges::transform(range, out, std::identity{});
};
static_assert(HasTransformOut<int*>);
static_assert(!HasTransformOut<WeaklyIncrementableNotMovable>);

// check indirectly_readable
static_assert(HasTransformOut<char*>);
static_assert(!HasTransformOut<int**>);

struct MoveOnlyFunctor {
  MoveOnlyFunctor(const MoveOnlyFunctor&) = delete;
  MoveOnlyFunctor(MoveOnlyFunctor&&)      = default;
  int operator()(int);
};

template <class Func>
concept HasTransformFuncUnary = requires(int* it, int* sent, int* out, std::array<int, 2> range, Func func) {
  std::ranges::transform(it, sent, out, func);
  std::ranges::transform(range, out, func);
};
static_assert(HasTransformFuncUnary<std::identity>);
static_assert(!HasTransformFuncUnary<MoveOnlyFunctor>);

static_assert(std::is_same_v<std::ranges::unary_transform_result<int, long>, std::ranges::in_out_result<int, long>>);

// clang-format off
template <class In1, class Out, class Sent1>
constexpr bool test_iterators() {
  { // simple
    {
      int a[] = {1, 2, 3, 4, 5};
      int b[5];
      std::same_as<std::ranges::in_out_result<In1, Out>> decltype(auto) ret =
        std::ranges::transform(In1(a), Sent1(In1(a + 5)), Out(b), [](int i) { return i * 2; });
      assert((std::to_array(b) == std::array{2, 4, 6, 8, 10}));
      assert(base(ret.in) == a + 5);
      assert(base(ret.out) == b + 5);
    }

    {
      int a[] = {1, 2, 3, 4, 5};
      int b[5];
      auto range = std::ranges::subrange(In1(a), Sent1(In1(a + 5)));
      std::same_as<std::ranges::in_out_result<In1, Out>> decltype(auto) ret =
        std::ranges::transform(range, Out(b), [](int i) { return i * 2; });
      assert((std::to_array(b) == std::array{2, 4, 6, 8, 10}));
      assert(base(ret.in) == a + 5);
      assert(base(ret.out) == b + 5);
    }
  }

  { // first range empty
    {
      std::array<int, 0> a = {};
      int b[5];
      auto ret = std::ranges::transform(In1(a.data()), Sent1(In1(a.data())), Out(b), [](int i) { return i * 2; });
      assert(base(ret.in) == a.data());
      assert(base(ret.out) == b);
    }

    {
      std::array<int, 0> a = {};
      int b[5];
      auto range = std::ranges::subrange(In1(a.data()), Sent1(In1(a.data())));
      auto ret = std::ranges::transform(range, Out(b), [](int i) { return i * 2; });
      assert(base(ret.in) == a.data());
      assert(base(ret.out) == b);
    }
  }

  { // one element range
    {
      int a[] = {2};
      int b[5];
      auto ret = std::ranges::transform(In1(a), Sent1(In1(a + 1)), Out(b), [](int i) { return i * 2; });
      assert(b[0] == 4);
      assert(base(ret.in) == a + 1);
      assert(base(ret.out) == b + 1);
    }

    {
      int a[] = {2};
      int b[5];
      auto range = std::ranges::subrange(In1(a), Sent1(In1(a + 1)));
      auto ret = std::ranges::transform(range, Out(b), [](int i) { return i * 2; });
      assert(b[0] == 4);
      assert(base(ret.in) == a + 1);
      assert(base(ret.out) == b + 1);
    }
  }

  { // check that the transform function and projection call counts are correct
    {
      int predCount = 0;
      int projCount = 0;
      auto pred = [&](int) { ++predCount; return 1; };
      auto proj = [&](int) { ++projCount; return 0; };
      int a[] = {1, 2, 3, 4};
      std::array<int, 4> c;
      std::ranges::transform(In1(a), Sent1(In1(a + 4)), Out(c.data()), pred, proj);
      assert(predCount == 4);
      assert(projCount == 4);
      assert((c == std::array{1, 1, 1, 1}));
    }
    {
      int predCount = 0;
      int projCount = 0;
      auto pred = [&](int) { ++predCount; return 1; };
      auto proj = [&](int) { ++projCount; return 0; };
      int a[] = {1, 2, 3, 4};
      std::array<int, 4> c;
      auto range = std::ranges::subrange(In1(a), Sent1(In1(a + 4)));
      std::ranges::transform(range, Out(c.data()), pred, proj);
      assert(predCount == 4);
      assert(projCount == 4);
      assert((c == std::array{1, 1, 1, 1}));
    }
  }
  return true;
}
// clang-format on

template <class Out>
constexpr void test_iterator_in1() {
  test_iterators<cpp17_input_iterator<int*>, Out, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterators<cpp20_input_iterator<int*>, Out, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<forward_iterator<int*>, Out, forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>, Out, bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>, Out, random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>, Out, contiguous_iterator<int*>>();
  test_iterators<int*, Out, int*>();
  // static_asserting here to avoid hitting the constant evaluation step limit
  static_assert(test_iterators<cpp17_input_iterator<int*>, Out, sentinel_wrapper<cpp17_input_iterator<int*>>>());
  static_assert(test_iterators<cpp20_input_iterator<int*>, Out, sentinel_wrapper<cpp20_input_iterator<int*>>>());
  static_assert(test_iterators<forward_iterator<int*>, Out, forward_iterator<int*>>());
  static_assert(test_iterators<bidirectional_iterator<int*>, Out, bidirectional_iterator<int*>>());
  static_assert(test_iterators<random_access_iterator<int*>, Out, random_access_iterator<int*>>());
  static_assert(test_iterators<contiguous_iterator<int*>, Out, contiguous_iterator<int*>>());
  static_assert(test_iterators<int*, Out, int*>());
}

constexpr bool test() {
  { // check that std::ranges::dangling is returned properly
    std::array<int, 5> b;
    std::same_as<std::ranges::in_out_result<std::ranges::dangling, int*>> auto ret =
        std::ranges::transform(std::array{1, 2, 3, 5, 4}, b.data(), [](int i) { return i * i; });
    assert((b == std::array{1, 4, 9, 25, 16}));
    assert(ret.out == b.data() + b.size());
  }

  { // check that returning another type from the projection works
    {
      struct S { int i; int other; };
      S a[] = { S{0, 0}, S{1, 0}, S{3, 0}, S{10, 0} };
      std::array<int, 4> b;
      std::ranges::transform(a, a + 4, b.begin(), [](S s) { return s.i; });
      assert((b == std::array{0, 1, 3, 10}));
    }
    {
      struct S { int i; int other; };
      S a[] = { S{0, 0}, S{1, 0}, S{3, 0}, S{10, 0} };
      std::array<int, 4> b;
      std::ranges::transform(a, b.begin(), [](S s) { return s.i; });
      assert((b == std::array{0, 1, 3, 10}));
    }
  }

  { // check that std::invoke is used
    struct S { int i; };
    S a[] = { S{1}, S{3}, S{2} };
    std::array<int, 3> b;
    auto ret = std::ranges::transform(a, b.data(), [](int i) { return i; }, &S::i);
    assert((b == std::array{1, 3, 2}));
    assert(ret.out == b.data() + 3);
  }

  return true;
}

int main(int, char**) {
  test_iterator_in1<cpp17_output_iterator<int*>>();
  test_iterator_in1<cpp20_output_iterator<int*>>();
  test_iterator_in1<forward_iterator<int*>>();
  test_iterator_in1<bidirectional_iterator<int*>>();
  test_iterator_in1<random_access_iterator<int*>>();
  test_iterator_in1<contiguous_iterator<int*>>();
  test_iterator_in1<int*>();
  test();
  static_assert(test());

  return 0;
}
