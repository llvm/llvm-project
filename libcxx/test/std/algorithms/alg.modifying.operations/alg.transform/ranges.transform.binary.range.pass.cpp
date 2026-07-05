//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_range R1, input_range R2, weakly_incrementable O,
//          copy_constructible F, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_writable<O, indirect_result_t<F&, projected<iterator_t<R1>, Proj1>,
//                                          projected<iterator_t<R2>, Proj2>>>
//   constexpr ranges::binary_transform_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>, O>
//     ranges::transform(R1&& r1, R2&& r2, O result,
//                       F binary_op, Proj1 proj1 = {}, Proj2 proj2 = {});

// The iterator overloads are tested in ranges.transform.binary.iterator.pass.cpp.

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <ranges>

#include "test_iterators.h"
#include "almost_satisfies_types.h"

struct BinaryFunc {
  int operator()(int, int);
};

template <class Range>
concept HasTransformR = requires(Range r, int* out) { std::ranges::transform(r, r, out, BinaryFunc{}); };

static_assert(HasTransformR<std::array<int, 1>>);
static_assert(!HasTransformR<int>);
static_assert(!HasTransformR<InputRangeNotDerivedFrom>);
static_assert(!HasTransformR<InputRangeNotIndirectlyReadable>);
static_assert(!HasTransformR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasTransformR<InputRangeNotSentinelSemiregular>);
static_assert(!HasTransformR<InputRangeNotSentinelEqualityComparableWith>);

template <class It>
concept HasTransformOut = requires(int* it, int* sent, It out, std::array<int, 2> range) {
  std::ranges::transform(range, range, out, BinaryFunc{});
};
static_assert(HasTransformOut<int*>);
static_assert(!HasTransformOut<WeaklyIncrementableNotMovable>);

// check indirectly_readable
static_assert(HasTransformOut<char*>);
static_assert(!HasTransformOut<int**>);

struct MoveOnlyFunctor {
  MoveOnlyFunctor(const MoveOnlyFunctor&) = delete;
  MoveOnlyFunctor(MoveOnlyFunctor&&)      = default;
  int operator()(int, int);
};

template <class Func>
concept HasTransformFuncBinary = requires(int* it, int* sent, int* out, std::array<int, 2> range, Func func) {
  std::ranges::transform(range, range, out, func);
};
static_assert(HasTransformFuncBinary<BinaryFunc>);
static_assert(!HasTransformFuncBinary<MoveOnlyFunctor>);

static_assert(std::is_same_v<std::ranges::binary_transform_result<int, long, char>,
                             std::ranges::in_in_out_result<int, long, char>>);

// clang-format off
template <class In1, class In2, class Out, class Sent1, class Sent2>
constexpr bool test_iterators() {
  { // simple
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {5, 4, 3, 2, 1};
    int c[5];

    auto range1 = std::ranges::subrange(In1(a), Sent1(In1(a + 5)));
    auto range2 = std::ranges::subrange(In2(b), Sent2(In2(b + 5)));

    std::same_as<std::ranges::in_in_out_result<In1, In2, Out>> decltype(auto) ret = std::ranges::transform(
        range1, range2, Out(c), [](int i, int j) { return i + j; });

    assert((std::to_array(c) == std::array{6, 6, 6, 6, 6}));
    assert(base(ret.in1) == a + 5);
    assert(base(ret.in2) == b + 5);
    assert(base(ret.out) == c + 5);
  }

  { // first range empty
    std::array<int, 0> a = {};
    int b[] = {5, 4, 3, 2, 1};
    int c[5];

    auto range1 = std::ranges::subrange(In1(a.data()), Sent1(In1(a.data())));
    auto range2 = std::ranges::subrange(In2(b), Sent2(In2(b + 5)));

    auto ret = std::ranges::transform(range1, range2, Out(c), [](int i, int j) { return i + j; });

    assert(base(ret.in1) == a.data());
    assert(base(ret.in2) == b);
    assert(base(ret.out) == c);
  }

  { // second range empty
    int a[] = {5, 4, 3, 2, 1};
    std::array<int, 0> b = {};
    int c[5];

    auto range1 = std::ranges::subrange(In1(a), Sent1(In1(a + 5)));
    auto range2 = std::ranges::subrange(In2(b.data()), Sent2(In2(b.data())));

    auto ret = std::ranges::transform(range1, range2, Out(c), [](int i, int j) { return i + j; });

    assert(base(ret.in1) == a);
    assert(base(ret.in2) == b.data());
    assert(base(ret.out) == c);
  }

  { // both ranges empty
    std::array<int, 0> a = {};
    std::array<int, 0> b = {};
    int c[5];

    auto range1 = std::ranges::subrange(In1(a.data()), Sent1(In1(a.data())));
    auto range2 = std::ranges::subrange(In2(b.data()), Sent2(In2(b.data())));

    auto ret = std::ranges::transform(range1, range2, Out(c), [](int i, int j) { return i + j; });

    assert(base(ret.in1) == a.data());
    assert(base(ret.in2) == b.data());
    assert(base(ret.out) == c);
  }

  { // first range one element
    int a[] = {2};
    int b[] = {5, 4, 3, 2, 1};
    int c[5];

    auto range1 = std::ranges::subrange(In1(a), Sent1(In1(a + 1)));
    auto range2 = std::ranges::subrange(In2(b), Sent2(In2(b + 5)));

    auto ret = std::ranges::transform(range1, range2, Out(c), [](int i, int j) { return i + j; });

    assert(c[0] == 7);
    assert(base(ret.in1) == a + 1);
    assert(base(ret.in2) == b + 1);
    assert(base(ret.out) == c + 1);
  }

  { // second range contains one element
    int a[] = {5, 4, 3, 2, 1};
    int b[] = {4};
    int c[5];

    auto range1 = std::ranges::subrange(In1(a), Sent1(In1(a + 5)));
    auto range2 = std::ranges::subrange(In2(b), Sent2(In2(b + 1)));

    auto ret = std::ranges::transform(range1, range2, Out(c), [](int i, int j) { return i + j; });

    assert(c[0] == 9);
    assert(base(ret.in1) == a + 1);
    assert(base(ret.in2) == b + 1);
    assert(base(ret.out) == c + 1);
  }

  { // check that the transform function and projection call counts are correct
    int predCount = 0;
    int proj1Count = 0;
    int proj2Count = 0;
    auto pred = [&](int, int) { ++predCount; return 1; };
    auto proj1 = [&](int) { ++proj1Count; return 0; };
    auto proj2 = [&](int) { ++proj2Count; return 0; };
    int a[] = {1, 2, 3, 4};
    int b[] = {1, 2, 3, 4};
    std::array<int, 4> c;
    auto range1 = std::ranges::subrange(In1(a), Sent1(In1(a + 4)));
    auto range2 = std::ranges::subrange(In2(b), Sent2(In2(b + 4)));
    std::ranges::transform(range1, range2, Out(c.data()), pred, proj1, proj2);
    assert(predCount == 4);
    assert(proj1Count == 4);
    assert(proj2Count == 4);
    assert((c == std::array{1, 1, 1, 1}));
  }

  return true;
}
// clang-format on

template <class In2, class Out, class Sent2 = In2>
constexpr void test_iterator_in1() {
  test_iterators<cpp17_input_iterator<int*>, In2, Out, sentinel_wrapper<cpp17_input_iterator<int*>>, Sent2>();
  test_iterators<cpp20_input_iterator<int*>, In2, Out, sentinel_wrapper<cpp20_input_iterator<int*>>, Sent2>();
  test_iterators<forward_iterator<int*>, In2, Out, forward_iterator<int*>, Sent2>();
  test_iterators<bidirectional_iterator<int*>, In2, Out, bidirectional_iterator<int*>, Sent2>();
  test_iterators<random_access_iterator<int*>, In2, Out, random_access_iterator<int*>, Sent2>();
  test_iterators<contiguous_iterator<int*>, In2, Out, contiguous_iterator<int*>, Sent2>();
  test_iterators<int*, In2, Out, int*, Sent2>();
  // static_asserting here to avoid hitting the constant evaluation step limit
  static_assert(test_iterators<cpp17_input_iterator<int*>, In2, Out, sentinel_wrapper<cpp17_input_iterator<int*>>, Sent2>());
  static_assert(test_iterators<cpp20_input_iterator<int*>, In2, Out, sentinel_wrapper<cpp20_input_iterator<int*>>, Sent2>());
  static_assert(test_iterators<forward_iterator<int*>, In2, Out, forward_iterator<int*>, Sent2>());
  static_assert(test_iterators<bidirectional_iterator<int*>, In2, Out, bidirectional_iterator<int*>, Sent2>());
  static_assert(test_iterators<random_access_iterator<int*>, In2, Out, random_access_iterator<int*>, Sent2>());
  static_assert(test_iterators<contiguous_iterator<int*>, In2, Out, contiguous_iterator<int*>, Sent2>());
  static_assert(test_iterators<int*, In2, Out, int*, Sent2>());
}

template <class Out>
constexpr void test_iterators_in1_in2() {
  test_iterator_in1<cpp17_input_iterator<int*>, Out, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterator_in1<cpp20_input_iterator<int*>, Out, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterator_in1<forward_iterator<int*>, Out>();
  test_iterator_in1<bidirectional_iterator<int*>, Out>();
  test_iterator_in1<random_access_iterator<int*>, Out>();
  test_iterator_in1<contiguous_iterator<int*>, Out>();
  test_iterator_in1<int*, Out>();
}

constexpr bool test() {
  test_iterators_in1_in2<cpp17_output_iterator<int*>>();
  test_iterators_in1_in2<cpp20_output_iterator<int*>>();
  test_iterators_in1_in2<forward_iterator<int*>>();
  test_iterators_in1_in2<bidirectional_iterator<int*>>();
  test_iterators_in1_in2<random_access_iterator<int*>>();
  test_iterators_in1_in2<contiguous_iterator<int*>>();
  test_iterators_in1_in2<int*>();

  { // check that std::ranges::dangling is returned properly
    {
      int b[] = {2, 5, 4, 3, 1};
      std::array<int, 5> c;
      std::same_as<std::ranges::in_in_out_result<std::ranges::dangling, int*, int*>> auto ret =
          std::ranges::transform(std::array{1, 2, 3, 5, 4}, b, c.data(), [](int i, int j) { return i * j; });
      assert((c == std::array{2, 10, 12, 15, 4}));
      assert(ret.in2 == b + 5);
      assert(ret.out == c.data() + c.size());
    }
    {
      int a[] = {2, 5, 4, 3, 1, 4, 5, 6};
      std::array<int, 8> c;
      std::same_as<std::ranges::in_in_out_result<int*, std::ranges::dangling, int*>> auto ret =
          std::ranges::transform(a, std::array{1, 2, 3, 5, 4, 5, 6, 7}, c.data(), [](int i, int j) { return i * j; });
      assert((c == std::array{2, 10, 12, 15, 4, 20, 30, 42}));
      assert(ret.in1 == a + 8);
      assert(ret.out == c.data() + c.size());
    }
    {
      std::array<int, 3> c;
      std::same_as<std::ranges::in_in_out_result<std::ranges::dangling, std::ranges::dangling, int*>> auto ret =
          std::ranges::transform(std::array{4, 4, 4}, std::array{4, 4, 4}, c.data(), [](int i, int j) { return i * j; });
      assert((c == std::array{16, 16, 16}));
      assert(ret.out == c.data() + c.size());
    }
  }

  { // check that returning another type from the projection works
    struct S { int i; int other; };
    S a[] = { S{0, 0}, S{1, 0}, S{3, 0}, S{10, 0} };
    S b[] = { S{0, 10}, S{1, 20}, S{3, 30}, S{10, 40} };
    std::array<int, 4> c;
    std::ranges::transform(a, b, c.begin(), [](S s1, S s2) { return s1.i + s2.other; });
    assert((c == std::array{10, 21, 33, 50}));
  }

  { // check that std::invoke is used
    struct S { int i; };
    S a[] = { S{1}, S{3}, S{2} };
    S b[] = { S{2}, S{5}, S{3} };
    std::array<int, 3> c;
    auto ret = std::ranges::transform(a, b, c.data(), [](int i, int j) { return i + j + 2; }, &S::i, &S::i);
    assert((c == std::array{5, 10, 7}));
    assert(ret.out == c.data() + 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
