//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          weakly_incrementable O, copy_constructible F, class Proj1 = identity,
//          class Proj2 = identity>
//   requires indirectly_writable<O, indirect_result_t<F&, projected<I1, Proj1>,
//                                          projected<I2, Proj2>>>
//   constexpr ranges::binary_transform_result<I1, I2, O>
//     ranges::transform(I1 first1, S1 last1, I2 first2, S2 last2, O result,
//                       F binary_op, Proj1 proj1 = {}, Proj2 proj2 = {});

// The range overloads are tested in ranges.transform.binary.range.pass.cpp.

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

template <class It, class Sent = It>
concept HasTransformIt =
    requires(It it, Sent sent, int* out) { std::ranges::transform(it, sent, it, sent, out, BinaryFunc{}); };
static_assert(HasTransformIt<int*>);
static_assert(!HasTransformIt<InputIteratorNotDerivedFrom>);
static_assert(!HasTransformIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasTransformIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasTransformIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasTransformIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

template <class It>
concept HasTransformOut = requires(int* it, int* sent, It out, std::array<int, 2> range) {
  std::ranges::transform(it, sent, it, sent, out, BinaryFunc{});
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
  std::ranges::transform(it, sent, it, sent, out, func);
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

    std::same_as<std::ranges::in_in_out_result<In1, In2, Out>> decltype(auto) ret = std::ranges::transform(
        In1(a), Sent1(In1(a + 5)), In2(b), Sent2(In2(b + 5)), Out(c), [](int i, int j) { return i + j; });

    assert((std::to_array(c) == std::array{6, 6, 6, 6, 6}));
    assert(base(ret.in1) == a + 5);
    assert(base(ret.in2) == b + 5);
    assert(base(ret.out) == c + 5);
  }

  { // first range empty
    std::array<int, 0> a = {};
    int b[] = {5, 4, 3, 2, 1};
    int c[5];

    auto ret = std::ranges::transform(
        In1(a.data()), Sent1(In1(a.data())), In2(b), Sent2(In2(b + 5)), Out(c), [](int i, int j) { return i + j; });

    assert(base(ret.in1) == a.data());
    assert(base(ret.in2) == b);
    assert(base(ret.out) == c);
  }

  { // second range empty
    int a[] = {5, 4, 3, 2, 1};
    std::array<int, 0> b = {};
    int c[5];

    auto ret = std::ranges::transform(
        In1(a), Sent1(In1(a + 5)), In2(b.data()), Sent2(In2(b.data())), Out(c), [](int i, int j) { return i + j; });

    assert(base(ret.in1) == a);
    assert(base(ret.in2) == b.data());
    assert(base(ret.out) == c);
  }

  { // both ranges empty
    std::array<int, 0> a = {};
    std::array<int, 0> b = {};
    int c[5];

    auto ret = std::ranges::transform(
        In1(a.data()), Sent1(In1(a.data())), In2(b.data()), Sent2(In2(b.data())), Out(c), [](int i, int j) { return i + j; });

    assert(base(ret.in1) == a.data());
    assert(base(ret.in2) == b.data());
    assert(base(ret.out) == c);
  }

  { // first range one element
    int a[] = {2};
    int b[] = {5, 4, 3, 2, 1};
    int c[5];

    auto ret = std::ranges::transform(
        In1(a), Sent1(In1(a + 1)), In2(b), Sent2(In2(b + 5)), Out(c), [](int i, int j) { return i + j; });

    assert(c[0] == 7);
    assert(base(ret.in1) == a + 1);
    assert(base(ret.in2) == b + 1);
    assert(base(ret.out) == c + 1);
  }

  { // second range contains one element
    int a[] = {5, 4, 3, 2, 1};
    int b[] = {4};
    int c[5];

    auto ret = std::ranges::transform(
        In1(a), Sent1(In1(a + 5)), In2(b), Sent2(In2(b + 1)), Out(c), [](int i, int j) { return i + j; });

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
    std::ranges::transform(In1(a), Sent1(In1(a + 4)), In2(b), Sent2(In2(b + 4)), Out(c.data()), pred, proj1, proj2);
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

  { // check that returning another type from the projection works
    struct S { int i; int other; };
    S a[] = { S{0, 0}, S{1, 0}, S{3, 0}, S{10, 0} };
    S b[] = { S{0, 10}, S{1, 20}, S{3, 30}, S{10, 40} };
    std::array<int, 4> c;
    std::ranges::transform(a, a + 4, b, b + 4, c.begin(), [](S s1, S s2) { return s1.i + s2.other; });
    assert((c == std::array{10, 21, 33, 50}));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
