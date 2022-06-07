//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

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
// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          weakly_incrementable O, copy_constructible F, class Proj1 = identity,
//          class Proj2 = identity>
//   requires indirectly_writable<O, indirect_result_t<F&, projected<I1, Proj1>,
//                                          projected<I2, Proj2>>>
//   constexpr ranges::binary_transform_result<I1, I2, O>
//     ranges::transform(I1 first1, S1 last1, I2 first2, S2 last2, O result,
//                       F binary_op, Proj1 proj1 = {}, Proj2 proj2 = {});
// template<input_range R1, input_range R2, weakly_incrementable O,
//          copy_constructible F, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_writable<O, indirect_result_t<F&, projected<iterator_t<R1>, Proj1>,
//                                          projected<iterator_t<R2>, Proj2>>>
//   constexpr ranges::binary_transform_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>, O>
//     ranges::transform(R1&& r1, R2&& r2, O result,
//                       F binary_op, Proj1 proj1 = {}, Proj2 proj2 = {});

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
concept HasTranformR = requires(Range r, int* out) {
  std::ranges::transform(r, out, std::identity{});
  std::ranges::transform(r, r, out, BinaryFunc{});
};
static_assert(HasTranformR<std::array<int, 1>>);
static_assert(!HasTranformR<int>);
static_assert(!HasTranformR<InputRangeNotDerivedFrom>);
static_assert(!HasTranformR<InputRangeNotIndirectlyReadable>);
static_assert(!HasTranformR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasTranformR<InputRangeNotSentinelSemiregular>);
static_assert(!HasTranformR<InputRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = It>
concept HasTransformIt = requires(It it, Sent sent, int* out) {
  std::ranges::transform(it, sent, out, std::identity{});
  std::ranges::transform(it, sent, it, sent, out, BinaryFunc{});
};
static_assert(HasTransformIt<int*>);
static_assert(!HasTransformIt<InputIteratorNotDerivedFrom>);
static_assert(!HasTransformIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasTransformIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasTransformIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasTransformIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

template <class It>
concept HasTransformOut = requires(int* it, int* sent, It out, std::array<int, 2> range) {
  std::ranges::transform(it, sent, out, std::identity{});
  std::ranges::transform(it, sent, it, sent, out, BinaryFunc{});
  std::ranges::transform(range, out, std::identity{});
  std::ranges::transform(range, range, out, BinaryFunc{});
};
static_assert(HasTransformOut<int*>);
static_assert(!HasTransformOut<WeaklyIncrementableNotMovable>);

// check indirectly_readable
static_assert(HasTransformOut<char*>);
static_assert(!HasTransformOut<int**>);

struct MoveOnlyFunctor {
  MoveOnlyFunctor(const MoveOnlyFunctor&) = delete;
  MoveOnlyFunctor(MoveOnlyFunctor&&) = default;
  int operator()(int);
  int operator()(int, int);
};

template <class Func>
concept HasTransformFuncUnary = requires(int* it, int* sent, int* out, std::array<int, 2> range, Func func) {
  std::ranges::transform(it, sent, out, func);
  std::ranges::transform(range, out, func);
};
static_assert(HasTransformFuncUnary<std::identity>);
static_assert(!HasTransformFuncUnary<MoveOnlyFunctor>);

template <class Func>
concept HasTransformFuncBinary = requires(int* it, int* sent, int* out, std::array<int, 2> range, Func func) {
  std::ranges::transform(it, sent, it, sent, out, func);
  std::ranges::transform(range, range, out, func);
};
static_assert(HasTransformFuncBinary<BinaryFunc>);
static_assert(!HasTransformFuncBinary<MoveOnlyFunctor>);

static_assert(std::is_same_v<std::ranges::unary_transform_result<int, long>, std::ranges::in_out_result<int, long>>);
static_assert(std::is_same_v<std::ranges::binary_transform_result<int, long, char>,
                             std::ranges::in_in_out_result<int, long, char>>);

template <class In1, class In2, class Out, class Sent1, class Sent2>
constexpr bool test_iterators() {
  { // simple
    { // unary
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

    { // binary
      {
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

      {
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
    }
  }

  { // first range empty
    { // unary
      {
        int a[] = {};
        int b[5];
        auto ret = std::ranges::transform(In1(a), Sent1(In1(a)), Out(b), [](int i) { return i * 2; });
        assert(base(ret.in) == a);
        assert(base(ret.out) == b);
      }

      {
        int a[] = {};
        int b[5];
        auto range = std::ranges::subrange(In1(a), Sent1(In1(a)));
        auto ret = std::ranges::transform(range, Out(b), [](int i) { return i * 2; });
        assert(base(ret.in) == a);
        assert(base(ret.out) == b);
      }
    }

    { // binary
      {
        int a[] = {};
        int b[] = {5, 4, 3, 2, 1};
        int c[5];

        auto ret = std::ranges::transform(
            In1(a), Sent1(In1(a)), In2(b), Sent2(In2(b + 5)), Out(c), [](int i, int j) { return i + j; });

        assert(base(ret.in1) == a);
        assert(base(ret.in2) == b);
        assert(base(ret.out) == c);
      }

      {
        int a[] = {};
        int b[] = {5, 4, 3, 2, 1};
        int c[5];

        auto range1 = std::ranges::subrange(In1(a), Sent1(In1(a)));
        auto range2 = std::ranges::subrange(In2(b), Sent2(In2(b + 5)));

        auto ret = std::ranges::transform(range1, range2, Out(c), [](int i, int j) { return i + j; });

        assert(base(ret.in1) == a);
        assert(base(ret.in2) == b);
        assert(base(ret.out) == c);
      }
    }
  }

  { // second range empty (binary)
    {
      int a[] = {5, 4, 3, 2, 1};
      int b[] = {};
      int c[5];

      auto ret = std::ranges::transform(
          In1(a), Sent1(In1(a + 5)), In2(b), Sent2(In2(b)), Out(c), [](int i, int j) { return i + j; });

      assert(base(ret.in1) == a);
      assert(base(ret.in2) == b);
      assert(base(ret.out) == c);
    }

    {
      int a[] = {5, 4, 3, 2, 1};
      int b[] = {};
      int c[5];

      auto range1 = std::ranges::subrange(In1(a), Sent1(In1(a + 5)));
      auto range2 = std::ranges::subrange(In2(b), Sent2(In2(b)));

      auto ret = std::ranges::transform(range1, range2, Out(c), [](int i, int j) { return i + j; });

      assert(base(ret.in1) == a);
      assert(base(ret.in2) == b);
      assert(base(ret.out) == c);
    }
  }

  { // both ranges empty (binary)
    {
      int a[] = {};
      int b[] = {};
      int c[5];

      auto ret = std::ranges::transform(
          In1(a), Sent1(In1(a)), In2(b), Sent2(In2(b)), Out(c), [](int i, int j) { return i + j; });

      assert(base(ret.in1) == a);
      assert(base(ret.in2) == b);
      assert(base(ret.out) == c);
    }

    {
      int a[] = {};
      int b[] = {};
      int c[5];

      auto range1 = std::ranges::subrange(In1(a), Sent1(In1(a)));
      auto range2 = std::ranges::subrange(In2(b), Sent2(In2(b)));

      auto ret = std::ranges::transform(range1, range2, Out(c), [](int i, int j) { return i + j; });

      assert(base(ret.in1) == a);
      assert(base(ret.in2) == b);
      assert(base(ret.out) == c);
    }
  }

  { // first range one element
    { // unary
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

    { // binary
      {
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

      {
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
    }
  }

  { // second range contains one element (binary)
    {
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

    {
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
  }

  { // check that the transform function and projection call counts are correct
    { // unary
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
    { // binary
      {
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
      {
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
    }
  }

  return true;
}

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
void test_iterators_in1_in2() {
  test_iterator_in1<cpp17_input_iterator<int*>, Out, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterator_in1<cpp20_input_iterator<int*>, Out, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterator_in1<forward_iterator<int*>, Out>();
  test_iterator_in1<bidirectional_iterator<int*>, Out>();
  test_iterator_in1<random_access_iterator<int*>, Out>();
  test_iterator_in1<contiguous_iterator<int*>, Out>();
  test_iterator_in1<int*, Out>();
}

constexpr bool test() {
  { // check that std::ranges::dangling is returned properly
    { // unary
      std::array<int, 5> b;
      std::same_as<std::ranges::in_out_result<std::ranges::dangling, int*>> auto ret =
          std::ranges::transform(std::array{1, 2, 3, 5, 4}, b.data(), [](int i) { return i * i; });
      assert((b == std::array{1, 4, 9, 25, 16}));
      assert(ret.out == b.data() + b.size());
    }
    // binary
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
    { // unary
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
    { // binary
      {
        struct S { int i; int other; };
        S a[] = { S{0, 0}, S{1, 0}, S{3, 0}, S{10, 0} };
        S b[] = { S{0, 10}, S{1, 20}, S{3, 30}, S{10, 40} };
        std::array<int, 4> c;
        std::ranges::transform(a, a + 4, b, b + 4, c.begin(), [](S s1, S s2) { return s1.i + s2.other; });
        assert((c == std::array{10, 21, 33, 50}));
      }
      {
        struct S { int i; int other; };
        S a[] = { S{0, 0}, S{1, 0}, S{3, 0}, S{10, 0} };
        S b[] = { S{0, 10}, S{1, 20}, S{3, 30}, S{10, 40} };
        std::array<int, 4> c;
        std::ranges::transform(a, b, c.begin(), [](S s1, S s2) { return s1.i + s2.other; });
        assert((c == std::array{10, 21, 33, 50}));
      }
    }
  }

  { // check that std::invoke is used
    { // unary
      struct S { int i; };
      S a[] = { S{1}, S{3}, S{2} };
      std::array<int, 3> b;
      auto ret = std::ranges::transform(a, b.data(), [](int i) { return i; }, &S::i);
      assert((b == std::array{1, 3, 2}));
      assert(ret.out == b.data() + 3);
    }
    { // binary
      struct S { int i; };
      S a[] = { S{1}, S{3}, S{2} };
      S b[] = { S{2}, S{5}, S{3} };
      std::array<int, 3> c;
      auto ret = std::ranges::transform(a, b, c.data(), [](int i, int j) { return i + j + 2; }, &S::i, &S::i);
      assert((c == std::array{5, 10, 7}));
      assert(ret.out == c.data() + 3);
    }
  }

  return true;
}

int main(int, char**) {
  test_iterators_in1_in2<cpp17_output_iterator<int*>>();
  test_iterators_in1_in2<cpp20_output_iterator<int*>>();
  test_iterators_in1_in2<forward_iterator<int*>>();
  test_iterators_in1_in2<bidirectional_iterator<int*>>();
  test_iterators_in1_in2<random_access_iterator<int*>>();
  test_iterators_in1_in2<contiguous_iterator<int*>>();
  test_iterators_in1_in2<int*>();
  test();
  static_assert(test());

  return 0;
}
