//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          weakly_incrementable O, class Comp = ranges::less,
//          class Proj1 = identity, class Proj2 = identity>
//   requires mergeable<I1, I2, O, Comp, Proj1, Proj2>
//   constexpr set_intersection_result<I1, I2, O>
//     set_intersection(I1 first1, S1 last1, I2 first2, S2 last2, O result,
//                      Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});                         // since C++20
//
// template<input_range R1, input_range R2, weakly_incrementable O,
//          class Comp = ranges::less, class Proj1 = identity, class Proj2 = identity>
//   requires mergeable<iterator_t<R1>, iterator_t<R2>, O, Comp, Proj1, Proj2>
//   constexpr set_intersection_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>, O>
//     set_intersection(R1&& r1, R2&& r2, O result,
//                      Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});                         // since C++20

#include <algorithm>
#include <array>
#include <concepts>

#include "almost_satisfies_types.h"
#include "MoveOnly.h"
#include "test_iterators.h"
#include "../../sortable_helpers.h"

// Test iterator overload's constraints:
// =====================================
template <class InIter1 = int*, class Sent1 = int*, class InIter2 = int*, class Sent2 = int*, class OutIter = int*>
concept HasSetIntersectionIter =
    requires(InIter1&& inIter1, InIter2&& inIter2, OutIter&& outIter, Sent1&& sent1, Sent2&& sent2) {
      std::ranges::set_intersection(
          std::forward<InIter1>(inIter1),
          std::forward<Sent1>(sent1),
          std::forward<InIter2>(inIter2),
          std::forward<Sent2>(sent2),
          std::forward<OutIter>(outIter));
    };

static_assert(HasSetIntersectionIter<int*, int*, int*, int*, int*>);

// !std::input_iterator<I1>
static_assert(!HasSetIntersectionIter<InputIteratorNotDerivedFrom>);

// !std::sentinel_for<S1, I1>
static_assert(!HasSetIntersectionIter<int*, SentinelForNotSemiregular>);

// !std::input_iterator<I2>
static_assert(!HasSetIntersectionIter<int*, int*, InputIteratorNotDerivedFrom>);

// !std::sentinel_for<S2, I2>
static_assert(!HasSetIntersectionIter<int*, int*, int*, SentinelForNotSemiregular>);

// !std::weakly_incrementable<O>
static_assert(!HasSetIntersectionIter<int*, int*, int*, int*, WeaklyIncrementableNotMovable>);

// !std::mergeable<I1, I2, O, Comp, Proj1, Proj2>
static_assert(!HasSetIntersectionIter<MoveOnly*, MoveOnly*, MoveOnly*, MoveOnly*, MoveOnly*>);

// Test range overload's constraints:
// =====================================

template <class Range1, class Range2, class OutIter>
concept HasSetIntersectionRange =
    requires(Range1&& range1, Range2&& range2, OutIter&& outIter) {
      std::ranges::set_intersection(
          std::forward<Range1>(range1), std::forward<Range2>(range2), std::forward<OutIter>(outIter));
    };

static_assert(HasSetIntersectionRange<UncheckedRange<int*>, UncheckedRange<int*>, int*>);

// !std::input_range<R2>
static_assert(!HasSetIntersectionRange<UncheckedRange<InputIteratorNotDerivedFrom>, UncheckedRange<int*>, int*>);

// !std::input_range<R2>
static_assert(!HasSetIntersectionRange<UncheckedRange<int*>, UncheckedRange<InputIteratorNotDerivedFrom>, int*>);

// !std::weakly_incrementable<O>
static_assert(!HasSetIntersectionRange<UncheckedRange<int*>, UncheckedRange<int*>, WeaklyIncrementableNotMovable >);

// !std::mergeable<iterator_t<R1>, iterator_t<R2>, O, Comp, Proj1, Proj2>
static_assert(!HasSetIntersectionRange<UncheckedRange<MoveOnly*>, UncheckedRange<MoveOnly*>, MoveOnly*>);

using std::ranges::set_intersection_result;

template <class In1, class In2, class Out, std::size_t N1, std::size_t N2, std::size_t N3>
constexpr void testSetIntersectionImpl(std::array<int, N1> in1, std::array<int, N2> in2, std::array<int, N3> expected) {
  // TODO: std::ranges::set_intersection calls std::ranges::copy
  // std::ranges::copy(contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>, contiguous_iterator<int*>) doesn't seem to work.
  // It seems that std::ranges::copy calls std::copy, which unwraps contiguous_iterator<int*> into int*,
  // and then it failed because there is no == between int* and sentinel_wrapper<contiguous_iterator<int*>>
  using Sent1 = std::conditional_t<std::contiguous_iterator<In1>, In1, sentinel_wrapper<In1>>;
  using Sent2 = std::conditional_t<std::contiguous_iterator<In2>, In2, sentinel_wrapper<In2>>;

  // iterator overload
  {
    std::array<int, N3> out;
    std::same_as<set_intersection_result<In1, In2, Out>> decltype(auto) result = std::ranges::set_intersection(
        In1{in1.data()},
        Sent1{In1{in1.data() + in1.size()}},
        In2{in2.data()},
        Sent2{In2{in2.data() + in2.size()}},
        Out{out.data()});
    assert(std::ranges::equal(out, expected));

    assert(base(result.in1) == in1.data() + in1.size());
    assert(base(result.in2) == in2.data() + in2.size());
    assert(base(result.out) == out.data() + out.size());
  }

  // range overload
  {
    std::array<int, N3> out;
    std::ranges::subrange r1{In1{in1.data()}, Sent1{In1{in1.data() + in1.size()}}};
    std::ranges::subrange r2{In2{in2.data()}, Sent2{In2{in2.data() + in2.size()}}};
    std::same_as<set_intersection_result<In1, In2, Out>> decltype(auto) result =
        std::ranges::set_intersection(r1, r2, Out{out.data()});
    assert(std::ranges::equal(out, expected));

    assert(base(result.in1) == in1.data() + in1.size());
    assert(base(result.in2) == in2.data() + in2.size());
    assert(base(result.out) == out.data() + out.size());
  }
}

template <class In1, class In2, class Out>
constexpr void testImpl() {
  // range 1 shorter than range2
  {
    std::array in1{0, 1, 5, 6, 9, 10};
    std::array in2{3, 6, 7, 9, 13, 15, 100};
    std::array expected{6, 9};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 2 shorter than range 1
  {
    std::array in1{2, 6, 8, 12, 15, 16};
    std::array in2{0, 2, 8};
    std::array expected{2, 8};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 and range 2 has the same length but different elements
  {
    std::array in1{2, 6, 8, 12, 15, 16};
    std::array in2{0, 2, 8, 15, 17, 19};
    std::array expected{2, 8, 15};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 == range 2
  {
    std::array in1{0, 1, 2};
    std::array in2{0, 1, 2};
    std::array expected{0, 1, 2};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 is super set of range 2
  {
    std::array in1{8, 8, 10, 12, 13};
    std::array in2{8, 10};
    std::array expected{8, 10};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 2 is super set of range 1
  {
    std::array in1{0, 1, 1};
    std::array in2{0, 1, 1, 2, 5};
    std::array expected{0, 1, 1};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 and range 2 have no elements in common
  {
    std::array in1{7, 7, 9, 12};
    std::array in2{1, 5, 5, 8, 10};
    std::array<int, 0> expected{};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 and range 2 have duplicated equal elements
  {
    std::array in1{7, 7, 9, 12};
    std::array in2{7, 7, 7, 13};
    std::array expected{7, 7};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 is empty
  {
    std::array<int, 0> in1{};
    std::array in2{3, 4, 5};
    std::array<int, 0> expected{};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 2 is empty
  {
    std::array in1{3, 4, 5};
    std::array<int, 0> in2{};
    std::array<int, 0> expected{};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // both ranges are empty
  {
    std::array<int, 0> in1{};
    std::array<int, 0> in2{};
    std::array<int, 0> expected{};
    testSetIntersectionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // check that ranges::dangling is returned for non-borrowed_range
  {
    std::array r1{3, 6, 7, 9};
    int r2[] = {2, 3, 4, 5, 6};
    std::array<int, 2> out;
    std::same_as<set_intersection_result<std::ranges::dangling, int*, int*>> decltype(auto) result =
        std::ranges::set_intersection(NonBorrowedRange<In1>{r1.data(), r1.size()}, r2, out.data());
    assert(base(result.in2) == r2 + 5);
    assert(base(result.out) == out.data() + out.size());
    assert(std::ranges::equal(out, std::array{3, 6}));
  }
}

template <class InIter2, class OutIter>
constexpr void withAllPermutationsOfInIter1() {
  // C++17 InputIterator may or may not satisfy std::input_iterator
  testImpl<cpp20_input_iterator<int*>, InIter2, OutIter>();
  testImpl<forward_iterator<int*>, InIter2, OutIter>();
  testImpl<bidirectional_iterator<int*>, InIter2, OutIter>();
  testImpl<random_access_iterator<int*>, InIter2, OutIter>();
  testImpl<contiguous_iterator<int*>, InIter2, OutIter>();
}

template <class OutIter>
constexpr bool withAllPermutationsOfInIter1AndInIter2() {
  withAllPermutationsOfInIter1<cpp20_input_iterator<int*>, OutIter>();
  withAllPermutationsOfInIter1<forward_iterator<int*>, OutIter>();
  withAllPermutationsOfInIter1<bidirectional_iterator<int*>, OutIter>();
  withAllPermutationsOfInIter1<random_access_iterator<int*>, OutIter>();
  withAllPermutationsOfInIter1<contiguous_iterator<int*>, OutIter>();
  return true;
}

constexpr void runAllIteratorPermutationsTests() {
  withAllPermutationsOfInIter1AndInIter2<cpp20_output_iterator<int*>>();
  withAllPermutationsOfInIter1AndInIter2<cpp20_input_iterator<int*>>();
  withAllPermutationsOfInIter1AndInIter2<forward_iterator<int*>>();
  withAllPermutationsOfInIter1AndInIter2<bidirectional_iterator<int*>>();
  withAllPermutationsOfInIter1AndInIter2<random_access_iterator<int*>>();
  withAllPermutationsOfInIter1AndInIter2<contiguous_iterator<int*>>();

  static_assert(withAllPermutationsOfInIter1AndInIter2<cpp20_output_iterator<int*>>());
  static_assert(withAllPermutationsOfInIter1AndInIter2<cpp20_input_iterator<int*>>());
  static_assert(withAllPermutationsOfInIter1AndInIter2<forward_iterator<int*>>());
  static_assert(withAllPermutationsOfInIter1AndInIter2<bidirectional_iterator<int*>>());
  static_assert(withAllPermutationsOfInIter1AndInIter2<random_access_iterator<int*>>());
  static_assert(withAllPermutationsOfInIter1AndInIter2<contiguous_iterator<int*>>());
}

constexpr bool test() {
  // check that every element is copied exactly once
  {
    std::array<TracedCopy, 5> r1{3, 5, 8, 15, 16};
    std::array<TracedCopy, 3> r2{1, 3, 8};

    // iterator overload
    {
      std::array<TracedCopy, 2> out;
      auto result = std::ranges::set_intersection(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data());

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
      assert(std::ranges::equal(out, std::array<TracedCopy, 2>{3, 8}));

      assert(std::ranges::all_of(out, &TracedCopy::copiedOnce));
    }

    // range overload
    {
      std::array<TracedCopy, 2> out;
      auto result = std::ranges::set_intersection(r1, r2, out.data());

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
      assert(std::ranges::equal(out, std::array<TracedCopy, 2>{3, 8}));

      assert(std::ranges::all_of(out, &TracedCopy::copiedOnce));
    }
  }

  struct IntAndOrder {
    int data;
    int order;

    constexpr auto operator==(const IntAndOrder& o) const { return data == o.data; }
    constexpr auto operator<=>(const IntAndOrder& o) const { return data <=> o.data; }
  };

  // Stable. If [first1, last1) contains m elements that are equivalent to each other and [first2, last2)
  // contains n elements that are equivalent to them, the first min(m, n) elements are copied from the first
  // range to the output range, in order.
  {
    std::array<IntAndOrder, 5> r1{{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}}};
    std::array<IntAndOrder, 3> r2{{{0, 5}, {0, 6}, {0, 7}}};

    // iterator overload
    {
      std::array<IntAndOrder, 3> out;
      std::ranges::set_intersection(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data());

      assert(std::ranges::equal(out, std::array{0, 0, 0}, {}, &IntAndOrder::data));
      assert(std::ranges::equal(out, std::array{0, 1, 2}, {}, &IntAndOrder::order));
    }

    // range overload
    {
      std::array<IntAndOrder, 3> out;
      std::ranges::set_intersection(r1, r2, out.data());

      assert(std::ranges::equal(out, std::array{0, 0, 0}, {}, &IntAndOrder::data));
      assert(std::ranges::equal(out, std::array{0, 1, 2}, {}, &IntAndOrder::order));
    }
  }

  struct Data {
    int data;

    constexpr bool smallerThan(const Data& o) const { return data < o.data; }
  };

  // Test custom comparator
  {
    std::array r1{Data{4}, Data{8}, Data{12}};
    std::array r2{Data{8}, Data{9}};

    // iterator overload
    {
      std::array<Data, 1> out;
      auto result = std::ranges::set_intersection(
          r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), [](const Data& x, const Data& y) {
            return x.data < y.data;
          });

      assert(std::ranges::equal(out, std::array{8}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
    }

    // range overload
    {
      std::array<Data, 1> out;
      auto result = std::ranges::set_intersection(r1, r2, out.data(), [](const Data& x, const Data& y) {
        return x.data < y.data;
      });

      assert(std::ranges::equal(out, std::array{8}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
    }

    // member pointer Comparator iterator overload
    {
      std::array<Data, 1> out;
      auto result =
          std::ranges::set_intersection(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), &Data::smallerThan);

      assert(std::ranges::equal(out, std::array{8}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
    }

    // member pointer Comparator range overload
    {
      std::array<Data, 1> out;
      auto result = std::ranges::set_intersection(r1, r2, out.data(), &Data::smallerThan);

      assert(std::ranges::equal(out, std::array{8}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
    }
  }

  // Test Projection
  {
    std::array r1{Data{1}, Data{3}, Data{5}};
    std::array r2{Data{2}, Data{3}, Data{5}};

    const auto proj = [](const Data& d) { return d.data; };

    // iterator overload
    {
      std::array<Data, 2> out;
      auto result = std::ranges::set_intersection(
          r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), std::ranges::less{}, proj, proj);

      assert(std::ranges::equal(out, std::array{3, 5}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
    }

    // range overload
    {
      std::array<Data, 2> out;
      auto result = std::ranges::set_intersection(r1, r2, out.data(), std::ranges::less{}, proj, proj);

      assert(std::ranges::equal(out, std::array{3, 5}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
    }

    // member pointer Projection iterator overload
    {
      std::array<Data, 2> out;
      auto result = std::ranges::set_intersection(
          r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), {}, &Data::data, &Data::data);

      assert(std::ranges::equal(out, std::array{3, 5}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
    }

    // member pointer Projection range overload
    {
      std::array<Data, 2> out;
      auto result = std::ranges::set_intersection(r1, r2, out.data(), std::ranges::less{}, &Data::data, &Data::data);

      assert(std::ranges::equal(out, std::array{3, 5}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.end());
    }
  }

  // Complexity: At most 2 * ((last1 - first1) + (last2 - first2)) - 1 comparisons and applications of each projection.
  {
    std::array<Data, 5> r1{{{1}, {3}, {5}, {7}, {9}}};
    std::array<Data, 5> r2{{{2}, {4}, {6}, {8}, {10}}};
    std::array<int, 0> expected{};

    const std::size_t maxOperation = 2 * (r1.size() + r2.size()) - 1;

    // iterator overload
    {
      std::array<Data, 0> out{};
      std::size_t numberOfComp  = 0;
      std::size_t numberOfProj1 = 0;
      std::size_t numberOfProj2 = 0;

      const auto comp = [&numberOfComp](int x, int y) {
        ++numberOfComp;
        return x < y;
      };

      const auto proj1 = [&numberOfProj1](const Data& d) {
        ++numberOfProj1;
        return d.data;
      };

      const auto proj2 = [&numberOfProj2](const Data& d) {
        ++numberOfProj2;
        return d.data;
      };

      std::ranges::set_intersection(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), comp, proj1, proj2);

      assert(std::ranges::equal(out, expected, {}, &Data::data));
      assert(numberOfComp < maxOperation);
      assert(numberOfProj1 < maxOperation);
      assert(numberOfProj2 < maxOperation);
    }

    // range overload
    {
      std::array<Data, 0> out{};
      std::size_t numberOfComp  = 0;
      std::size_t numberOfProj1 = 0;
      std::size_t numberOfProj2 = 0;

      const auto comp = [&numberOfComp](int x, int y) {
        ++numberOfComp;
        return x < y;
      };

      const auto proj1 = [&numberOfProj1](const Data& d) {
        ++numberOfProj1;
        return d.data;
      };

      const auto proj2 = [&numberOfProj2](const Data& d) {
        ++numberOfProj2;
        return d.data;
      };

      std::ranges::set_intersection(r1, r2, out.data(), comp, proj1, proj2);

      assert(std::ranges::equal(out, expected, {}, &Data::data));
      assert(numberOfComp < maxOperation);
      assert(numberOfProj1 < maxOperation);
      assert(numberOfProj2 < maxOperation);
    }
  }

  // Comparator convertible to bool
  {
    struct ConvertibleToBool {
      bool b;
      constexpr operator bool() const { return b; }
    };
    Data r1[] = {{3}, {4}};
    Data r2[] = {{3}, {4}, {5}};

    const auto comp = [](const Data& x, const Data& y) { return ConvertibleToBool{x.data < y.data}; };

    // iterator overload
    {
      std::array<Data, 2> out;
      std::ranges::set_intersection(r1, r1 + 2, r2, r2 + 3, out.data(), comp);
      assert(std::ranges::equal(out, std::array{3, 4}, {}, &Data::data));
    }

    // range overload
    {
      std::array<Data, 2> out;
      std::ranges::set_intersection(r1, r2, out.data(), comp);
      assert(std::ranges::equal(out, std::array{3, 4}, {}, &Data::data));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  // Cannot static_assert on the entire permutation test because it exceeds the constexpr execution step limit
  // due to the large number of combination of types of iterators (it is a 3-dimensional cartesian product)
  // Instead of having one single static_assert that tests all the combinations, in the runAllIteratorPermutationsTests 
  // function, it has lots of smaller static_assert and each of them test 2-dimensional cartesian product which is less 
  // than the step limit.
  runAllIteratorPermutationsTests();

  return 0;
}
