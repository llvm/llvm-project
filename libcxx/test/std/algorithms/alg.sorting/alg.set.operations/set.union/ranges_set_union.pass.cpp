//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <algorithm>

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          weakly_incrementable O, class Comp = ranges::less,
//          class Proj1 = identity, class Proj2 = identity>
//   requires mergeable<I1, I2, O, Comp, Proj1, Proj2>
//   constexpr set_union_result<I1, I2, O>
//     set_union(I1 first1, S1 last1, I2 first2, S2 last2, O result, Comp comp = {},
//               Proj1 proj1 = {}, Proj2 proj2 = {});                                               // Since C++20
//
// template<input_range R1, input_range R2, weakly_incrementable O,
//          class Comp = ranges::less, class Proj1 = identity, class Proj2 = identity>
//   requires mergeable<iterator_t<R1>, iterator_t<R2>, O, Comp, Proj1, Proj2>
//   constexpr set_union_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>, O>
//     set_union(R1&& r1, R2&& r2, O result, Comp comp = {},
//               Proj1 proj1 = {}, Proj2 proj2 = {});                                               // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "MoveOnly.h"
#include "test_iterators.h"
#include "../../sortable_helpers.h"

// Test iterator overload's constraints:
// =====================================
template <class InIter1 = int*, class Sent1 = int*, class InIter2 = int*, class Sent2 = int*, class OutIter = int*>
concept HasSetUnionIter =
    requires(InIter1&& inIter1, InIter2&& inIter2, OutIter&& outIter, Sent1&& sent1, Sent2&& sent2) {
      std::ranges::set_union(
          std::forward<InIter1>(inIter1),
          std::forward<Sent1>(sent1),
          std::forward<InIter2>(inIter2),
          std::forward<Sent2>(sent2),
          std::forward<OutIter>(outIter));
    };

static_assert(HasSetUnionIter<int*, int*, int*, int*, int*>);

// !std::input_iterator<I1>
static_assert(!HasSetUnionIter<InputIteratorNotDerivedFrom>);

// !std::sentinel_for<S1, I1>
static_assert(!HasSetUnionIter<int*, SentinelForNotSemiregular>);

// !std::input_iterator<I2>
static_assert(!HasSetUnionIter<int*, int*, InputIteratorNotDerivedFrom>);

// !std::sentinel_for<S2, I2>
static_assert(!HasSetUnionIter<int*, int*, int*, SentinelForNotSemiregular>);

// !std::weakly_incrementable<O>
static_assert(!HasSetUnionIter<int*, int*, int*, int*, WeaklyIncrementableNotMovable>);

// !std::mergeable<I1, I2, O, Comp, Proj1, Proj2>
static_assert(!HasSetUnionIter<MoveOnly*, MoveOnly*, MoveOnly*, MoveOnly*, MoveOnly*>);

// Test range overload's constraints:
// =====================================

template <class Range1, class Range2, class OutIter>
concept HasSetUnionRange =
    requires(Range1&& range1, Range2&& range2, OutIter&& outIter) {
      std::ranges::set_union(
          std::forward<Range1>(range1), std::forward<Range2>(range2), std::forward<OutIter>(outIter));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasSetUnionRange<R<int*>, R<int*>, int*>);

// !std::input_range<R2>
static_assert(!HasSetUnionRange<R<InputIteratorNotDerivedFrom>, R<int*>, int*>);

// !std::input_range<R2>
static_assert(!HasSetUnionRange<R<int*>, R<InputIteratorNotDerivedFrom>, int*>);

// !std::weakly_incrementable<O>
static_assert(!HasSetUnionRange<R<int*>, R<int*>, WeaklyIncrementableNotMovable >);

// !std::mergeable<iterator_t<R1>, iterator_t<R2>, O, Comp, Proj1, Proj2>
static_assert(!HasSetUnionRange<R<MoveOnly*>, R<MoveOnly*>, MoveOnly*>);

using std::ranges::set_union_result;

template <class In1, class In2, class Out, std::size_t N1, std::size_t N2, std::size_t N3>
constexpr void testSetUnionImpl(std::array<int, N1> in1, std::array<int, N2> in2, std::array<int, N3> expected) {
  // TODO: std::ranges::set_union calls std::ranges::copy
  // std::ranges::copy(contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>, contiguous_iterator<int*>) doesn't seem to work.
  // It seems that std::ranges::copy calls std::copy, which unwraps contiguous_iterator<int*> into int*,
  // and then it failed because there is no == between int* and sentinel_wrapper<contiguous_iterator<int*>>
  using Sent1 = std::conditional_t<std::contiguous_iterator<In1>, In1, sentinel_wrapper<In1>>;
  using Sent2 = std::conditional_t<std::contiguous_iterator<In2>, In2, sentinel_wrapper<In2>>;

  // iterator overload
  {
    std::array<int, N3> out;
    std::same_as<set_union_result<In1, In2, Out>> decltype(auto) result = std::ranges::set_union(
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
    std::same_as<set_union_result<In1, In2, Out>> decltype(auto) result =
        std::ranges::set_union(r1, r2, Out{out.data()});
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
    std::array expected{0, 1, 3, 5, 6, 7, 9, 10, 13, 15, 100};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }
  // range 2 shorter than range 1
  {
    std::array in1{2, 6, 8, 12, 15, 16};
    std::array in2{0, 2, 8};
    std::array expected{0, 2, 6, 8, 12, 15, 16};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 and range 2 has the same length but different elements
  {
    std::array in1{2, 6, 8, 12, 15, 16};
    std::array in2{0, 2, 8, 15, 17, 19};
    std::array expected{0, 2, 6, 8, 12, 15, 16, 17, 19};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 == range 2
  {
    std::array in1{0, 1, 2};
    std::array in2{0, 1, 2};
    std::array expected{0, 1, 2};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 is super set of range 2
  {
    std::array in1{8, 8, 10, 12, 13};
    std::array in2{8, 10};
    std::array expected{8, 8, 10, 12, 13};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 2 is super set of range 1
  {
    std::array in1{0, 1, 1};
    std::array in2{0, 1, 1, 2, 5};
    std::array expected{0, 1, 1, 2, 5};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 and range 2 have no elements in common
  {
    std::array in1{7, 7, 9, 12};
    std::array in2{1, 5, 5, 8, 10};
    std::array expected{1, 5, 5, 7, 7, 8, 9, 10, 12};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 and range 2 have duplicated equal elements
  {
    std::array in1{7, 7, 9, 12};
    std::array in2{7, 7, 7, 13};
    std::array expected{7, 7, 7, 9, 12, 13};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 is empty
  {
    std::array<int, 0> in1{};
    std::array in2{3, 4, 5};
    std::array expected{3, 4, 5};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 2 is empty
  {
    std::array in1{3, 4, 5};
    std::array<int, 0> in2{};
    std::array expected{3, 4, 5};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // both ranges are empty
  {
    std::array<int, 0> in1{};
    std::array<int, 0> in2{};
    std::array<int, 0> expected{};
    testSetUnionImpl<In1, In2, Out>(in1, in2, expected);
  }

  // check that ranges::dangling is returned for non-borrowed_range
  {
    std::array r1{3, 6, 7, 9};
    int r2[] = {2, 3, 4, 5, 6};
    std::array<int, 7> out;
    std::same_as<set_union_result<std::ranges::dangling, int*, int*>> decltype(auto) result =
        std::ranges::set_union(NonBorrowedRange<In1>{r1.data(), r1.size()}, r2, out.data());
    assert(base(result.in2) == r2 + 5);
    assert(base(result.out) == out.data() + out.size());
    assert(std::ranges::equal(out, std::array{2, 3, 4, 5, 6, 7, 9}));
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
      std::array<TracedCopy, 6> out;
      auto result = std::ranges::set_union(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data());

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.data() + out.size());
      assert(std::ranges::equal(out, std::array<TracedCopy, 6>{1, 3, 5, 8, 15, 16}));

      assert(std::ranges::all_of(out, &TracedCopy::copiedOnce));
    }

    // range overload
    {
      std::array<TracedCopy, 6> out;
      auto result = std::ranges::set_union(r1, r2, out.data());

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.data() + out.size());
      assert(std::ranges::equal(out, std::array<TracedCopy, 6>{1, 3, 5, 8, 15, 16}));

      assert(std::ranges::all_of(out, &TracedCopy::copiedOnce));
    }
  }

  struct IntAndOrder {
    int data;
    int order;

    constexpr auto operator==(const IntAndOrder& o) const { return data == o.data; }
    constexpr auto operator<=>(const IntAndOrder& o) const { return data <=> o.data; }
  };

  // Stable ([algorithm.stable]). If [first1, last1) contains m elements that are
  // equivalent to each other and [first2, last2) contains n elements that are
  // equivalent to them, then all m elements from the first range are copied to the
  // output range, in order, and then the final max(n-m,0) elements from the second
  // range are copied to the output range, in order.
  {
    std::array<IntAndOrder, 3> r1{{{0, 0}, {0, 1}, {0, 2}}};
    std::array<IntAndOrder, 5> r2{{{0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}}};

    // iterator overload
    {
      std::array<IntAndOrder, 5> out;
      std::ranges::set_union(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data());

      assert(std::ranges::equal(out, std::array{0, 0, 0, 0, 0}, {}, &IntAndOrder::data));
      assert(std::ranges::equal(out, std::array{0, 1, 2, 6, 7}, {}, &IntAndOrder::order));
    }

    // range overload
    {
      std::array<IntAndOrder, 5> out;
      std::ranges::set_union(r1, r2, out.data());

      assert(std::ranges::equal(out, std::array{0, 0, 0, 0, 0}, {}, &IntAndOrder::data));
      assert(std::ranges::equal(out, std::array{0, 1, 2, 6, 7}, {}, &IntAndOrder::order));
    }
  }

  struct Data {
    int data;
  };

  // Test custom comparator
  {
    std::array r1{Data{4}, Data{8}, Data{12}};
    std::array r2{Data{8}, Data{9}};

    // iterator overload
    {
      std::array<Data, 4> out;
      auto result = std::ranges::set_union(
          r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), [](const Data& x, const Data& y) {
            return x.data < y.data;
          });

      assert(std::ranges::equal(out, std::array{4, 8, 9, 12}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.data() + out.size());
    }

    // range overload
    {
      std::array<Data, 4> out;
      auto result = std::ranges::set_union(r1, r2, out.data(), [](const Data& x, const Data& y) {
        return x.data < y.data;
      });

      assert(std::ranges::equal(out, std::array{4, 8, 9, 12}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.data() + out.size());
    }
  }

  // Test Projection
  {
    std::array r1{Data{1}, Data{3}, Data{5}};
    std::array r2{Data{2}, Data{3}, Data{5}};

    const auto proj = [](const Data& d) { return d.data; };

    // iterator overload
    {
      std::array<Data, 4> out;
      auto result = std::ranges::set_union(
          r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), std::ranges::less{}, proj, proj);

      assert(std::ranges::equal(out, std::array{1, 2, 3, 5}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.data() + out.size());
    }

    // range overload
    {
      std::array<Data, 4> out;
      auto result = std::ranges::set_union(r1, r2, out.data(), std::ranges::less{}, proj, proj);

      assert(std::ranges::equal(out, std::array{1, 2, 3, 5}, {}, &Data::data));

      assert(result.in1 == r1.end());
      assert(result.in2 == r2.end());
      assert(result.out == out.data() + out.size());
    }
  }

  // Complexity: At most 2 * ((last1 - first1) + (last2 - first2)) - 1 comparisons and applications of each projection.
  {
    struct CompProjs {
      std::size_t numberOfComp  = 0;
      std::size_t numberOfProj1 = 0;
      std::size_t numberOfProj2 = 0;

      constexpr auto comp() {
        return [this](int x, int y) {
          ++numberOfComp;
          return x < y;
        };
      }

      constexpr auto proj1() {
        return [this](const Data& d) {
          ++numberOfProj1;
          return d.data;
        };
      }

      constexpr auto proj2() {
        return [this](const Data& d) {
          ++numberOfProj2;
          return d.data;
        };
      }
    };

    std::array<Data, 3> r1{{{0}, {1}, {2}}};
    std::array<Data, 4> r2{{{0}, {2}, {2}, {5}}};
    std::array expected{0, 1, 2, 2, 5};

    const std::size_t maxOperation = 2 * (r1.size() + r2.size()) - 1;

    // iterator overload
    {
      std::array<Data, 5> out;
      CompProjs compProjs{};

      std::ranges::set_union(
          r1.begin(),
          r1.end(),
          r2.begin(),
          r2.end(),
          out.data(),
          compProjs.comp(),
          compProjs.proj1(),
          compProjs.proj2());

      assert(std::ranges::equal(out, expected, {}, &Data::data));
      assert(compProjs.numberOfComp < maxOperation);
      assert(compProjs.numberOfProj1 < maxOperation);
      assert(compProjs.numberOfProj2 < maxOperation);
    }

    // range overload
    {
      std::array<Data, 5> out;
      CompProjs compProjs{};

      std::ranges::set_union(r1, r2, out.data(), compProjs.comp(), compProjs.proj1(), compProjs.proj2());

      assert(std::ranges::equal(out, expected, {}, &Data::data));
      assert(compProjs.numberOfComp < maxOperation);
      assert(compProjs.numberOfProj1 < maxOperation);
      assert(compProjs.numberOfProj2 < maxOperation);
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
