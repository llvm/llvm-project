
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
//   constexpr set_difference_result<I1, O>
//     set_difference(I1 first1, S1 last1, I2 first2, S2 last2, O result,
//                    Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});                           // since C++20

// template<input_range R1, input_range R2, weakly_incrementable O,
//          class Comp = ranges::less, class Proj1 = identity, class Proj2 = identity>
//   requires mergeable<iterator_t<R1>, iterator_t<R2>, O, Comp, Proj1, Proj2>
//   constexpr set_difference_result<borrowed_iterator_t<R1>, O>
//     set_difference(R1&& r1, R2&& r2, O result,
//                    Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});                           // since C++20

#include <algorithm>
#include <array>
#include <concepts>

#include "almost_satisfies_types.h"
#include "MoveOnly.h"
#include "test_iterators.h"
#include "../../sortable_helpers.h"

// Test iterator overload's constraints:
// =====================================
template <
    class InIter1,
    class InIter2,
    class OutIter,
    class Sent1 = sentinel_wrapper<InIter1>,
    class Sent2 = sentinel_wrapper<InIter2>>
concept HasSetDifferenceIter =
    requires(InIter1&& inIter1, InIter2&& inIter2, OutIter&& outIter, Sent1&& sent1, Sent2&& sent2) {
      std::ranges::set_difference(
          std::forward<InIter1>(inIter1),
          std::forward<Sent1>(sent1),
          std::forward<InIter2>(inIter2),
          std::forward<Sent2>(sent2),
          std::forward<OutIter>(outIter));
    };

static_assert(HasSetDifferenceIter<int*, int*, int*, int*, int*>);

// !std::input_iterator<I1>
static_assert(!HasSetDifferenceIter<InputIteratorNotDerivedFrom, int*, int*>);

// !std::sentinel_for<S1, I1>
static_assert(!HasSetDifferenceIter<int*, int*, int*, SentinelForNotSemiregular>);

// !std::input_iterator<I2>
static_assert(!HasSetDifferenceIter<int*, InputIteratorNotDerivedFrom, int*>);

// !std::sentinel_for<S2, I2>
static_assert(!HasSetDifferenceIter<int*, int*, int*, int*, SentinelForNotSemiregular>);

// !std::weakly_incrementable<O>
static_assert(!HasSetDifferenceIter<int*, int*, WeaklyIncrementableNotMovable>);

// !std::mergeable<I1, I2, O, Comp, Proj1, Proj2>
static_assert(!HasSetDifferenceIter<MoveOnly*, MoveOnly*, MoveOnly*, MoveOnly*, MoveOnly*>);

// Test range overload's constraints:
// =====================================

template <class Range1, class Range2, class OutIter>
concept HasSetDifferenceRange =
    requires(Range1&& range1, Range2&& range2, OutIter&& outIter) {
      std::ranges::set_difference(
          std::forward<Range1>(range1), std::forward<Range2>(range2), std::forward<OutIter>(outIter));
    };

static_assert(HasSetDifferenceRange<UncheckedRange<int*>, UncheckedRange<int*>, int*>);

// !std::input_range<R2>
static_assert(!HasSetDifferenceRange<UncheckedRange<InputIteratorNotDerivedFrom>, UncheckedRange<int*>, int*>);

// !std::input_range<R2>
static_assert(!HasSetDifferenceRange<UncheckedRange<int*>, UncheckedRange<InputIteratorNotDerivedFrom>, int*>);

// !std::weakly_incrementable<O>
static_assert(!HasSetDifferenceRange<UncheckedRange<int*>, UncheckedRange<int*>, WeaklyIncrementableNotMovable >);

// !std::mergeable<iterator_t<R1>, iterator_t<R2>, O, Comp, Proj1, Proj2>
static_assert(!HasSetDifferenceRange< UncheckedRange<MoveOnly*>, UncheckedRange<MoveOnly*>, MoveOnly*>);

using std::ranges::set_difference_result;

template <class In1, class In2, class Out, std::size_t N1, std::size_t N2, std::size_t N3>
constexpr void testSetDifferenceImpl(std::array<int, N1> in1, std::array<int, N2> in2, std::array<int, N3> expected) {
  // TODO: std::ranges::set_difference calls std::ranges::copy
  // std::ranges::copy(contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>, contiguous_iterator<int*>) doesn't seem to work.
  // It seems that std::ranges::copy calls std::copy, which unwraps contiguous_iterator<int*> into int*,
  // and then it failed because there is no == between int* and sentinel_wrapper<contiguous_iterator<int*>>
  using Sent1 = std::conditional_t<std::contiguous_iterator<In1>, In1, sentinel_wrapper<In1>>;
  using Sent2 = std::conditional_t<std::contiguous_iterator<In2>, In2, sentinel_wrapper<In2>>;

  // iterator overload
  {
    std::array<int, N3> out;
    std::same_as<set_difference_result<In1, Out>> decltype(auto) result = std::ranges::set_difference(
        In1{in1.data()},
        Sent1{In1{in1.data() + in1.size()}},
        In2{in2.data()},
        Sent2{In2{in2.data() + in2.size()}},
        Out{out.data()});
    assert(std::ranges::equal(out, expected));

    assert(base(result.in) == in1.data() + in1.size());
    assert(base(result.out) == out.data() + out.size());
  }

  // range overload
  {
    std::array<int, N3> out;
    std::ranges::subrange r1{In1{in1.data()}, Sent1{In1{in1.data() + in1.size()}}};
    std::ranges::subrange r2{In2{in2.data()}, Sent2{In2{in2.data() + in2.size()}}};
    std::same_as<set_difference_result<In1, Out>> decltype(auto) result =
        std::ranges::set_difference(r1, r2, Out{out.data()});
    assert(std::ranges::equal(out, expected));

    assert(base(result.in) == in1.data() + in1.size());
    assert(base(result.out) == out.data() + out.size());
  }
}

template <class In1, class In2, class Out>
constexpr void testImpl() {
  // range 1 shorter than range2
  {
    std::array in1{0, 1, 5, 6, 9, 10};
    std::array in2{3, 6, 7, 9, 13, 15, 100};
    std::array expected{0, 1, 5, 10};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 2 shorter than range 1
  {
    std::array in1{2, 6, 8, 12, 15, 16};
    std::array in2{0, 2, 8};
    std::array expected{6, 12, 15, 16};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 and range 2 has the same length but different elements
  {
    std::array in1{2, 6, 8, 12, 15, 16};
    std::array in2{0, 2, 8, 15, 17, 19};
    std::array expected{6, 12, 16};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 == range 2
  {
    std::array in1{0, 1, 2};
    std::array in2{0, 1, 2};
    std::array<int, 0> expected{};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 is super set of range 2
  {
    std::array in1{8, 8, 10, 12, 13};
    std::array in2{8, 10};
    std::array expected{8, 12, 13};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 2 is super set of range 1
  {
    std::array in1{0, 1, 1};
    std::array in2{0, 1, 1, 2, 5};
    std::array<int, 0> expected{};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 1 is empty
  {
    std::array<int, 0> in1{};
    std::array in2{3, 4, 5};
    std::array<int, 0> expected{};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // range 2 is empty
  {
    std::array in1{3, 4, 5};
    std::array<int, 0> in2{};
    std::array expected{3, 4, 5};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // both ranges are empty
  {
    std::array<int, 0> in1{};
    std::array<int, 0> in2{};
    std::array<int, 0> expected{};
    testSetDifferenceImpl<In1, In2, Out>(in1, in2, expected);
  }

  // check that ranges::dangling is returned for non-borrowed_range
  {
    struct NonBorrowedRange {
      int* data_;
      size_t size_;

      // TODO: std::ranges::set_difference calls std::__copy
      // std::ranges::copy(contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>, contiguous_iterator<int*>) doesn't seem to work.
      // It seems that std::ranges::copy calls std::copy, which unwraps contiguous_iterator<int*> into int*,
      // and then it failed because there is no == between int* and sentinel_wrapper<contiguous_iterator<int*>>
      using Sent = std::conditional_t<std::contiguous_iterator<In2>, In2, sentinel_wrapper<In2>>;

      constexpr NonBorrowedRange(int* d, size_t s) : data_{d}, size_{s} {}

      constexpr In2 begin() const { return In2{data_}; };
      constexpr Sent end() const { return Sent{In2{data_ + size_}}; };
    };

    std::array r1{3, 6, 7, 9};
    std::array r2{2, 3, 4, 5, 6};
    std::array<int, 2> out;
    std::same_as<set_difference_result<std::ranges::dangling, int*>> decltype(auto) result =
        std::ranges::set_difference(NonBorrowedRange{r1.data(), r1.size()}, r2, out.data());
    assert(base(result.out) == out.data() + out.size());
    assert(std::ranges::equal(out, std::array{7, 9}));
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
      std::array<TracedCopy, 3> out;
      auto result = std::ranges::set_difference(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data());

      assert(result.in == r1.end());
      assert(result.out == out.end());
      assert(std::ranges::equal(out, std::array<TracedCopy, 3>{5, 15, 16}));

      assert(std::ranges::all_of(out, &TracedCopy::copiedOnce));
    }

    // range overload
    {
      std::array<TracedCopy, 3> out;
      auto result = std::ranges::set_difference(r1, r2, out.data());

      assert(result.in == r1.end());
      assert(result.out == out.end());
      assert(std::ranges::equal(out, std::array<TracedCopy, 3>{5, 15, 16}));

      assert(std::ranges::all_of(out, &TracedCopy::copiedOnce));
    }
  }

  struct IntAndOrder {
    int data;
    int order;

    constexpr auto operator==(const IntAndOrder& o) const { return data == o.data; }
    constexpr auto operator<=>(const IntAndOrder& o) const { return data <=> o.data; }
  };

  // equal elements should be copied in the original order and only m-n elements are copied
  {
    std::array<IntAndOrder, 5> r1{{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}}};
    std::array<IntAndOrder, 3> r2{{{0, 1}, {0, 2}, {0, 3}}};

    // iterator overload
    {
      std::array<IntAndOrder, 2> out;
      std::ranges::set_difference(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data());

      assert(std::ranges::equal(out, std::array{0, 0}, {}, &IntAndOrder::data));
      // m-n elements are copied in order
      assert(std::ranges::equal(out, std::array{3, 4}, {}, &IntAndOrder::order));
    }

    // range overload
    {
      std::array<IntAndOrder, 2> out;
      std::ranges::set_difference(r1, r2, out.data());

      assert(std::ranges::equal(out, std::array{0, 0}, {}, &IntAndOrder::data));
      // ID should be in their original order
      assert(std::ranges::equal(out, std::array{3, 4}, {}, &IntAndOrder::order));
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
    using Iter1 = std::array<Data, 3>::iterator;

    // iterator overload
    {
      std::array<Data, 2> out;
      std::same_as<set_difference_result<Iter1, Data*>> decltype(auto) result = std::ranges::set_difference(
          r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), [](const Data& x, const Data& y) {
            return x.data < y.data;
          });

      assert(std::ranges::equal(out, std::array{4, 12}, {}, &Data::data));

      assert(result.in == r1.end());
      assert(result.out == out.end());
    }

    // range overload
    {
      std::array<Data, 2> out;
      std::same_as<set_difference_result<Iter1, Data*>> decltype(auto) result =
          std::ranges::set_difference(r1, r2, out.data(), [](const Data& x, const Data& y) { return x.data < y.data; });

      assert(std::ranges::equal(out, std::array{4, 12}, {}, &Data::data));

      assert(result.in == r1.end());
      assert(result.out == out.end());
    }

    // member pointer Comparator iterator overload
    {
      std::array<Data, 2> out;
      std::same_as<set_difference_result<Iter1, Data*>> decltype(auto) result =
          std::ranges::set_difference(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), &Data::smallerThan);

      assert(std::ranges::equal(out, std::array{4, 12}, {}, &Data::data));

      assert(result.in == r1.end());
      assert(result.out == out.end());
    }

    // member pointer Comparator range overload
    {
      std::array<Data, 2> out;
      std::same_as<set_difference_result<Iter1, Data*>> decltype(auto) result =
          std::ranges::set_difference(r1, r2, out.data(), &Data::smallerThan);

      assert(std::ranges::equal(out, std::array{4, 12}, {}, &Data::data));

      assert(result.in == r1.end());
      assert(result.out == out.end());
    }
  }

  // Test Projection
  {
    std::array r1{Data{1}, Data{3}, Data{5}};
    std::array r2{Data{2}, Data{3}, Data{4}};
    using Iter1 = std::array<Data, 3>::iterator;

    const auto proj = [](const Data& d) { return d.data; };

    // iterator overload
    {
      std::array<Data, 2> out;
      std::same_as<set_difference_result<Iter1, Data*>> decltype(auto) result = std::ranges::set_difference(
          r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), std::ranges::less{}, proj, proj);

      assert(std::ranges::equal(out, std::array{1, 5}, {}, &Data::data));

      assert(result.in == r1.end());
      assert(result.out == out.end());
    }

    // range overload
    {
      std::array<Data, 2> out;
      std::same_as<set_difference_result<Iter1, Data*>> decltype(auto) result =
          std::ranges::set_difference(r1, r2, out.data(), std::ranges::less{}, proj, proj);

      assert(std::ranges::equal(out, std::array{1, 5}, {}, &Data::data));

      assert(result.in == r1.end());
      assert(result.out == out.end());
    }

    // member pointer Projection iterator overload
    {
      std::array<Data, 2> out;
      std::same_as<set_difference_result<Iter1, Data*>> decltype(auto) result = std::ranges::set_difference(
          r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), {}, &Data::data, &Data::data);

      assert(std::ranges::equal(out, std::array{1, 5}, {}, &Data::data));

      assert(result.in == r1.end());
      assert(result.out == out.end());
    }

    // member pointer Projection range overload
    {
      std::array<Data, 2> out;
      std::same_as<set_difference_result<Iter1, Data*>> decltype(auto) result =
          std::ranges::set_difference(r1, r2, out.data(), std::ranges::less{}, &Data::data, &Data::data);

      assert(std::ranges::equal(out, std::array{1, 5}, {}, &Data::data));

      assert(result.in == r1.end());
      assert(result.out == out.end());
    }
  }

  // Complexity: At most 2 * ((last1 - first1) + (last2 - first2)) - 1 comparisons and applications of each projection.
  {
    std::array<Data, 3> r1{{{4}, {8}, {12}}};
    std::array<Data, 9> r2{{{1}, {2}, {3}, {5}, {6}, {7}, {9}, {10}, {11}}};
    std::array expected{4, 8, 12};

    const std::size_t maxOperation = 2 * (r1.size() + r2.size()) - 1;

    // iterator overload
    {
      std::array<Data, 3> out;
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

      std::ranges::set_difference(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), comp, proj1, proj2);

      assert(std::ranges::equal(out, expected, {}, &Data::data));
      assert(numberOfComp < maxOperation);
      assert(numberOfProj1 < maxOperation);
      assert(numberOfProj2 < maxOperation);
    }

    // range overload
    {
      std::array<Data, 3> out;
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

      std::ranges::set_difference(r1, r2, out.data(), comp, proj1, proj2);

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
    Data r1[] = {{2}, {4}};
    Data r2[] = {{3}, {4}, {5}};

    const auto comp = [](const Data& x, const Data& y) { return ConvertibleToBool{x.data < y.data}; };

    // iterator overload
    {
      std::array<Data, 1> out;
      std::ranges::set_difference(r1, r1 + 2, r2, r2 + 3, out.data(), comp);
      assert(std::ranges::equal(out, std::array{2}, {}, &Data::data));
    }

    // range overload
    {
      std::array<Data, 1> out;
      std::ranges::set_difference(r1, r2, out.data(), comp);
      assert(std::ranges::equal(out, std::array{2}, {}, &Data::data));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  runAllIteratorPermutationsTests();

  return 0;
}
