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
//          class Proj = identity, indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires indirectly_copyable<I, O>
//   constexpr remove_copy_if_result<I, O>
//     remove_copy_if(I first, S last, O result, Pred pred, Proj proj = {});                        // Since C++20
//
// template<input_range R, weakly_incrementable O, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires indirectly_copyable<iterator_t<R>, O>
//   constexpr remove_copy_if_result<borrowed_iterator_t<R>, O>
//     remove_copy_if(R&& r, O result, Pred pred, Proj proj = {});                                  // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>

#include "almost_satisfies_types.h"
#include "counting_predicates.h"
#include "counting_projection.h"
#include "test_iterators.h"

struct AlwaysTrue {
  constexpr bool operator()(auto&&...) const { return true; }
};

template <
    class I,
    class S    = sentinel_wrapper<std::decay_t<I>>,
    class O    = int*,
    class Pred = AlwaysTrue>
concept HasRemoveCopyIfIter =
  requires(I&& iter, S&& sent, O&& out, Pred&& pred) {
    std::ranges::remove_copy_if(std::forward<I>(iter), std::forward<S>(sent),
                                std::forward<O>(out), std::forward<Pred>(pred));
};

static_assert(HasRemoveCopyIfIter<int*, int*, int*>);

// !input_iterator<I>
static_assert(!HasRemoveCopyIfIter<InputIteratorNotDerivedFrom>);
static_assert(!HasRemoveCopyIfIter<cpp20_output_iterator<int*>>);

// !sentinel_for<S, I>
static_assert(!HasRemoveCopyIfIter<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasRemoveCopyIfIter<int*, SentinelForNotSemiregular>);

// !weakly_incrementable<O>
static_assert(!HasRemoveCopyIfIter<int*, int*, WeaklyIncrementableNotMovable>);

// !indirect_unary_predicate<Pred, projected<I, Proj>>
static_assert(!HasRemoveCopyIfIter<int*, int*, int*, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasRemoveCopyIfIter<int*, int*, int*, IndirectUnaryPredicateNotCopyConstructible>);

// !indirectly_copyable<I, O>
static_assert(!HasRemoveCopyIfIter<int*, int*, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasRemoveCopyIfIter<const int*, const int*, const int*>);

template < class R, class O = int*, class Pred = AlwaysTrue, class Proj = std::identity>
concept HasRemoveCopyIfRange =
    requires(R&& r, O&& out, Pred&& pred, Proj&& proj) {
      std::ranges::remove_copy_if(
          std::forward<R>(r), std::forward<O>(out), std::forward<Pred>(pred), std::forward<Proj>(proj));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasRemoveCopyIfRange<R<int*>>);

// !input_range<R>
static_assert(!HasRemoveCopyIfRange<R<InputIteratorNotDerivedFrom>>);
static_assert(!HasRemoveCopyIfRange<R<cpp20_output_iterator<int*>>>);

// !weakly_incrementable<O>
static_assert(!HasRemoveCopyIfRange<R<int*>, WeaklyIncrementableNotMovable>);

// !indirect_unary_predicate<Pred, projected<iterator_t<R>, Proj>>
static_assert(!HasRemoveCopyIfRange<R<int*>, int*, IndirectUnaryPredicateNotPredicate>);
static_assert(!HasRemoveCopyIfRange<R<int*>, int*, IndirectUnaryPredicateNotCopyConstructible>);

// !indirectly_copyable<iterator_t<R>, O>
static_assert(!HasRemoveCopyIfRange<R<int*>, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasRemoveCopyIfRange<R<const int*>, const int*>);

template <class InIter, class OutIter, template <class> class SentWrapper, std::size_t N1, std::size_t N2, class Pred>
constexpr void testRemoveCopyIfImpl(std::array<int, N1> in, std::array<int, N2> expected, Pred pred) {
  using Sent = SentWrapper<InIter>;
  using Result = std::ranges::remove_copy_if_result<InIter, OutIter>;

  // iterator overload
  {
    std::array<int, N2> out;
    std::same_as<Result> decltype(auto) result =
        std::ranges::remove_copy_if(InIter{in.data()}, Sent{InIter{in.data() + in.size()}}, OutIter{out.data()}, pred);
    assert(std::ranges::equal(out, expected));
    assert(base(result.in) == in.data() + in.size());
    assert(base(result.out) == out.data() + out.size());
  }

  // range overload
  {
    std::array<int, N2> out;
    std::ranges::subrange r{InIter{in.data()}, Sent{InIter{in.data() + in.size()}}};
    std::same_as<Result> decltype(auto) result =
        std::ranges::remove_copy_if(r, OutIter{out.data()}, pred);
    assert(std::ranges::equal(out, expected));
    assert(base(result.in) == in.data() + in.size());
    assert(base(result.out) == out.data() + out.size());
  }
}

template <class InIter, class OutIter, template <class> class SentWrapper>
constexpr void testImpl() {
  // remove multiple elements
  {
    std::array in{1, 2, 3, 2, 1};
    std::array expected{1, 3, 1};
    auto pred = [](int i) { return i == 2; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }

  // remove single elements
  {
    std::array in{1, 2, 3, 2, 1};
    std::array expected{1, 2, 2, 1};
    auto pred = [](int i) { return i == 3; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }

  // nothing removed
  {
    std::array in{1, 2, 3, 2, 1};
    std::array expected{1, 2, 3, 2, 1};
    auto pred = [](int) { return false; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }

  // all removed
  {
    std::array in{1, 2, 3, 2, 1};
    std::array<int, 0> expected{};
    auto pred = [](int) { return true; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }

  // remove first
  {
    std::array in{1, 2, 3, 2};
    std::array expected{2, 3, 2};
    auto pred = [](int i) { return i < 2; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }

  // remove last
  {
    std::array in{1, 2, 3, 2, 5};
    std::array expected{1, 2, 3, 2};
    auto pred = [](int i) { return i > 3; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }

  // stable
  {
    std::array in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::array expected{6, 7, 8, 9, 10};
    auto pred = [](int i) { return i < 6; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }

  // empty range
  {
    std::array<int, 0> in{};
    std::array<int, 0> expected{};
    auto pred = [](int) { return false; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }

  // one element range
  {
    std::array in{1};
    std::array<int, 0> expected{};
    auto pred = [](int i) { return i == 1; };
    testRemoveCopyIfImpl<InIter, OutIter, SentWrapper>(in, expected, pred);
  }
}

template <class OutIter, template <class> class SentWrapper>
constexpr void withAllPermutationsOfInIter() {
  testImpl<cpp20_input_iterator<int*>, OutIter, sentinel_wrapper>();
  testImpl<forward_iterator<int*>, OutIter, SentWrapper>();
  testImpl<bidirectional_iterator<int*>, OutIter, SentWrapper>();
  testImpl<random_access_iterator<int*>, OutIter, SentWrapper>();
  testImpl<contiguous_iterator<int*>, OutIter, SentWrapper>();
  testImpl<int*, OutIter, SentWrapper>();
}

template <template <class> class SentWrapper>
constexpr void withAllPermutationsOfInIterOutIter() {
  withAllPermutationsOfInIter<cpp20_output_iterator<int*>, SentWrapper>();
  withAllPermutationsOfInIter<int*, SentWrapper>();
}

constexpr bool test() {
  withAllPermutationsOfInIterOutIter<std::type_identity_t>();
  withAllPermutationsOfInIterOutIter<sentinel_wrapper>();

  // Test custom projection
  {
    struct Data {
      int data;
    };

    std::array in{Data{4}, Data{8}, Data{12}, Data{12}};
    std::array expected{Data{4}, Data{12}, Data{12}};

    const auto proj = &Data::data;
    const auto pred = [](int i) { return i == 8; };

    const auto equals = [](const Data& x, const Data& y) { return x.data == y.data; };
    // iterator overload
    {
      std::array<Data, 3> out;
      auto result = std::ranges::remove_copy_if(in.begin(), in.end(), out.begin(), pred, proj);
      assert(std::ranges::equal(out, expected, equals));
      assert(result.in == in.end());
      assert(result.out == out.end());
    }

    // range overload
    {
      std::array<Data, 3> out;
      auto result = std::ranges::remove_copy_if(in, out.begin(), pred, proj);
      assert(std::ranges::equal(out, expected, equals));
      assert(result.in == in.end());
      assert(result.out == out.end());
    }
  }

  // Complexity: Exactly last - first applications of the corresponding predicate and any projection.
  {
    std::array in{4, 4, 5, 6};
    std::array expected{5, 6};

    const auto pred = [](int i) { return i == 4; };

    // iterator overload
    {
      int numberOfPred = 0;
      int numberOfProj = 0;
      std::array<int, 2> out;
      std::ranges::remove_copy_if(
          in.begin(), in.end(), out.begin(), counting_predicate(pred, numberOfPred), counting_projection(numberOfProj));

      assert(numberOfPred == static_cast<int>(in.size()));
      assert(numberOfProj == static_cast<int>(in.size()));

      assert(std::ranges::equal(out, expected));
    }

    // range overload
    {
      int numberOfPred = 0;
      int numberOfProj = 0;
      std::array<int, 2> out;
      std::ranges::remove_copy_if(
          in, out.begin(), counting_predicate(pred, numberOfPred), counting_projection(numberOfProj));
      assert(numberOfPred == static_cast<int>(in.size()));
      assert(numberOfProj == static_cast<int>(in.size()));
      assert(std::ranges::equal(out, expected));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
