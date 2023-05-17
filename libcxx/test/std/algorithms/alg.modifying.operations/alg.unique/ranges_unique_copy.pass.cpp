//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O, class Proj = identity,
//          indirect_equivalence_relation<projected<I, Proj>> C = ranges::equal_to>
//   requires indirectly_copyable<I, O> &&
//            (forward_iterator<I> ||
//             (input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>) ||
//             indirectly_copyable_storable<I, O>)
//   constexpr unique_copy_result<I, O>
//     unique_copy(I first, S last, O result, C comp = {}, Proj proj = {});                         // Since C++20
//
// template<input_range R, weakly_incrementable O, class Proj = identity,
//          indirect_equivalence_relation<projected<iterator_t<R>, Proj>> C = ranges::equal_to>
//   requires indirectly_copyable<iterator_t<R>, O> &&
//            (forward_iterator<iterator_t<R>> ||
//             (input_iterator<O> && same_as<range_value_t<R>, iter_value_t<O>>) ||
//             indirectly_copyable_storable<iterator_t<R>, O>)
//   constexpr unique_copy_result<borrowed_iterator_t<R>, O>
//     unique_copy(R&& r, O result, C comp = {}, Proj proj = {});                                   // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "counting_predicates.h"
#include "counting_projection.h"
#include "MoveOnly.h"
#include "test_iterators.h"

template <
    class InIter  = int*,
    class Sent    = int*,
    class OutIter = int*,
    class Comp    = std::ranges::equal_to,
    class Proj    = std::identity>
concept HasUniqueCopyIter =
    requires(InIter&& in, Sent&& sent, OutIter&& out, Comp&& comp, Proj&& proj) {
      std::ranges::unique_copy(
          std::forward<InIter>(in),
          std::forward<Sent>(sent),
          std::forward<OutIter>(out),
          std::forward<Comp>(comp),
          std::forward<Proj>(proj));
    };

static_assert(HasUniqueCopyIter<int*, int*, int*>);

// !input_iterator<I>
static_assert(!HasUniqueCopyIter<InputIteratorNotDerivedFrom, sentinel_wrapper<InputIteratorNotDerivedFrom>>);

// !sentinel_for<S, I>
static_assert(!HasUniqueCopyIter<int*, SentinelForNotSemiregular>);

// !weakly_incrementable<O>
static_assert(!HasUniqueCopyIter<int*, int*, WeaklyIncrementableNotMovable>);

// !indirect_equivalence_relation<Comp, projected<I, Proj>>
static_assert(!HasUniqueCopyIter<int*, int*, int*, ComparatorNotCopyable<int>>);

// !indirectly_copyable<I, O>
static_assert(!HasUniqueCopyIter<const int*, const int*, const int*>);

// forward_iterator<I>
// !(input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>)
// !indirectly_copyable_storable<I, O>
struct AssignableFromMoveOnly {
  int data;
  constexpr AssignableFromMoveOnly& operator=(MoveOnly const& m) {
    data = m.get();
    return *this;
  }
};
static_assert(HasUniqueCopyIter<MoveOnly*, MoveOnly*, AssignableFromMoveOnly*>);
// because:
static_assert(std::forward_iterator<MoveOnly*>);
static_assert(!std::same_as<std::iter_value_t<MoveOnly*>, std::iter_value_t<AssignableFromMoveOnly*>>);
static_assert(!std::indirectly_copyable_storable<MoveOnly*, AssignableFromMoveOnly*>);

// !forward_iterator<I>
// (input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>)
// !indirectly_copyable_storable<I, O>
struct CopyAssignableNotCopyConstructible {
  int data;
  constexpr CopyAssignableNotCopyConstructible(int i = 0) : data(i) {}
  CopyAssignableNotCopyConstructible(const CopyAssignableNotCopyConstructible&)            = delete;
  CopyAssignableNotCopyConstructible& operator=(const CopyAssignableNotCopyConstructible&) = default;
  friend constexpr bool
  operator==(CopyAssignableNotCopyConstructible const&, CopyAssignableNotCopyConstructible const&) = default;
};

using InputAndOutputIterator = cpp17_input_iterator<CopyAssignableNotCopyConstructible*>;
static_assert(std::input_iterator<InputAndOutputIterator>);
static_assert(std::output_iterator<InputAndOutputIterator, CopyAssignableNotCopyConstructible>);

static_assert(
    HasUniqueCopyIter<
        cpp20_input_iterator<CopyAssignableNotCopyConstructible*>,
        sentinel_wrapper<cpp20_input_iterator<CopyAssignableNotCopyConstructible*>>,
        InputAndOutputIterator>);
// because:
static_assert(!std::forward_iterator< cpp20_input_iterator<CopyAssignableNotCopyConstructible*>>);
static_assert(
    std::input_iterator<InputAndOutputIterator> &&
    std::same_as<std::iter_value_t<cpp20_input_iterator<CopyAssignableNotCopyConstructible*>>,
                 std::iter_value_t<InputAndOutputIterator>>);
static_assert(
    !std::indirectly_copyable_storable<
        cpp20_input_iterator<CopyAssignableNotCopyConstructible*>,
        InputAndOutputIterator>);

// !forward_iterator<I>
// !(input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>)
// indirectly_copyable_storable<I, O>
static_assert(
    HasUniqueCopyIter<
        cpp20_input_iterator<int*>,
        sentinel_wrapper<cpp20_input_iterator<int*>>,
        cpp20_output_iterator<int*>>);
// because:
static_assert(!std::forward_iterator<cpp20_input_iterator<int*>>);
static_assert(!std::input_iterator<cpp20_output_iterator<int*>>);
static_assert(std::indirectly_copyable_storable<cpp20_input_iterator<int*>, cpp20_output_iterator<int*>>);

// !forward_iterator<I>
// !(input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>)
// !indirectly_copyable_storable<I, O>
static_assert(
    !HasUniqueCopyIter<
        cpp20_input_iterator<MoveOnly*>,
        sentinel_wrapper<cpp20_input_iterator<MoveOnly*>>,
        cpp20_output_iterator<AssignableFromMoveOnly*>>);
// because:
static_assert(!std::forward_iterator<cpp20_input_iterator<MoveOnly*>>);
static_assert(!std::input_iterator<cpp20_output_iterator<MoveOnly*>>);
static_assert(
    !std::
        indirectly_copyable_storable<cpp20_input_iterator<MoveOnly*>, cpp20_output_iterator<AssignableFromMoveOnly*>>);

template < class Range, class OutIter = int*, class Comp = std::ranges::equal_to, class Proj = std::identity>
concept HasUniqueCopyRange =
    requires(Range&& range, OutIter&& out, Comp&& comp, Proj&& proj) {
      std::ranges::unique_copy(
          std::forward<Range>(range), std::forward<OutIter>(out), std::forward<Comp>(comp), std::forward<Proj>(proj));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasUniqueCopyRange<R<int*>, int*>);

// !input_range<R>
static_assert(!HasUniqueCopyRange<R<InputIteratorNotDerivedFrom>>);

// !weakly_incrementable<O>
static_assert(!HasUniqueCopyIter<R<int*>, WeaklyIncrementableNotMovable>);

// !indirect_equivalence_relation<Comp, projected<I, Proj>>
static_assert(!HasUniqueCopyIter<R<int*>, int*, ComparatorNotCopyable<int>>);

// !indirectly_copyable<I, O>
static_assert(!HasUniqueCopyIter<R<const int*>, const int*>);

// !forward_iterator<iterator_t<R>>
// !(input_iterator<O> && same_as<range_value_t<R>, iter_value_t<O>>)
// !indirectly_copyable_storable<iterator_t<R>, O>
static_assert(!HasUniqueCopyIter< R<cpp20_input_iterator<MoveOnly*>>, cpp20_output_iterator<AssignableFromMoveOnly*>>);

template <class InIter, class OutIter, template <class> class SentWrapper, std::size_t N1, std::size_t N2>
constexpr void testUniqueCopyImpl(std::array<int, N1> in, std::array<int, N2> expected) {
  using Sent = SentWrapper<InIter>;

  // iterator overload
  {
    std::array<int, N2> out;
    std::same_as<std::ranges::unique_copy_result<InIter, OutIter>> decltype(auto) result =
        std::ranges::unique_copy(InIter{in.data()}, Sent{InIter{in.data() + in.size()}}, OutIter{out.begin()});
    assert(std::ranges::equal(out, expected));
    assert(base(result.in) == in.data() + in.size());
    assert(base(result.out) == out.data() + out.size());
  }

  // range overload
  {
    std::array<int, N2> out;
    std::ranges::subrange r{InIter{in.data()}, Sent{InIter{in.data() + in.size()}}};
    std::same_as<std::ranges::unique_copy_result<InIter, OutIter>> decltype(auto) result =
        std::ranges::unique_copy(r, OutIter{out.begin()});
    assert(std::ranges::equal(out, expected));
    assert(base(result.in) == in.data() + in.size());
    assert(base(result.out) == out.data() + out.size());
  }
}

template <class InIter, class OutIter, template <class> class SentWrapper>
constexpr void testImpl() {
  // no consecutive elements
  {
    std::array in{1, 2, 3, 2, 1};
    std::array expected{1, 2, 3, 2, 1};
    testUniqueCopyImpl<InIter, OutIter, SentWrapper>(in, expected);
  }

  // one group of consecutive elements
  {
    std::array in{2, 3, 3, 3, 4, 3};
    std::array expected{2, 3, 4, 3};
    testUniqueCopyImpl<InIter, OutIter, SentWrapper>(in, expected);
  }

  // multiple groups of consecutive elements
  {
    std::array in{2, 3, 3, 3, 4, 3, 3, 5, 5, 5};
    std::array expected{2, 3, 4, 3, 5};
    testUniqueCopyImpl<InIter, OutIter, SentWrapper>(in, expected);
  }

  // all the same
  {
    std::array in{1, 1, 1, 1, 1, 1};
    std::array expected{1};
    testUniqueCopyImpl<InIter, OutIter, SentWrapper>(in, expected);
  }

  // empty range
  {
    std::array<int, 0> in{};
    std::array<int, 0> expected{};
    testUniqueCopyImpl<InIter, OutIter, SentWrapper>(in, expected);
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
constexpr void withAllPermutationsOfInIterAndOutIter() {
  withAllPermutationsOfInIter<cpp20_output_iterator<int*>, SentWrapper>();
  withAllPermutationsOfInIter<forward_iterator<int*>, SentWrapper>();
  withAllPermutationsOfInIter<bidirectional_iterator<int*>, SentWrapper>();
  withAllPermutationsOfInIter<random_access_iterator<int*>, SentWrapper>();
  withAllPermutationsOfInIter<contiguous_iterator<int*>, SentWrapper>();
  withAllPermutationsOfInIter<int*, SentWrapper>();
}

constexpr bool test() {
  withAllPermutationsOfInIterAndOutIter<std::type_identity_t>();
  withAllPermutationsOfInIterAndOutIter<sentinel_wrapper>();

  // Test the overload that re-reads from the input iterator
  // forward_iterator<I>
  // !(input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>)
  // !indirectly_copyable_storable<I, O>
  {
    MoveOnly in[5] = {1, 3, 3, 3, 1};
    // iterator overload
    {
      AssignableFromMoveOnly out[3] = {};
      auto result                   = std::ranges::unique_copy(in, in + 5, out);
      assert(std::ranges::equal(out, std::array{1, 3, 1}, {}, &AssignableFromMoveOnly::data));
      assert(result.in == in + 5);
      assert(result.out == out + 3);
    }
    // range overload
    {
      AssignableFromMoveOnly out[3] = {};
      auto result                   = std::ranges::unique_copy(std::ranges::subrange{in, in + 5}, out);
      assert(std::ranges::equal(out, std::array{1, 3, 1}, {}, &AssignableFromMoveOnly::data));
      assert(result.in == in + 5);
      assert(result.out == out + 3);
    }
  }

  // Test the overload that re-reads from the output iterator
  // !forward_iterator<I>
  // (input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>)
  // !indirectly_copyable_storable<I, O>
  {
    using InIter                             = cpp20_input_iterator<CopyAssignableNotCopyConstructible*>;
    using Sent                               = sentinel_wrapper<InIter>;
    CopyAssignableNotCopyConstructible in[6] = {1, 1, 2, 2, 3, 3};
    // iterator overload
    {
      CopyAssignableNotCopyConstructible out[3];
      auto result = std::ranges::unique_copy(InIter{in}, Sent{InIter{in + 6}}, InputAndOutputIterator{out});
      assert(std::ranges::equal(out, std::array{1, 2, 3}, {}, &CopyAssignableNotCopyConstructible::data));
      assert(base(result.in) == in + 6);
      assert(base(result.out) == out + 3);
    }
    // range overload
    {
      CopyAssignableNotCopyConstructible out[3];
      auto r      = std::ranges::subrange(InIter{in}, Sent{InIter{in + 6}});
      auto result = std::ranges::unique_copy(r, InputAndOutputIterator{out});
      assert(std::ranges::equal(out, std::array{1, 2, 3}, {}, &CopyAssignableNotCopyConstructible::data));
      assert(base(result.in) == in + 6);
      assert(base(result.out) == out + 3);
    }
  }

  // Test the overload that reads from the temporary copy of the value
  // !forward_iterator<I>
  // !(input_iterator<O> && same_as<iter_value_t<I>, iter_value_t<O>>)
  // indirectly_copyable_storable<I, O>
  {
    using InIter = cpp20_input_iterator<int*>;
    using Sent   = sentinel_wrapper<InIter>;
    int in[4]    = {1, 1, 1, 2};
    // iterator overload
    {
      int out[2];
      auto result = std::ranges::unique_copy(InIter{in}, Sent{InIter{in + 4}}, cpp20_output_iterator<int*>{out});
      assert(std::ranges::equal(out, std::array{1, 2}));
      assert(base(result.in) == in + 4);
      assert(base(result.out) == out + 2);
    }
    // range overload
    {
      int out[2];
      auto r      = std::ranges::subrange(InIter{in}, Sent{InIter{in + 4}});
      auto result = std::ranges::unique_copy(r, cpp20_output_iterator<int*>{out});
      assert(std::ranges::equal(out, std::array{1, 2}));
      assert(base(result.in) == in + 4);
      assert(base(result.out) == out + 2);
    }
  }

  struct Data {
    int data;
  };

  // Test custom comparator
  {
    std::array in{Data{4}, Data{8}, Data{8}, Data{8}};
    std::array expected{Data{4}, Data{8}};
    const auto comp = [](const Data& x, const Data& y) { return x.data == y.data; };

    // iterator overload
    {
      std::array<Data, 2> out;
      auto result = std::ranges::unique_copy(in.begin(), in.end(), out.begin(), comp);
      assert(std::ranges::equal(out, expected, comp));
      assert(base(result.in) == in.begin() + 4);
      assert(base(result.out) == out.begin() + 2);
    }

    // range overload
    {
      std::array<Data, 2> out;
      auto result = std::ranges::unique_copy(in, out.begin(), comp);
      assert(std::ranges::equal(out, expected, comp));
      assert(base(result.in) == in.begin() + 4);
      assert(base(result.out) == out.begin() + 2);
    }
  }

  // Test custom projection
  {
    std::array in{Data{4}, Data{8}, Data{8}, Data{8}};
    std::array expected{Data{4}, Data{8}};

    const auto proj = &Data::data;

    // iterator overload
    {
      std::array<Data, 2> out;
      auto result = std::ranges::unique_copy(in.begin(), in.end(), out.begin(), {}, proj);
      assert(std::ranges::equal(out, expected, {}, proj, proj));
      assert(base(result.in) == in.begin() + 4);
      assert(base(result.out) == out.begin() + 2);
    }

    // range overload
    {
      std::array<Data, 2> out;
      auto result = std::ranges::unique_copy(in, out.begin(), {}, proj);
      assert(std::ranges::equal(out, expected, {}, proj, proj));
      assert(base(result.in) == in.begin() + 4);
      assert(base(result.out) == out.begin() + 2);
    }
  }

  // Exactly last - first - 1 applications of the corresponding predicate and no
  // more than twice as many applications of any projection.
  {
    std::array in{1, 2, 3, 3, 3, 4, 3, 3, 5, 5, 6, 6, 1};
    std::array expected{1, 2, 3, 4, 3, 5, 6, 1};
    // iterator overload
    {
      std::array<int, 8> out;
      int numberOfComp = 0;
      int numberOfProj = 0;
      auto result      = std::ranges::unique_copy(
          in.begin(),
          in.end(),
          out.begin(),
          counting_predicate{std::ranges::equal_to{}, numberOfComp},
          counting_projection{numberOfProj});
      assert(std::ranges::equal(out, expected));
      assert(base(result.in) == in.end());
      assert(base(result.out) == out.end());
      assert(numberOfComp == in.size() - 1);
      assert(numberOfProj <= static_cast<int>(2 * (in.size() - 1)));
    }
    // range overload
    {
      std::array<int, 8> out;
      int numberOfComp = 0;
      int numberOfProj = 0;
      auto result      = std::ranges::unique_copy(
          in,
          out.begin(),
          counting_predicate{std::ranges::equal_to{}, numberOfComp},
          counting_projection{numberOfProj});
      assert(std::ranges::equal(out, expected));
      assert(base(result.in) == in.end());
      assert(base(result.out) == out.end());
      assert(numberOfComp == in.size() - 1);
      assert(numberOfProj <= static_cast<int>(2 * (in.size() - 1)));
    }
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
