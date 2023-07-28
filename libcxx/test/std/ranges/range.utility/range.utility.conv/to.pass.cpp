//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class C, input_range R, class... Args> requires (!view<C>)
//   constexpr C to(R&& r, Args&&... args);     // Since C++23

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>
#include "container.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

template <class Container, class Range, class... Args>
concept HasTo = requires (Range&& range, Args ...args) {
  std::ranges::to<Container>(std::forward<Range>(range), std::forward<Args>(args)...);
};

struct InputRange {
  int x = 0;
  constexpr cpp20_input_iterator<int*> begin() {
    return cpp20_input_iterator<int*>(&x);
  }
  constexpr sentinel_wrapper<cpp20_input_iterator<int*>> end() {
    return sentinel_wrapper<cpp20_input_iterator<int*>>(begin());
  }
};
static_assert(std::ranges::input_range<InputRange>);

struct common_cpp20_input_iterator {
  using value_type = int;
  using difference_type = long long;
  using iterator_concept = std::input_iterator_tag;
  // Deliberately not defining `iterator_category` to make sure this class satisfies the `input_iterator` concept but
  // would fail `derived_from<iterator_category, input_iterator_tag>`.

  int x = 0;

  // Copyable so that it can be used as a sentinel against itself.
  constexpr decltype(auto) operator*() const { return x; }
  constexpr common_cpp20_input_iterator& operator++() { return *this; }
  constexpr void operator++(int) {}
  constexpr friend bool operator==(common_cpp20_input_iterator, common_cpp20_input_iterator) { return true; }
};
static_assert(std::input_iterator<common_cpp20_input_iterator>);
static_assert(std::sentinel_for<common_cpp20_input_iterator, common_cpp20_input_iterator>);
template <class T>
concept HasIteratorCategory = requires {
  typename std::iterator_traits<T>::iterator_category;
};
static_assert(!HasIteratorCategory<common_cpp20_input_iterator>);

struct CommonInputRange {
  int x = 0;
  constexpr common_cpp20_input_iterator begin() { return {}; }
  constexpr common_cpp20_input_iterator end() { return begin(); }
};
static_assert(std::ranges::input_range<CommonInputRange>);
static_assert(std::ranges::common_range<CommonInputRange>);

struct CommonRange {
  int x = 0;
  constexpr forward_iterator<int*> begin() {
    return forward_iterator<int*>(&x);
  }
  constexpr forward_iterator<int*> end() {
    return begin();
  }
};
static_assert(std::ranges::input_range<CommonRange>);
static_assert(std::ranges::common_range<CommonRange>);

struct NonCommonRange {
  int x = 0;
  constexpr forward_iterator<int*> begin() {
    return forward_iterator<int*>(&x);
  }
  constexpr sentinel_wrapper<forward_iterator<int*>> end() {
    return sentinel_wrapper<forward_iterator<int*>>(begin());
  }
};
static_assert(std::ranges::input_range<NonCommonRange>);
static_assert(!std::ranges::common_range<NonCommonRange>);
static_assert(std::derived_from<
    typename std::iterator_traits<std::ranges::iterator_t<NonCommonRange>>::iterator_category,
    std::input_iterator_tag>);

using ContainerT = int;
static_assert(!std::ranges::view<ContainerT>);
static_assert(HasTo<ContainerT, InputRange>);
static_assert(!HasTo<test_view<forward_iterator>, InputRange>);

// Note: it's not possible to check the `input_range` constraint because if it's not satisfied, the pipe adaptor
// overload hijacks the call (it takes unconstrained variadic arguments).

// Check the exact constraints for each one of the cases inside `ranges::to`.

struct Empty {};

struct Fallback {
  using value_type = int;

  CtrChoice ctr_choice = CtrChoice::Invalid;
  int x = 0;

  constexpr Fallback() : ctr_choice(CtrChoice::DefaultCtrAndInsert) {}
  constexpr Fallback(Empty) : ctr_choice(CtrChoice::DefaultCtrAndInsert) {}

  constexpr void push_back(value_type) {}
  constexpr value_type* begin() { return &x; }
  constexpr value_type* end() { return &x; }
  std::size_t size() const { return 0; }
};

struct CtrDirectOrFallback : Fallback {
  using Fallback::Fallback;
  constexpr CtrDirectOrFallback(InputRange&&, int = 0) { ctr_choice = CtrChoice::DirectCtr; }
};

struct CtrFromRangeTOrFallback : Fallback {
  using Fallback::Fallback;
  constexpr CtrFromRangeTOrFallback(std::from_range_t, InputRange&&, int = 0) { ctr_choice = CtrChoice::FromRangeT; }
};

struct CtrBeginEndPairOrFallback : Fallback {
  using Fallback::Fallback;
  template <class Iter>
  constexpr CtrBeginEndPairOrFallback(Iter, Iter, int = 0) { ctr_choice = CtrChoice::BeginEndPair; }
};

template <bool HasSize>
struct MaybeSizedRange {
  int x = 0;
  constexpr forward_iterator<int*> begin() { return forward_iterator<int*>(&x); }
  constexpr forward_iterator<int*> end() { return begin(); }

  constexpr std::size_t size() const
  requires HasSize {
    return 0;
  }
};
static_assert(std::ranges::sized_range<MaybeSizedRange<true>>);
static_assert(!std::ranges::sized_range<MaybeSizedRange<false>>);

template <bool HasCapacity = true, bool CapacityReturnsSizeT = true,
          bool HasMaxSize = true, bool MaxSizeReturnsSizeT = true>
struct Reservable : Fallback {
  bool reserve_called = false;

  using Fallback::Fallback;

  constexpr std::size_t capacity() const
  requires (HasCapacity && CapacityReturnsSizeT) {
    return 0;
  }
  constexpr int capacity() const
  requires (HasCapacity && !CapacityReturnsSizeT) {
    return 0;
  }

  constexpr std::size_t max_size() const
  requires (HasMaxSize && MaxSizeReturnsSizeT) {
    return 0;
  }
  constexpr int max_size() const
  requires (HasMaxSize && !MaxSizeReturnsSizeT) {
    return 0;
  }

  constexpr void reserve(std::size_t) {
    reserve_called = true;
  }
};
LIBCPP_STATIC_ASSERT(std::ranges::__reservable_container<Reservable<>>);

constexpr void test_constraints() {
  { // Case 1 -- construct directly from the range.
    { // (range)
      auto result = std::ranges::to<CtrDirectOrFallback>(InputRange());
      assert(result.ctr_choice == CtrChoice::DirectCtr);
    }

    { // (range, arg)
      auto result = std::ranges::to<CtrDirectOrFallback>(InputRange(), 1);
      assert(result.ctr_choice == CtrChoice::DirectCtr);
    }

    { // (range, convertible-to-arg)
      auto result = std::ranges::to<CtrDirectOrFallback>(InputRange(), 1.0);
      assert(result.ctr_choice == CtrChoice::DirectCtr);
    }

    { // (range, BAD_arg)
      auto result = std::ranges::to<CtrDirectOrFallback>(InputRange(), Empty());
      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
    }
  }

  { // Case 2 -- construct using the `from_range_t` tagged constructor.
    { // (range)
      auto result = std::ranges::to<CtrFromRangeTOrFallback>(InputRange());
      assert(result.ctr_choice == CtrChoice::FromRangeT);
    }

    { // (range, arg)
      auto result = std::ranges::to<CtrFromRangeTOrFallback>(InputRange(), 1);
      assert(result.ctr_choice == CtrChoice::FromRangeT);
    }

    { // (range, convertible-to-arg)
      auto result = std::ranges::to<CtrFromRangeTOrFallback>(InputRange(), 1.0);
      assert(result.ctr_choice == CtrChoice::FromRangeT);
    }

    { // (range, BAD_arg)
      auto result = std::ranges::to<CtrFromRangeTOrFallback>(InputRange(), Empty());
      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
    }
  }

  { // Case 3 -- construct from a begin-end iterator pair.
    { // (range)
      auto result = std::ranges::to<CtrBeginEndPairOrFallback>(CommonRange());
      assert(result.ctr_choice == CtrChoice::BeginEndPair);
    }

    { // (range, arg)
      auto result = std::ranges::to<CtrBeginEndPairOrFallback>(CommonRange(), 1);
      assert(result.ctr_choice == CtrChoice::BeginEndPair);
    }

    { // (range, convertible-to-arg)
      auto result = std::ranges::to<CtrBeginEndPairOrFallback>(CommonRange(), 1.0);
      assert(result.ctr_choice == CtrChoice::BeginEndPair);
    }

    { // (BAD_range) -- not a common range.
      auto result = std::ranges::to<CtrBeginEndPairOrFallback>(NonCommonRange());
      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
    }

    { // (BAD_range) -- iterator type not derived from `input_iterator_tag`.
      auto result = std::ranges::to<CtrBeginEndPairOrFallback>(CommonInputRange());
      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
    }

    { // (range, BAD_arg)
      auto result = std::ranges::to<CtrBeginEndPairOrFallback>(CommonRange(), Empty());
      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
    }
  }

  { // Case 4 -- default-construct (or construct from the extra arguments) and insert, reserving the size if possible.
    // Note: it's not possible to check the constraints on the default constructor using this approach because there is
    // nothing to fall back to -- the call will result in a hard error.
    // However, it's possible to check the constraints on reserving the capacity.

    { // All constraints satisfied.
      using C = Reservable<>;
      auto result = std::ranges::to<C>(MaybeSizedRange<true>());
      assert(result.reserve_called);
    }

    { // !sized_range
      using C = Reservable<>;
      auto result = std::ranges::to<C>(MaybeSizedRange<false>());
      assert(!result.reserve_called);
    }

    { // Missing `capacity`.
      using C = Reservable</*HasCapacity=*/false>;
      auto result = std::ranges::to<C>(MaybeSizedRange<true>());
      assert(!result.reserve_called);
    }

    { // `capacity` doesn't return `size_type`.
      using C = Reservable</*HasCapacity=*/true, /*CapacityReturnsSizeT=*/false>;
      auto result = std::ranges::to<C>(MaybeSizedRange<true>());
      assert(!result.reserve_called);
    }

    { // Missing `max_size`.
      using C = Reservable</*HasCapacity=*/true, /*CapacityReturnsSizeT=*/true, /*HasMaxSize=*/false>;
      auto result = std::ranges::to<C>(MaybeSizedRange<true>());
      assert(!result.reserve_called);
    }

    { // `max_size` doesn't return `size_type`.
      using C = Reservable<
        /*HasCapacity=*/true, /*CapacityReturnsSizeT=*/true, /*HasMaxSize=*/true, /*MaxSizeReturnsSizeT=*/false>;
      auto result = std::ranges::to<C>(MaybeSizedRange<true>());
      assert(!result.reserve_called);
    }
  }
}

constexpr void test_ctr_choice_order() {
  std::array in = {1, 2, 3, 4, 5};
  int arg1 = 42;
  char arg2 = 'a';

  { // Case 1 -- construct directly from the given range.
    {
      using C = Container<int, CtrChoice::DirectCtr>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);

      assert(result.ctr_choice == CtrChoice::DirectCtr);
      assert(std::ranges::equal(result, in));
      assert((in | std::ranges::to<C>()) == result);
      auto closure = std::ranges::to<C>();
      assert((in | closure) == result);
    }

    { // Extra arguments.
      using C = Container<int, CtrChoice::DirectCtr>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in, arg1, arg2);

      assert(result.ctr_choice == CtrChoice::DirectCtr);
      assert(std::ranges::equal(result, in));
      assert(result.extra_arg1 == arg1);
      assert(result.extra_arg2 == arg2);
      assert((in | std::ranges::to<C>(arg1, arg2)) == result);
      auto closure = std::ranges::to<C>(arg1, arg2);
      assert((in | closure) == result);
    }
  }

  { // Case 2 -- construct using the `from_range_t` tag.
    {
      using C = Container<int, CtrChoice::FromRangeT>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);

      assert(result.ctr_choice == CtrChoice::FromRangeT);
      assert(std::ranges::equal(result, in));
      assert((in | std::ranges::to<C>()) == result);
      auto closure = std::ranges::to<C>();
      assert((in | closure) == result);
    }

    { // Extra arguments.
      using C = Container<int, CtrChoice::FromRangeT>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in, arg1, arg2);

      assert(result.ctr_choice == CtrChoice::FromRangeT);
      assert(std::ranges::equal(result, in));
      assert(result.extra_arg1 == arg1);
      assert(result.extra_arg2 == arg2);
      assert((in | std::ranges::to<C>(arg1, arg2)) == result);
      auto closure = std::ranges::to<C>(arg1, arg2);
      assert((in | closure) == result);
    }
  }

  { // Case 3 -- construct from a begin-end pair.
    {
      using C = Container<int, CtrChoice::BeginEndPair>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);

      assert(result.ctr_choice == CtrChoice::BeginEndPair);
      assert(std::ranges::equal(result, in));
      assert((in | std::ranges::to<C>()) == result);
      auto closure = std::ranges::to<C>();
      assert((in | closure) == result);
    }

    { // Extra arguments.
      using C = Container<int, CtrChoice::BeginEndPair>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in, arg1, arg2);

      assert(result.ctr_choice == CtrChoice::BeginEndPair);
      assert(std::ranges::equal(result, in));
      assert(result.extra_arg1 == arg1);
      assert(result.extra_arg2 == arg2);
      assert((in | std::ranges::to<C>(arg1, arg2)) == result);
      auto closure = std::ranges::to<C>(arg1, arg2);
      assert((in | closure) == result);
    }
  }

  { // Case 4 -- default-construct then insert elements.
    {
      using C = Container<int, CtrChoice::DefaultCtrAndInsert, InserterChoice::Insert, /*CanReserve=*/false>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);

      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
      assert(result.inserter_choice == InserterChoice::Insert);
      assert(std::ranges::equal(result, in));
      assert(!result.called_reserve);
      assert((in | std::ranges::to<C>()) == result);
      auto closure = std::ranges::to<C>();
      assert((in | closure) == result);
    }

    {
      using C = Container<int, CtrChoice::DefaultCtrAndInsert, InserterChoice::Insert, /*CanReserve=*/true>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);

      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
      assert(result.inserter_choice == InserterChoice::Insert);
      assert(std::ranges::equal(result, in));
      assert(result.called_reserve);
      assert((in | std::ranges::to<C>()) == result);
      auto closure = std::ranges::to<C>();
      assert((in | closure) == result);
    }

    {
      using C = Container<int, CtrChoice::DefaultCtrAndInsert, InserterChoice::PushBack, /*CanReserve=*/false>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);

      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
      assert(result.inserter_choice == InserterChoice::PushBack);
      assert(std::ranges::equal(result, in));
      assert(!result.called_reserve);
      assert((in | std::ranges::to<C>()) == result);
      auto closure = std::ranges::to<C>();
      assert((in | closure) == result);
    }

    {
      using C = Container<int, CtrChoice::DefaultCtrAndInsert, InserterChoice::PushBack, /*CanReserve=*/true>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);

      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
      assert(result.inserter_choice == InserterChoice::PushBack);
      assert(std::ranges::equal(result, in));
      assert(result.called_reserve);
      assert((in | std::ranges::to<C>()) == result);
      auto closure = std::ranges::to<C>();
      assert((in | closure) == result);
    }

    { // Extra arguments.
      using C = Container<int, CtrChoice::DefaultCtrAndInsert, InserterChoice::Insert, /*CanReserve=*/false>;
      std::same_as<C> decltype(auto) result = std::ranges::to<C>(in, arg1, arg2);

      assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);
      assert(result.inserter_choice == InserterChoice::Insert);
      assert(std::ranges::equal(result, in));
      assert(!result.called_reserve);
      assert(result.extra_arg1 == arg1);
      assert(result.extra_arg2 == arg2);
      assert((in | std::ranges::to<C>(arg1, arg2)) == result);
      auto closure = std::ranges::to<C>(arg1, arg2);
      assert((in | closure) == result);
    }
  }
}

template <CtrChoice Rank>
struct NotARange {
  using value_type = int;

  constexpr NotARange(std::ranges::input_range auto&&)
  requires (Rank >= CtrChoice::DirectCtr)
  {}

  constexpr NotARange(std::from_range_t, std::ranges::input_range auto&&)
  requires (Rank >= CtrChoice::FromRangeT)
  {}

  template <class Iter>
  constexpr NotARange(Iter, Iter)
  requires (Rank >= CtrChoice::BeginEndPair)
  {}

  constexpr NotARange()
  requires (Rank >= CtrChoice::DefaultCtrAndInsert)
  = default;

  constexpr void push_back(int) {}
};

static_assert(!std::ranges::range<NotARange<CtrChoice::DirectCtr>>);

constexpr void test_lwg_3785() {
  // Test LWG 3785 ("`ranges::to` is over-constrained on the destination type being a range") -- make sure it's possible
  // to convert the given input range to a non-range type.
  std::array in = {1, 2, 3, 4, 5};

  {
    using C = NotARange<CtrChoice::DirectCtr>;
    [[maybe_unused]] std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);
  }

  {
    using C = NotARange<CtrChoice::FromRangeT>;
    [[maybe_unused]] std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);
  }

  {
    using C = NotARange<CtrChoice::BeginEndPair>;
    [[maybe_unused]] std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);
  }

  {
    using C = NotARange<CtrChoice::DefaultCtrAndInsert>;
    [[maybe_unused]] std::same_as<C> decltype(auto) result = std::ranges::to<C>(in);
  }
}

constexpr void test_recursive() {
  using C1 = Container<int, CtrChoice::DirectCtr>;
  using C2 = Container<C1, CtrChoice::FromRangeT>;
  using C3 = Container<C2, CtrChoice::BeginEndPair>;
  using C4 = Container<C3, CtrChoice::DefaultCtrAndInsert, InserterChoice::PushBack>;
  using A1 = std::array<int, 4>;
  using A2 = std::array<A1, 3>;
  using A3 = std::array<A2, 2>;
  using A4 = std::array<A3, 2>;

  A4 in = {};
  { // Fill the nested array with incremental values.
    int x = 0;
    for (auto& a3 : in) {
      for (auto& a2 : a3) {
        for (auto& a1 : a2) {
          for (int& el : a1) {
            el = x++;
          }
        }
      }
    }
  }

  std::same_as<C4> decltype(auto) result = std::ranges::to<C4>(in);

  assert(result.ctr_choice == CtrChoice::DefaultCtrAndInsert);

  int expected_value = 0;
  for (auto& c3 : result) {
    assert(c3.ctr_choice == CtrChoice::BeginEndPair);

    for (auto& c2 : c3) {
      assert(c2.ctr_choice == CtrChoice::FromRangeT);

      for (auto& c1 : c2) {
        assert(c1.ctr_choice == CtrChoice::DirectCtr);

        for (int el : c1) {
          assert(el == expected_value);
          ++expected_value;
        }
      }
    }
  }

  assert((in | std::ranges::to<C4>()) == result);
}

constexpr bool test() {
  test_constraints();
  test_ctr_choice_order();
  test_lwg_3785();
  test_recursive();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
