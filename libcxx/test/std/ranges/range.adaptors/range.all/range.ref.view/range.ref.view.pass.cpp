//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<range R>
//  requires is_object_v<R>
// class ref_view;

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

int globalBuff[8];

template<class T>
concept ValidRefView = requires { typename std::ranges::ref_view<T>; };

struct Range {
  int start = 0;
  friend constexpr int* begin(Range const& range) { return globalBuff + range.start; }
  friend constexpr int* end(Range const&) { return globalBuff + 8; }
  friend constexpr int* begin(Range& range) { return globalBuff + range.start; }
  friend constexpr int* end(Range&) { return globalBuff + 8; }
};

struct BeginOnly {
  friend int* begin(BeginOnly const&);
  friend int* begin(BeginOnly &);
};

static_assert( ValidRefView<Range>);
static_assert(!ValidRefView<BeginOnly>);
static_assert(!ValidRefView<int (&)[4]>);
static_assert( ValidRefView<int[4]>);

static_assert(std::derived_from<std::ranges::ref_view<Range>, std::ranges::view_interface<std::ranges::ref_view<Range>>>);

struct RangeConvertible {
  operator Range& ();
};

struct RValueRangeConvertible {
  operator Range&& ();
};

static_assert( std::is_constructible_v<std::ranges::ref_view<Range>, Range&>);
static_assert( std::is_constructible_v<std::ranges::ref_view<Range>, RangeConvertible>);
static_assert(!std::is_constructible_v<std::ranges::ref_view<Range>, RValueRangeConvertible>);

struct ConstConvertibleToLValueAndRValue {
  operator Range& () const;
  operator Range&& () const;
};
static_assert( std::is_convertible_v<RangeConvertible, std::ranges::ref_view<Range>>);
static_assert(!std::is_convertible_v<RValueRangeConvertible, std::ranges::ref_view<Range>>);
static_assert(!std::is_convertible_v<ConstConvertibleToLValueAndRValue, std::ranges::ref_view<Range>>);

struct ForwardRange {
  constexpr forward_iterator<int*> begin() const { return forward_iterator<int*>(globalBuff); }
  constexpr forward_iterator<int*> end() const { return forward_iterator<int*>(globalBuff + 8); }
};

struct Cpp17InputRange {
  struct sentinel {
    friend constexpr bool operator==(sentinel, cpp17_input_iterator<int*> iter) { return base(iter) == globalBuff + 8; }
    friend constexpr std::ptrdiff_t operator-(sentinel, cpp17_input_iterator<int*>) { return -8; }
    friend constexpr std::ptrdiff_t operator-(cpp17_input_iterator<int*>, sentinel) { return 8; }
  };

  constexpr cpp17_input_iterator<int*> begin() const {
    return cpp17_input_iterator<int*>(globalBuff);
  }
  constexpr sentinel end() const { return {}; }
};

struct Cpp20InputRange {
  struct sentinel {
    friend constexpr bool operator==(sentinel, const cpp20_input_iterator<int*> &iter) { return base(iter) == globalBuff + 8; }
    friend constexpr std::ptrdiff_t operator-(sentinel, const cpp20_input_iterator<int*>&) { return -8; }
  };

  constexpr cpp20_input_iterator<int*> begin() const {
    return cpp20_input_iterator<int*>(globalBuff);
  }
  constexpr sentinel end() const { return {}; }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<Cpp20InputRange> = true;

template<class R>
concept EmptyIsInvocable = requires (std::ranges::ref_view<R> view) { view.empty(); };

template<class R>
concept SizeIsInvocable = requires (std::ranges::ref_view<R> view) { view.size(); };

template<class R>
concept DataIsInvocable = requires (std::ranges::ref_view<R> view) { view.data(); };

// Testing ctad.
static_assert(std::same_as<decltype(std::ranges::ref_view(std::declval<Range&>())),
              std::ranges::ref_view<Range>>);

constexpr bool test() {
  {
    // ref_view::base
    Range range;
    std::ranges::ref_view<Range> view{range};
    assert(view.begin() == globalBuff);
    view.base() = Range{2};
    assert(view.begin() == globalBuff + 2);
  }

  {
    // ref_view::begin
    Range range1;
    std::ranges::ref_view<Range> view1 = range1;
    assert(view1.begin() == globalBuff);

    ForwardRange range2;
    std::ranges::ref_view<ForwardRange> view2 = range2;
    assert(base(view2.begin()) == globalBuff);

    Cpp17InputRange range3;
    std::ranges::ref_view<Cpp17InputRange> view3 = range3;
    assert(base(view3.begin()) == globalBuff);

    Cpp20InputRange range4;
    std::ranges::ref_view<Cpp20InputRange> view4 = range4;
    assert(base(view4.begin()) == globalBuff);
  }

  {
    // ref_view::end
    Range range1;
    std::ranges::ref_view<Range> view1 = range1;
    assert(view1.end() == globalBuff + 8);

    ForwardRange range2;
    std::ranges::ref_view<ForwardRange> view2 = range2;
    assert(base(view2.end()) == globalBuff + 8);

    Cpp17InputRange range3;
    std::ranges::ref_view<Cpp17InputRange> view3 = range3;
    assert(view3.end() == cpp17_input_iterator(globalBuff + 8));

    Cpp20InputRange range4;
    std::ranges::ref_view<Cpp20InputRange> view4 = range4;
    assert(view4.end() == cpp20_input_iterator(globalBuff + 8));
  }

  {
    // ref_view::empty
    Range range{8};
    std::ranges::ref_view<Range> view1 = range;
    assert(view1.empty());

    ForwardRange range2;
    std::ranges::ref_view<ForwardRange> view2 = range2;
    assert(!view2.empty());

    static_assert(!EmptyIsInvocable<Cpp17InputRange>);
    static_assert(!EmptyIsInvocable<Cpp20InputRange>);
  }

  {
    // ref_view::size
    Range range1{8};
    std::ranges::ref_view<Range> view1 = range1;
    assert(view1.size() == 0);

    Range range2{2};
    std::ranges::ref_view<Range> view2 = range2;
    assert(view2.size() == 6);

    static_assert(!SizeIsInvocable<ForwardRange>);
  }

  {
    // ref_view::data
    Range range1;
    std::ranges::ref_view<Range> view1 = range1;
    assert(view1.data() == globalBuff);

    Range range2{2};
    std::ranges::ref_view<Range> view2 = range2;
    assert(view2.data() == globalBuff + 2);

    static_assert(!DataIsInvocable<ForwardRange>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
