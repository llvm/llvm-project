//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr __iterator(__iterator<!_Const> __i)
//    requires _Const && convertible_to<ranges::iterator_t<_View>, iterator_t<_Base>> &&
//                 convertible_to<sentinel_t<_View>, sentinel_t<_Base>>

#include <iterator>
#include <ranges>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

struct NotSimpleViewIterEnd;
template <bool, bool>
struct NotSimpleViewConstIterEnd;
template <bool, bool>
struct NotSimpleViewConstIterBegin;

struct NotSimpleViewIterBegin : InputIterBase<NotSimpleViewIterBegin> {
  template <bool CopyConvertible, bool MoveConvertible>
  friend constexpr bool
  operator==(const NotSimpleViewIterBegin&, const NotSimpleViewConstIterEnd<CopyConvertible, MoveConvertible>&) {
    return true;
  }
  template <bool CopyConvertible, bool MoveConvertible>
  friend constexpr bool
  operator==(const NotSimpleViewIterBegin&, const NotSimpleViewConstIterBegin<CopyConvertible, MoveConvertible>&) {
    return true;
  }
};

template <bool CopyConvertible, bool MoveConvertible>
struct NotSimpleViewConstIterBegin : InputIterBase<NotSimpleViewConstIterBegin<CopyConvertible, MoveConvertible>> {
  constexpr NotSimpleViewConstIterBegin()                                              = default;
  constexpr NotSimpleViewConstIterBegin(NotSimpleViewConstIterBegin&&)                 = default;
  constexpr NotSimpleViewConstIterBegin& operator=(const NotSimpleViewConstIterBegin&) = default;
  constexpr NotSimpleViewConstIterBegin& operator=(NotSimpleViewConstIterBegin&&)      = default;

  constexpr NotSimpleViewConstIterBegin(const NotSimpleViewConstIterBegin&) {}
  constexpr NotSimpleViewConstIterBegin(const NotSimpleViewIterBegin&)
    requires CopyConvertible
  {}
  constexpr NotSimpleViewConstIterBegin(NotSimpleViewIterBegin&&)
    requires MoveConvertible
  {}

  friend constexpr bool
  operator==(const NotSimpleViewConstIterBegin<CopyConvertible, MoveConvertible>&, const NotSimpleViewIterEnd&) {
    return true;
  }
  friend constexpr bool
  operator==(const NotSimpleViewConstIterBegin<CopyConvertible, MoveConvertible>&, const NotSimpleViewIterBegin&) {
    return true;
  }
};

struct NotSimpleViewIterEnd : InputIterBase<NotSimpleViewIterEnd> {
  template <bool CopyConvertible, bool MoveConvertible>
  friend constexpr bool
  operator==(const NotSimpleViewIterEnd&, const NotSimpleViewConstIterBegin<CopyConvertible, MoveConvertible>&) {
    return true;
  }
  template <bool CopyConvertible, bool MoveConvertible>
  friend constexpr bool
  operator==(const NotSimpleViewIterEnd&, const NotSimpleViewConstIterEnd<CopyConvertible, MoveConvertible>&) {
    return true;
  }

  friend constexpr bool operator==(const NotSimpleViewIterEnd&, const NotSimpleViewIterBegin&) { return true; }
};

template <bool CopyConvertible, bool MoveConvertible>
struct NotSimpleViewConstIterEnd : InputIterBase<NotSimpleViewConstIterEnd<CopyConvertible, MoveConvertible>> {
  constexpr NotSimpleViewConstIterEnd()                                            = default;
  constexpr NotSimpleViewConstIterEnd(NotSimpleViewConstIterEnd&&)                 = default;
  constexpr NotSimpleViewConstIterEnd& operator=(const NotSimpleViewConstIterEnd&) = default;
  constexpr NotSimpleViewConstIterEnd& operator=(NotSimpleViewConstIterEnd&&)      = default;

  constexpr NotSimpleViewConstIterEnd(const NotSimpleViewConstIterEnd&) {}
  constexpr NotSimpleViewConstIterEnd(const NotSimpleViewIterEnd&)
    requires CopyConvertible
  {}
  constexpr NotSimpleViewConstIterEnd(NotSimpleViewIterEnd&&)
    requires MoveConvertible
  {}

  friend constexpr bool
  operator==(const NotSimpleViewConstIterEnd<CopyConvertible, MoveConvertible>&, const NotSimpleViewIterEnd&) {
    return true;
  }
  friend constexpr bool
  operator==(const NotSimpleViewConstIterEnd<CopyConvertible, MoveConvertible>&, const NotSimpleViewIterBegin&) {
    return true;
  }
};

/*
 * Goal: We will need a way to get a stride_view<true>::__iterator and a
 * stride_view<false>::__iterator because those are the two possible types
 * of the stride_view::__iterator constructor. The template value is determined
 * by whether the stride_view::__iterator is derivative of a stride_view over a
 * view that is simple.
 *
 * So, first things first, we need to build a stride_view over a (non-)simple view.
 * There are (at least) two ways that a view can be non-simple:
 * 1. The iterator type for const begin is different than the iterator type for begin
 * 2. The iterator type for const end is different that the iterator type for end
 *
 * So, let's create two different classes where that is the case so that we can test
 * for those conditions individually. We parameterize with a template to decide
 * whether to
 * 1. enable converting constructors between the non-const and the const version.
 * That feature is important for testing the stride_view::__iterator<true> converting
 * constructor from a stride_view::_iterator<false> iterator.
 * 2. enable copyability. That feature is important for testing whether the requirement
 * the that copy constructor for the stride_view::__iterator<false> type actually moves
 * the underlying iterator.
 */
template <bool CopyConvertible = false, bool MoveConvertible = true>
struct NotSimpleViewDifferentBegin : std::ranges::view_base {
  constexpr NotSimpleViewConstIterBegin<CopyConvertible, MoveConvertible> begin() const { return {}; }
  constexpr NotSimpleViewIterBegin begin() { return {}; }

  constexpr NotSimpleViewIterEnd end() const { return {}; }
};

template <bool CopyConvertible = false, bool MoveConvertible = true>
struct NotSimpleViewDifferentEnd : std::ranges::view_base {
  constexpr NotSimpleViewIterBegin begin() const { return {}; }
  constexpr NotSimpleViewConstIterEnd<CopyConvertible, MoveConvertible> end() const { return {}; }
  constexpr NotSimpleViewIterEnd end() { return {}; }
};

constexpr bool non_simple_view_iter_ctor_test() {
  using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentBegin<false>>;
  using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
  using NotSimpleStrideViewIterConst = std::ranges::iterator_t<const NotSimpleStrideView>;
  static_assert(!std::is_same_v<NotSimpleStrideViewIterConst, NotSimpleStrideViewIter>);
  return true;
}

constexpr bool non_const_iterator_copy_ctor() {
  // All tests share the following general configuration.
  //
  // Instantiate a stride view StrideView over a non-simple view (NotSimpleViewBeingStrided) whose
  // 1. std::ranges::iterator_t<StrideView> base's type is NotSimpleViewBeingStridedIterator
  // 2. std::ranges::iterator_t<const StrideView> base's type is NotSimpleViewBeingStridedConstIterator
  // 3. NotSimpleViewBeingStridedIterator is ONLY move-convertible to NotSimpleViewBeingStridedConstIterator
  // 4. std::ranges::sentinel_t are the same whether SV is const or not.
  // 5. the type of StrideView::end is the same whether StrideView is const or not.
  // 6. the type of StrideView::begin is stride_view::iterator<true> when StrideView is const and
  //    stride_view::iterator<false> when StrideView is non const.
  // Visually, it looks like this:
  //
  //  NotSimpleViewBeingStrided(Const)Iterator <-----
  //                ^                               |
  //                |                               |
  //                | begin (const?)                |
  //                |                               |
  //     NotSimpleViewBeingStrided                  |
  //                ^                               |
  //                |                               |
  //                | Strides over                  |
  //                |                               |
  //            StrideView                          |
  //                |                               |
  //                | begin (const?)                |
  //                |                               |
  //                \/                              |
  //       StrideView(Const)Iter                    |
  //                |                               |
  //                | base                          |
  //                |                               |
  //                ---------------------------------

  {
    // Stride over non-simple view over whose iterators are copy convertible -- should look (statically)
    // like it is possible copy construct the stride view's iterator (the move-only requirement comes from
    // a move of the current between the copied-from iterator to the copied-to iterator).
    using NotSimpleViewBeingStrided              = NotSimpleViewDifferentEnd<true, false>;
    using NotSimpleViewBeingStridedIterator      = std::ranges::iterator_t<NotSimpleViewBeingStrided>;
    using NotSimpleViewBeingStridedConstIterator = std::ranges::iterator_t<const NotSimpleViewBeingStrided>;

    using StrideView = std::ranges::stride_view<NotSimpleViewBeingStrided>;

    using StrideViewIter      = std::ranges::iterator_t<StrideView>;
    using StrideViewConstIter = std::ranges::iterator_t<const StrideView>;

    using StrideViewSentinel      = std::ranges::sentinel_t<StrideView>;
    using StrideViewConstSentinel = std::ranges::sentinel_t<const StrideView>;

    static_assert(std::convertible_to<StrideViewIter, StrideViewConstIter>);
    static_assert(std::constructible_from<StrideViewConstIter, StrideViewIter>);
  }

  {
    // Stride over non-simple view over whose iterators are move convertible -- should look (statically)
    // like it is possible copy construct the stride view's iterator (the move-only requirement comes from
    // a move of the current between the copied-from iterator to the copied-to iterator).
    using NotSimpleViewBeingStrided              = NotSimpleViewDifferentEnd<false, true>;
    using NotSimpleViewBeingStridedIterator      = std::ranges::iterator_t<NotSimpleViewBeingStrided>;
    using NotSimpleViewBeingStridedConstIterator = std::ranges::iterator_t<const NotSimpleViewBeingStrided>;

    using StrideView = std::ranges::stride_view<NotSimpleViewBeingStrided>;

    using StrideViewIter      = std::ranges::iterator_t<StrideView>;
    using StrideViewConstIter = std::ranges::iterator_t<const StrideView>;

    using StrideViewSentinel      = std::ranges::sentinel_t<StrideView>;
    using StrideViewConstSentinel = std::ranges::sentinel_t<const StrideView>;

    static_assert(std::convertible_to<StrideViewIter, StrideViewConstIter>);
    static_assert(std::constructible_from<StrideViewConstIter, StrideViewIter>);
  }

  {
    // Stride over non-simple view over whose iterators are not convertible -- should not be able
    // to copy construct the stride view's iterator.
    using NotSimpleViewBeingStrided              = NotSimpleViewDifferentEnd<false, false>;
    using NotSimpleViewBeingStridedIterator      = std::ranges::iterator_t<NotSimpleViewBeingStrided>;
    using NotSimpleViewBeingStridedConstIterator = std::ranges::iterator_t<const NotSimpleViewBeingStrided>;

    using StrideView = std::ranges::stride_view<NotSimpleViewBeingStrided>;

    using StrideViewIter      = std::ranges::iterator_t<StrideView>;
    using StrideViewConstIter = std::ranges::iterator_t<const StrideView>;

    using StrideViewSentinel      = std::ranges::sentinel_t<StrideView>;
    using StrideViewConstSentinel = std::ranges::sentinel_t<const StrideView>;

    static_assert(!std::convertible_to<StrideViewIter, StrideViewConstIter>);
    static_assert(!std::constructible_from<StrideViewConstIter, StrideViewIter>);
  }

  {
    // Stride over non-simple view over whose iterators are not convertible -- should not be able
    // to copy construct the stride view's iterator.
    using NotSimpleViewBeingStrided              = NotSimpleViewDifferentEnd<false, true>;
    using NotSimpleViewBeingStridedIterator      = std::ranges::iterator_t<NotSimpleViewBeingStrided>;
    using NotSimpleViewBeingStridedConstIterator = std::ranges::iterator_t<const NotSimpleViewBeingStrided>;

    using StrideView = std::ranges::stride_view<NotSimpleViewBeingStrided>;

    using StrideViewIter      = std::ranges::iterator_t<StrideView>;
    using StrideViewConstIter = std::ranges::iterator_t<const StrideView>;

    using StrideViewSentinel      = std::ranges::sentinel_t<StrideView>;
    using StrideViewConstSentinel = std::ranges::sentinel_t<const StrideView>;

    static_assert(std::convertible_to<NotSimpleViewBeingStridedIterator, NotSimpleViewBeingStridedIterator>);
    static_assert(std::convertible_to<StrideViewIter, StrideViewConstIter>);

    StrideView str{NotSimpleViewBeingStrided{}, 5};
    // Confirm (5)
    ASSERT_SAME_TYPE(StrideViewIter, decltype(str.begin()));

    // Now, do what we wanted the whole time: make sure that we can copy construct a
    // stride_view::iterator<true> from a stride_view::iterator<false>. The copy
    // constructor requires that the new __current_ StrideViewConstIter (type
    // NotSimpleViewBeingStridedConstIterator) be constructable
    // from the moved str.begin() __current_ (type NotSimpleViewBeingStridedConstIterator).
    StrideViewConstIter iterator_copy{str.begin()};
  }

  {
    // Stride over non-simple view over whose iterators are copy convertible -- should look (statically)
    // like it is possible copy construct the stride view's iterator (the move-only requirement comes from
    // a move of the current between the copied-from iterator to the copied-to iterator).
    using NotSimpleViewBeingStrided              = NotSimpleViewDifferentBegin<true, false>;
    using NotSimpleViewBeingStridedIterator      = std::ranges::iterator_t<NotSimpleViewBeingStrided>;
    using NotSimpleViewBeingStridedConstIterator = std::ranges::iterator_t<const NotSimpleViewBeingStrided>;

    using StrideView = std::ranges::stride_view<NotSimpleViewBeingStrided>;

    using StrideViewIter      = std::ranges::iterator_t<StrideView>;
    using StrideViewConstIter = std::ranges::iterator_t<const StrideView>;

    using StrideViewSentinel      = std::ranges::sentinel_t<StrideView>;
    using StrideViewConstSentinel = std::ranges::sentinel_t<const StrideView>;

    static_assert(std::convertible_to<StrideViewIter, StrideViewConstIter>);
    static_assert(std::constructible_from<StrideViewConstIter, StrideViewIter>);
  }

  {
    // Stride over non-simple view over whose iterators are move convertible -- should look (statically)
    // like it is possible copy construct the stride view's iterator (the move-only requirement comes from
    // a move of the current between the copied-from iterator to the copied-to iterator).
    using NotSimpleViewBeingStrided              = NotSimpleViewDifferentBegin<false, true>;
    using NotSimpleViewBeingStridedIterator      = std::ranges::iterator_t<NotSimpleViewBeingStrided>;
    using NotSimpleViewBeingStridedConstIterator = std::ranges::iterator_t<const NotSimpleViewBeingStrided>;

    using StrideView = std::ranges::stride_view<NotSimpleViewBeingStrided>;

    using StrideViewIter      = std::ranges::iterator_t<StrideView>;
    using StrideViewConstIter = std::ranges::iterator_t<const StrideView>;

    using StrideViewSentinel      = std::ranges::sentinel_t<StrideView>;
    using StrideViewConstSentinel = std::ranges::sentinel_t<const StrideView>;

    static_assert(std::convertible_to<StrideViewIter, StrideViewConstIter>);
    static_assert(std::constructible_from<StrideViewConstIter, StrideViewIter>);
  }

  {
    // Stride over non-simple view over whose iterators are not convertible -- should not be able
    // to copy construct the stride view's iterator.
    using NotSimpleViewBeingStrided              = NotSimpleViewDifferentBegin<false, false>;
    using NotSimpleViewBeingStridedIterator      = std::ranges::iterator_t<NotSimpleViewBeingStrided>;
    using NotSimpleViewBeingStridedConstIterator = std::ranges::iterator_t<const NotSimpleViewBeingStrided>;

    using StrideView = std::ranges::stride_view<NotSimpleViewBeingStrided>;

    using StrideViewIter      = std::ranges::iterator_t<StrideView>;
    using StrideViewConstIter = std::ranges::iterator_t<const StrideView>;

    using StrideViewSentinel      = std::ranges::sentinel_t<StrideView>;
    using StrideViewConstSentinel = std::ranges::sentinel_t<const StrideView>;

    static_assert(!std::convertible_to<StrideViewIter, StrideViewConstIter>);
    static_assert(!std::constructible_from<StrideViewConstIter, StrideViewIter>);
  }

  {
    // The NotSimpleViewBeingStrided template parameters mean that NotSimpleViewBeingStridedIterator
    // can be move-converted to NotSimpleViewBeingStridedConstIterator but not copy-converted.
    using NotSimpleViewBeingStrided              = NotSimpleViewDifferentBegin<false, true>;
    using NotSimpleViewBeingStridedIterator      = std::ranges::iterator_t<NotSimpleViewBeingStrided>;
    using NotSimpleViewBeingStridedConstIterator = std::ranges::iterator_t<const NotSimpleViewBeingStrided>;

    using StrideView = std::ranges::stride_view<NotSimpleViewBeingStrided>;

    using StrideViewIter      = std::ranges::iterator_t<StrideView>;
    using StrideViewConstIter = std::ranges::iterator_t<const StrideView>;

    using StrideViewSentinel      = std::ranges::sentinel_t<StrideView>;
    using StrideViewConstSentinel = std::ranges::sentinel_t<const StrideView>;

    // Confirm (1) and (2)
    ASSERT_SAME_TYPE(NotSimpleViewBeingStridedIterator, decltype(std::declval<StrideViewIter>().base()));
    ASSERT_SAME_TYPE(NotSimpleViewBeingStridedConstIterator, decltype(std::declval<StrideViewConstIter>().base()));
    // Confirm (3)
    static_assert(std::convertible_to<NotSimpleViewBeingStridedIterator, NotSimpleViewBeingStridedIterator>);
    static_assert(std::convertible_to<StrideViewIter, StrideViewConstIter>);
    // Confirm (4)
    ASSERT_SAME_TYPE(StrideViewSentinel, StrideViewConstSentinel);

    StrideView str{NotSimpleViewBeingStrided{}, 5};
    // Confirm (5)
    ASSERT_SAME_TYPE(StrideViewIter, decltype(str.begin()));

    // Now, do what we wanted the whole time: make sure that we can copy construct a
    // stride_view::iterator<true> from a stride_view::iterator<false>. The copy
    // constructor requires that the new __current_ StrideViewConstIter (type
    // NotSimpleViewBeingStridedConstIterator) be constructable
    // from the moved str.begin() __current_ (type NotSimpleViewBeingStridedConstIterator).
    StrideViewConstIter iterator_copy{str.begin()};
  }
  return true;
}

int main(int, char**) {
  non_simple_view_iter_ctor_test();
  static_assert(non_simple_view_iter_ctor_test());

  return 0;
}
