//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::views::stride_view

#include "../test.h"
#include "__concepts/convertible_to.h"
#include "__iterator/concepts.h"
#include "__ranges/access.h"
#include "__ranges/concepts.h"
#include "__ranges/stride_view.h"
#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

struct NotSimpleViewIter : InputIterBase<NotSimpleViewIter> {};
struct NotSimpleViewIterEnd : InputIterBase<NotSimpleViewIterEnd> {};
constexpr bool operator==(const NotSimpleViewIter&, const NotSimpleViewIterEnd&) { return true; }
constexpr bool operator==(const NotSimpleViewIterEnd&, const NotSimpleViewIter&) { return true; }

template <bool Convertible, bool Copyable>
struct NotSimpleViewConstIterEnd : InputIterBase<NotSimpleViewConstIterEnd<Convertible, Copyable>> {
  constexpr NotSimpleViewConstIterEnd() = default;
  constexpr NotSimpleViewConstIterEnd(const NotSimpleViewConstIterEnd&&) {}
  constexpr NotSimpleViewConstIterEnd& operator=(const NotSimpleViewConstIterEnd&) {}
  constexpr NotSimpleViewConstIterEnd& operator=(const NotSimpleViewConstIterEnd&&) {}

  constexpr NotSimpleViewConstIterEnd(const NotSimpleViewConstIterEnd&)
    requires Copyable
  {}
  constexpr NotSimpleViewConstIterEnd(const NotSimpleViewIterEnd&)
    requires Convertible
  {}
};

template <bool Convertible, bool Copyable>
struct NotSimpleViewConstIter : InputIterBase<NotSimpleViewConstIter<Convertible, Copyable>> {
  constexpr NotSimpleViewConstIter() = default;
  constexpr NotSimpleViewConstIter(const NotSimpleViewConstIter&&) {}
  constexpr NotSimpleViewConstIter& operator=(const NotSimpleViewConstIter&&) {}
  constexpr NotSimpleViewConstIter& operator=(const NotSimpleViewConstIter&) {}

  constexpr NotSimpleViewConstIter(const NotSimpleViewConstIter&)
    requires Copyable
  {}
  constexpr NotSimpleViewConstIter(const NotSimpleViewIter&)
    requires Convertible
  {}
};

template <bool Convertible, bool Copyable>
constexpr bool operator==(const NotSimpleViewConstIter<Convertible, Copyable>&, const NotSimpleViewIterEnd&) {
  return true;
}
template <bool Convertible, bool Copyable>
constexpr bool operator==(const NotSimpleViewIterEnd&, const NotSimpleViewConstIter<Convertible, Copyable>&) {
  return true;
}
template <bool Convertible, bool Copyable>
constexpr bool operator==(const NotSimpleViewConstIterEnd<Convertible, Copyable>&, const NotSimpleViewIterEnd&) {
  return true;
}
template <bool Convertible, bool Copyable>
constexpr bool operator==(const NotSimpleViewIterEnd&, const NotSimpleViewConstIterEnd<Convertible, Copyable>&) {
  return true;
}
template <bool Convertible, bool Copyable>
constexpr bool operator==(const NotSimpleViewIter&, const NotSimpleViewConstIterEnd<Convertible, Copyable>&) {
  return true;
}
template <bool Convertible, bool Copyable>
constexpr bool operator==(const NotSimpleViewConstIterEnd<Convertible, Copyable>&, const NotSimpleViewIter&) {
  return true;
}

/*
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
template <bool Convertible = false, bool Copyable = true>
struct NotSimpleViewDifferentBegin : std::ranges::view_base {
  constexpr NotSimpleViewConstIter<Convertible, Copyable> begin() const { return {}; }
  constexpr NotSimpleViewIter begin() { return {}; }
  constexpr NotSimpleViewIterEnd end() const { return {}; }
  constexpr NotSimpleViewIterEnd end() { return {}; }
};

template <bool Convertible = false, bool Copyable = true>
struct NotSimpleViewDifferentEnd : std::ranges::view_base {
  constexpr NotSimpleViewIter begin() const { return {}; }
  constexpr NotSimpleViewIter begin() { return {}; }
  constexpr NotSimpleViewConstIterEnd<Convertible, Copyable> end() const {
    return std::move(NotSimpleViewConstIterEnd<Convertible, Copyable>{});
  }
  constexpr NotSimpleViewIterEnd end() { return {}; }
};

constexpr bool non_simple_view_iter_ctor_test() {
  using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentBegin<false>>;
  using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
  using NotSimpleStrideViewIterConst = std::ranges::iterator_t<const NotSimpleStrideView>;
  static_assert(!std::is_same_v<NotSimpleStrideViewIterConst, NotSimpleStrideViewIter>);
  return true;
}

struct NonDefaultConstructibleIterator : InputIterBase<NonDefaultConstructibleIterator> {
  NonDefaultConstructibleIterator() = delete;
  constexpr NonDefaultConstructibleIterator(int) {}
};

struct ViewWithNonDefaultConstructibleIterator : std::ranges::view_base {
  constexpr NonDefaultConstructibleIterator begin() const { return NonDefaultConstructibleIterator{5}; }
  constexpr std::default_sentinel_t end() const { return {}; }
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<ViewWithNonDefaultConstructibleIterator> = true;

constexpr bool iterator_default_constructible() {
  {
    // If the type of the iterator of the range being strided is non-default
    // constructible, then the stride view's iterator should not be default
    // constructible, either!
    constexpr ViewWithNonDefaultConstructibleIterator v{};
    constexpr auto stride   = std::ranges::stride_view(v, 1);
    using stride_iterator_t = decltype(stride.begin());
    static_assert(!std::is_default_constructible<stride_iterator_t>());
  }
  {
    // If the type of the iterator of the range being strided is default
    // constructible, then the stride view's iterator should be default
    // constructible, too!
    constexpr int arr[]     = {1, 2, 3};
    auto stride             = std::ranges::stride_view(arr, 1);
    using stride_iterator_t = decltype(stride.begin());
    static_assert(std::is_default_constructible<stride_iterator_t>());
  }

  return true;
}

constexpr bool non_const_iterator_copy_ctor() {
  {
    // Instantiate a stride view over a non-simple view whose const/non-const begin iterators are not-convertible.
    using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentBegin<false, true>>;
    using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
    using NotSimpleStrideViewConstIter = std::ranges::iterator_t<const NotSimpleStrideView>;

    // It should not be possible to construct a stride view iterator from a non-const stride view iterator
    // when the strided-over type has inconvertible begin iterator types.
    static_assert(!std::ranges::__simple_view<NotSimpleStrideView>);
    static_assert(!std::convertible_to<NotSimpleStrideViewIter, NotSimpleStrideViewConstIter>);
    static_assert(!std::is_constructible_v<NotSimpleStrideViewConstIter, NotSimpleStrideViewIter>);
  }
  {
    // Instantiate a stride view over a non-simple view whose const/non-const begin iterators are convertible.
    using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentBegin<true, true>>;
    using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
    using NotSimpleStrideViewConstIter = std::ranges::iterator_t<const NotSimpleStrideView>;

    // It should be possible to construct a stride view iterator from a non-const stride view iterator
    // when the strided-over type has convertible begin iterator types.
    static_assert(!std::ranges::__simple_view<NotSimpleStrideView>);
    static_assert(std::convertible_to<NotSimpleStrideViewIter, NotSimpleStrideViewConstIter>);
    static_assert(std::is_constructible_v<NotSimpleStrideViewConstIter, NotSimpleStrideViewIter>);
  }

  {
    // Instantiate a stride view over a non-simple view whose const/non-const end iterators are not convertible.
    using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentEnd<false, true>>;
    using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
    using NotSimpleStrideViewConstIter = std::ranges::iterator_t<const NotSimpleStrideView>;

    static_assert(std::ranges::__can_borrow<const NotSimpleStrideView&>);

    // It should not be possible to construct a stride view iterator from a non-const stride view iterator
    // when the strided-over type has inconvertible end iterator types.
    static_assert(!std::ranges::__simple_view<NotSimpleStrideView>);
    static_assert(!std::convertible_to<NotSimpleStrideViewIter, NotSimpleStrideViewConstIter>);
    static_assert(!std::is_constructible_v<NotSimpleStrideViewConstIter, NotSimpleStrideViewIter>);
  }

  {
    // Instantiate a stride view over a non-simple view whose const/non-const end iterators are convertible.
    using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentEnd<true, true>>;
    using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
    using NotSimpleStrideViewConstIter = std::ranges::iterator_t<const NotSimpleStrideView>;

    // It should not be possible to construct a stride view iterator from a non-const stride view iterator
    // when the strided-over type has inconvertible end iterator types.
    static_assert(std::is_copy_constructible_v<NotSimpleStrideViewConstIter>);
    static_assert(!std::ranges::__simple_view<NotSimpleStrideView>);
    static_assert(std::convertible_to<NotSimpleStrideViewIter, NotSimpleStrideViewConstIter>);
    static_assert(std::is_constructible_v<NotSimpleStrideViewConstIter, NotSimpleStrideViewIter>);
  }

  {
    // Instantiate a stride view over a non-simple view whose iterators are not copyable but whose const
    // and non-const end iterators are convertible.
    using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentBegin<true, false>>;
    using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
    using NotSimpleStrideViewConstIter = std::ranges::iterator_t<const NotSimpleStrideView>;

    // It should not be possible to copy construct a stride view iterator from a non-const stride view iterator
    // when the strided-over type has non copyable end iterator type.
    static_assert(!std::is_copy_constructible_v<NotSimpleStrideViewConstIter>);

    // Given the difference between the (non-) constness of the end iterator types and the fact that
    // they can be converted between, it should
    // 1. not be a simple view
    static_assert(!std::ranges::__simple_view<NotSimpleStrideView>);
    // 2. the types should be convertible
    static_assert(std::convertible_to<NotSimpleStrideViewIter, NotSimpleStrideViewConstIter>);
    // 3. and a const thing should be constructible from a non const thing because they are convertible.
    static_assert(std::is_constructible_v<NotSimpleStrideViewConstIter, NotSimpleStrideViewIter>);
  }
  return true;
}

int main(int, char**) {
  non_simple_view_iter_ctor_test();
  static_assert(non_simple_view_iter_ctor_test());
  static_assert(iterator_default_constructible());
  static_assert(non_const_iterator_copy_ctor());

  return 0;
}
