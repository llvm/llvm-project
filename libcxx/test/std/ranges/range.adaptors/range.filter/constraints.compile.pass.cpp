//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Check constraints on the type itself.
//
// template<input_range View, indirect_unary_predicate<iterator_t<View>> Pred>
//    requires view<View> && is_object_v<Pred>
// class filter_view;

#include <ranges>

#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>

template <class View, class Pred>
concept can_form_filter_view = requires {
  typename std::ranges::filter_view<View, Pred>;
};

// filter_view is not valid when the view is not an input_range
namespace test1 {
  struct View : std::ranges::view_base {
    struct NotInputIterator {
      NotInputIterator& operator++();
      void operator++(int);
      int& operator*() const;
      using difference_type = std::ptrdiff_t;
      friend bool operator==(NotInputIterator const&, NotInputIterator const&);
    };
    NotInputIterator begin() const;
    NotInputIterator end() const;
  };
  struct Pred { bool operator()(int) const; };

  static_assert(!std::ranges::input_range<View>);
  static_assert( std::indirect_unary_predicate<Pred, int*>);
  static_assert( std::ranges::view<View>);
  static_assert( std::is_object_v<Pred>);
  static_assert(!can_form_filter_view<View, Pred>);
}

// filter_view is not valid when the predicate is not indirect_unary_predicate
namespace test2 {
  struct View : std::ranges::view_base {
    int* begin() const;
    int* end() const;
  };
  struct Pred { };

  static_assert( std::ranges::input_range<View>);
  static_assert(!std::indirect_unary_predicate<Pred, int*>);
  static_assert( std::ranges::view<View>);
  static_assert( std::is_object_v<Pred>);
  static_assert(!can_form_filter_view<View, Pred>);
}

// filter_view is not valid when the view is not a view
namespace test3 {
  struct View {
    int* begin() const;
    int* end() const;
  };
  struct Pred { bool operator()(int) const; };

  static_assert( std::ranges::input_range<View>);
  static_assert( std::indirect_unary_predicate<Pred, int*>);
  static_assert(!std::ranges::view<View>);
  static_assert( std::is_object_v<Pred>);
  static_assert(!can_form_filter_view<View, Pred>);
}

// filter_view is not valid when the predicate is not an object type
namespace test4 {
  struct View : std::ranges::view_base {
    int* begin() const;
    int* end() const;
  };
  using Pred = bool(&)(int);

  static_assert( std::ranges::input_range<View>);
  static_assert( std::indirect_unary_predicate<Pred, int*>);
  static_assert( std::ranges::view<View>);
  static_assert(!std::is_object_v<Pred>);
  static_assert(!can_form_filter_view<View, Pred>);
}

// filter_view is valid when all the constraints are satisfied (test the test)
namespace test5 {
  struct View : std::ranges::view_base {
    int* begin() const;
    int* end() const;
  };
  struct Pred { bool operator()(int) const; };

  static_assert( std::ranges::input_range<View>);
  static_assert( std::indirect_unary_predicate<Pred, int*>);
  static_assert( std::ranges::view<View>);
  static_assert( std::is_object_v<Pred>);
  static_assert( can_form_filter_view<View, Pred>);
}
