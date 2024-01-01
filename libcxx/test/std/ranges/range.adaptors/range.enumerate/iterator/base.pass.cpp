//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// constexpr const iterator_t<Base>& base() const & noexcept;
// constexpr iterator_t<Base> base() &&;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>
#include <tuple>

#include "test_iterators.h"
#include "../types.h"

// template <class It, class ItTraits = It>
// class MovableIterator {
//   using Traits = std::iterator_traits<ItTraits>;
//   It it_;

//   template <class U, class T>
//   friend class MovableIterator;

// public:
//   using iterator_category = std::input_iterator_tag;
//   using value_type        = typename Traits::value_type;
//   using difference_type   = typename Traits::difference_type;
//   using pointer           = It;
//   using reference         = typename Traits::reference;

//   TEST_CONSTEXPR explicit MovableIterator(It it) : it_(it), justInitialized{true} { static_assert(false); }

//   template <class U, class T>
//   TEST_CONSTEXPR MovableIterator(const MovableIterator<U, T>& u) : it_(u.it_), wasCopyInitialized{true} {
//     static_assert(false);
//   }

//   template <class U, class T, class = typename std::enable_if<std::is_default_constructible<U>::value>::type>
//   TEST_CONSTEXPR_CXX14 MovableIterator(MovableIterator<U, T>&& u) : it_(u.it_), wasMoveInitialized{true} {
//     static_assert(false);
//     u.it_ = U();
//   }

//   TEST_CONSTEXPR reference operator*() const { return *it_; }

//   TEST_CONSTEXPR_CXX14 MovableIterator& operator++() {
//     ++it_;
//     return *this;
//   }
//   TEST_CONSTEXPR_CXX14 MovableIterator operator++(int) { return MovableIterator(it_++); }

//   friend TEST_CONSTEXPR bool operator==(const MovableIterator& x, const MovableIterator& y) { return x.it_ == y.it_; }
//   friend TEST_CONSTEXPR bool operator!=(const MovableIterator& x, const MovableIterator& y) { return x.it_ != y.it_; }

//   friend TEST_CONSTEXPR It base(const MovableIterator& i) { return i.it_; }

//   template <class T>
//   void operator,(T const&) = delete;

//   bool justInitialized    = false;
//   bool wasCopyInitialized = false;
//   bool wasMoveInitialized = false;
// };

template <class Iterator>
constexpr void testBase() {
  using Sentinel          = sentinel_wrapper<Iterator>;
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateIterator = std::ranges::iterator_t<EnumerateView>;

  auto make_enumerate_view = [](auto begin, auto end) {
    View view{Iterator(begin), Sentinel(Iterator(end))};

    return EnumerateView(std::move(view));
  };

  std::array array{0, 1, 2, 3, 84};
  const auto view = make_enumerate_view(array.begin(), array.end());

  // Test the const& version
  {
    EnumerateIterator const it                          = view.begin();
    std::same_as<const Iterator&> decltype(auto) result = it.base();
    ASSERT_NOEXCEPT(it.base());
    assert(base(result) == array.begin());
  }

  // Test the && version
  {
    EnumerateIterator it                         = view.begin();
    std::same_as<Iterator> decltype(auto) result = std::move(it).base();
    assert(base(result) == array.begin());

    // // Test move
    // if constexpr (std::same_as<Iterator, MovableIterator<int*>>) {
    //   assert(result.justInitialized);
    //   assert(!result.wasCopyInitialized);
    //   assert(!result.wasMoveInitialized);
    // }
  }
}

constexpr bool test() {
  // testBase<MovableIterator<int*>>();
  testBase<cpp17_input_iterator<int*>>();
  testBase<cpp20_input_iterator<int*>>();
  testBase<forward_iterator<int*>>();
  testBase<bidirectional_iterator<int*>>();
  testBase<random_access_iterator<int*>>();
  testBase<contiguous_iterator<int*>>();
  testBase<int*>();
  testBase<int const*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
