//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// transform_view::<iterator>::transform_view::<iterator>();

#include <ranges>

#include "test_macros.h"
#include "../types.h"

struct NoDefaultInit {
    typedef std::random_access_iterator_tag iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef NoDefaultInit                   self;

    NoDefaultInit(int*);

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;
    bool operator==(int *) const;

    self& operator++();
    self operator++(int);

    self& operator--();
    self operator--(int);

    self& operator+=(difference_type n);
    self operator+(difference_type n) const;
    friend self operator+(difference_type n, self x);

    self& operator-=(difference_type n);
    self operator-(difference_type n) const;
    difference_type operator-(const self&) const;

    reference operator[](difference_type n) const;
};

struct IterNoDefaultInitView : std::ranges::view_base {
  NoDefaultInit begin() const;
  int *end() const;
  NoDefaultInit begin();
  int *end();
};

constexpr bool test() {
  std::ranges::transform_view<ContiguousView, IncrementConst> transformView;
  auto iter = std::move(transformView).begin();
  std::ranges::iterator_t<std::ranges::transform_view<ContiguousView, IncrementConst>> i2(iter);
  (void)i2;
  std::ranges::iterator_t<const std::ranges::transform_view<ContiguousView, IncrementConst>> constIter(iter);
  (void)constIter;


  static_assert( std::default_initializable<std::ranges::iterator_t<std::ranges::transform_view<ContiguousView, IncrementConst>>>);
  static_assert(!std::default_initializable<std::ranges::iterator_t<std::ranges::transform_view<IterNoDefaultInitView, IncrementConst>>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
