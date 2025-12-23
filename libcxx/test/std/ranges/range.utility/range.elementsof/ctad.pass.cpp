//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>
//
// template<range R, class Allocator = allocator<byte>>
// struct std::ranges::elements_of;
//
// template<class R, class Allocator = allocator<byte>>
// elements_of(R&&, Allocator = Allocator()) -> elements_of<R&&, Allocator>;

#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <utility>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"

template <class Allocator, class Range>
constexpr void test_impl() {
  Allocator a;

  // With a lvalue range
  {
    Range r;
    std::ranges::elements_of elements(r);
    static_assert(std::same_as<decltype(elements), std::ranges::elements_of<Range&, std::allocator<std::byte>>>);
  }

  // With a rvalue range
  {
    Range r;
    std::ranges::elements_of elements(std::move(r));
    static_assert(std::same_as<decltype(elements), std::ranges::elements_of<Range&&, std::allocator<std::byte>>>);
  }

  // With lvalue range and allocator
  {
    Range r;
    std::ranges::elements_of elements(r, a);
    static_assert(std::same_as<decltype(elements), std::ranges::elements_of<Range&, Allocator>>);
  }

  // With rvalue range and allocator
  {
    Range r;
    std::ranges::elements_of elements(std::move(r), Allocator());
    static_assert(std::same_as<decltype(elements), std::ranges::elements_of<Range&&, Allocator>>);
  }

  // Ensure we can use designated initializers
  {
    // lvalues
    {
      Range r;
      std::ranges::elements_of elements{.range = r, .allocator = a};
      static_assert(std::same_as<decltype(elements), std::ranges::elements_of<Range&, Allocator>>);
    }

    // rvalues
    {
      Range r;
      std::ranges::elements_of elements{.range = std::move(r), .allocator = Allocator()};
      static_assert(std::same_as<decltype(elements), std::ranges::elements_of<Range&&, Allocator>>);
    }
  }
}

template <class Iterator>
struct Range {
  Iterator begin() const;
  sentinel_wrapper<Iterator> end() const;
};

constexpr bool test() {
  types::for_each(types::type_list<std::allocator<std::byte>, min_allocator<std::byte>, test_allocator<std::byte>>{},
                  []<class Allocator> {
                    types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iterator> {
                      test_impl<Allocator, Range<Iterator>>();
                    });
                  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
