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

#include <cassert>
#include <concepts>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "type_algorithms.h"

template <class Iterator>
struct Range {
  using Sentinel = sentinel_wrapper<Iterator>;

  constexpr Iterator begin() { return Iterator(data_.data()); }

  constexpr Sentinel end() { return Sentinel(Iterator(data_.data() + data_.size())); }

private:
  std::vector<int> data_ = {0, 1, 2, 3};
};

template <class Range, class Allocator>
constexpr bool test_range() {
  Range r;

  using elements_of_t = std::ranges::elements_of<Range&, Allocator>;
  {
    // constructor
    std::same_as<elements_of_t> decltype(auto) e = std::ranges::elements_of(r, Allocator());
    std::same_as<Range&> decltype(auto) range    = e.range;
    assert(std::ranges::distance(range) == 4);
    [[maybe_unused]] std::same_as<Allocator> decltype(auto) allocator = e.allocator;
  }
  {
    // designated initializer
    std::same_as<elements_of_t> decltype(auto) e = std::ranges::elements_of{
        .range     = r,
        .allocator = Allocator(),
    };
    std::same_as<Range&> decltype(auto) range = e.range;
    assert(&range == &r);
    assert(std::ranges::distance(range) == 4);
    [[maybe_unused]] std::same_as<Allocator> decltype(auto) allocator = e.allocator;
  }
  {
    // copy constructor
    std::same_as<elements_of_t> decltype(auto) e   = std::ranges::elements_of(r, Allocator());
    std::same_as<elements_of_t> auto copy          = e;
    std::same_as<Range&> decltype(auto) range      = e.range;
    std::same_as<Range&> decltype(auto) copy_range = copy.range;
    assert(&range == &r);
    assert(&range == &copy_range);
    assert(std::ranges::distance(range) == 4);
    [[maybe_unused]] std::same_as<Allocator> decltype(auto) copy_allocator = copy.allocator;
  }

  using elements_of_r_t = std::ranges::elements_of<Range&&, Allocator>;
  {
    // move constructor
    std::same_as<elements_of_r_t> decltype(auto) e  = std::ranges::elements_of(std::move(r), Allocator());
    std::same_as<elements_of_r_t> auto copy         = std::move(e);
    std::same_as<Range&&> decltype(auto) range      = std::move(e.range);
    std::same_as<Range&&> decltype(auto) copy_range = std::move(copy.range);
    assert(&range == &r);
    assert(&range == &copy_range);
    assert(std::ranges::distance(range) == 4);
    [[maybe_unused]] std::same_as<Allocator> decltype(auto) copy_allocator = copy.allocator;
  }
  return true;
}

constexpr bool test() {
  types::for_each(types::type_list<std::allocator<std::byte>, min_allocator<std::byte>, test_allocator<std::byte>>{},
                  []<class Allocator> {
                    types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iterator> {
                      test_range<Range<Iterator>, Allocator>();
                    });
                  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
