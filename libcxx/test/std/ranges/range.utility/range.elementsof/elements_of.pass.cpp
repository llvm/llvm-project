//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::ranges::elements_of;

#include <ranges>

#include <concepts>
#include <memory>
#include <type_traits>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"

template <class Iterator>
struct Range {
  using Sentinel = sentinel_wrapper<Iterator>;

  Iterator begin() { return Iterator(data_.data()); }

  Sentinel end() { return Sentinel(Iterator(data_.data() + data_.size())); }

private:
  std::vector<int> data_ = {0, 1, 2, 3};
};

template <class Range, class Allocator>
constexpr bool test_range() {
  Range r;

  using elements_of_t = std::ranges::elements_of<Range&, Allocator>;
  {
    // constructor
    std::same_as<elements_of_t> decltype(auto) elements_of                 = std::ranges::elements_of(r, Allocator());
    [[maybe_unused]] std::same_as<Range&> decltype(auto) elements_of_range = elements_of.range;
    if (!std::is_constant_evaluated()) {
      assert(std::ranges::distance(elements_of_range) == 4);
    }
    [[maybe_unused]] std::same_as<Allocator> decltype(auto) elements_of_allocator = elements_of.allocator;
  }
  {
// designated initializer
// AppleClang 15 hasn't implemented P0960R3 and P1816R0
#if defined(__cpp_aggregate_paren_init) && __cpp_aggregate_paren_init >= 201902L && defined(__cpp_deduction_guides) && \
    __cpp_deduction_guides >= 201907L
    std::same_as<elements_of_t> decltype(auto) elements_of = std::ranges::elements_of{
        .range     = r,
        .allocator = Allocator(),
    };
    [[maybe_unused]] std::same_as<Range&> decltype(auto) elements_of_range = elements_of.range;
    if (!std::is_constant_evaluated()) {
      assert(std::ranges::distance(elements_of_range) == 4);
    }
    [[maybe_unused]] std::same_as<Allocator> decltype(auto) elements_of_allocator = elements_of.allocator;
#endif
  }
  {
    // copy constructor
    std::same_as<elements_of_t> decltype(auto) elements_of_1                 = std::ranges::elements_of(r, Allocator());
    std::same_as<elements_of_t> auto elements_of_2                           = elements_of_1;
    [[maybe_unused]] std::same_as<Range&> decltype(auto) elements_of_1_range = elements_of_1.range;
    [[maybe_unused]] std::same_as<Range&> decltype(auto) elements_of_2_range = elements_of_2.range;
    if (!std::is_constant_evaluated()) {
      assert(std::ranges::distance(elements_of_1_range) == 4);
      assert(std::ranges::distance(elements_of_2_range) == 4);
    }
    [[maybe_unused]] std::same_as<Allocator> decltype(auto) elements_of_2_allocator = elements_of_2.allocator;
  }

  using elements_of_r_t = std::ranges::elements_of<Range&&, Allocator>;
  {
    // move constructor
    std::same_as<elements_of_r_t> decltype(auto) elements_of_1 = std::ranges::elements_of(std::move(r), Allocator());
    std::same_as<elements_of_r_t> auto elements_of_2           = std::move(elements_of_1);
    [[maybe_unused]] std::same_as<Range&&> decltype(auto) elements_of_1_range = std::move(elements_of_1.range);
    [[maybe_unused]] std::same_as<Range&&> decltype(auto) elements_of_2_range = std::move(elements_of_2.range);
    if (!std::is_constant_evaluated()) {
      assert(std::ranges::distance(elements_of_1_range) == 4);
      assert(std::ranges::distance(elements_of_2_range) == 4);
    }
    [[maybe_unused]] std::same_as<Allocator> decltype(auto) elements_of_2_allocator = elements_of_2.allocator;
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
