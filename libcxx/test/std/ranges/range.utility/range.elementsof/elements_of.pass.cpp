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
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"

template <typename Range>
constexpr bool test_range() {
  std::same_as<std::ranges::elements_of<Range&&, std::allocator<std::byte>>> decltype(auto) elements_of =
      std::ranges::elements_of(Range());
  [[maybe_unused]] std::same_as<Range&&> decltype(auto) elements_of_range = std::move(elements_of.range);
  [[maybe_unused]] std::same_as<std::allocator<std::byte>> decltype(auto) elements_of_allocator = elements_of.allocator;
  return true;
}

template <typename Range, typename Allocator>
constexpr bool test_range_with_allocator() {
  std::same_as< std::ranges::elements_of< Range&&, Allocator >> decltype(auto) elements_of =
      std::ranges::elements_of(Range(), Allocator());
  [[maybe_unused]] std::same_as<Range&&> decltype(auto) elements_of_range       = std::move(elements_of.range);
  [[maybe_unused]] std::same_as<Allocator> decltype(auto) elements_of_allocator = elements_of.allocator;
  return true;
}

constexpr bool test() {
  types::for_each(types::type_list<std::allocator<std::byte>, min_allocator<std::byte>, test_allocator<std::byte>>{},
                  []<class Allocator> {
                    types::for_each(types::type_list<std::vector<int>>{}, []<class Range> {
                      test_range<Range>();
                      test_range_with_allocator<Range, Allocator>();
                    });
                  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
