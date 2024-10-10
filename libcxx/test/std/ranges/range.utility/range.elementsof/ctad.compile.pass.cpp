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

#include <memory>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"

template <class Iterator>
struct Range {
  Iterator begin() const;
  sentinel_wrapper<Iterator> end() const;
};

constexpr bool test() {
  types::for_each(
      types::type_list<std::allocator<std::byte>, min_allocator<std::byte>, test_allocator<std::byte>>{},
      []<class Allocator> {
        types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iterator> {
          Range<Iterator> r;
          static_assert(std::same_as<decltype(std::ranges::elements_of(r)),
                                     std::ranges::elements_of<Range<Iterator>&, std::allocator<std::byte>>>);
          static_assert(std::same_as<decltype(std::ranges::elements_of(Range<Iterator>())),
                                     std::ranges::elements_of<Range<Iterator>&&, std::allocator<std::byte>>>);

          Allocator a;
          static_assert(std::same_as<decltype(std::ranges::elements_of(r, a)),
                                     std::ranges::elements_of<Range<Iterator>&, Allocator>>);
          static_assert(std::same_as<decltype(std::ranges::elements_of(Range<Iterator>(), Allocator())),
                                     std::ranges::elements_of<Range<Iterator>&&, Allocator>>);

// AppleClang 15 hasn't implemented P0960R3 and P1816R0
#if defined(__cpp_aggregate_paren_init) && __cpp_aggregate_paren_init >= 201902L && defined(__cpp_deduction_guides) && \
    __cpp_deduction_guides >= 201907L
          static_assert(std::same_as<decltype(std::ranges::elements_of{.range = r, .allocator = a}),
                                     std::ranges::elements_of<Range<Iterator>&, Allocator>>);
          static_assert(
              std::same_as<decltype(std::ranges::elements_of{.range = Range<Iterator>(), .allocator = Allocator()}),
                           std::ranges::elements_of<Range<Iterator>&&, Allocator>>);
#endif
        });
      });

  return true;
}

static_assert(test());
