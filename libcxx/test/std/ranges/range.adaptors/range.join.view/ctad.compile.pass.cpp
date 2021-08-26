//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class R>
//   explicit join_view(R&&) -> join_view<views::all_t<R>>;

#include <ranges>

#include "test_iterators.h"

template<class T>
struct View : std::ranges::view_base {
  // All friends here are defined to prevent GCC warnings.
  friend T* begin(View&) { return nullptr; }
  friend T* begin(View const&) { return nullptr; }
  friend sentinel_wrapper<T*> end(View&) { return sentinel_wrapper<T*>(nullptr); }
  friend sentinel_wrapper<T*> end(View const&) { return sentinel_wrapper<T*>(nullptr); }
};

template<class T>
struct Range {
  friend T* begin(Range&) { return nullptr; }
  friend T* begin(Range const&) { return nullptr; }
  friend sentinel_wrapper<T*> end(Range&) { return sentinel_wrapper<T*>(nullptr); }
  friend sentinel_wrapper<T*> end(Range const&) { return sentinel_wrapper<T*>(nullptr); }
};

template<class T>
struct BorrowedRange {
  friend T* begin(BorrowedRange&) { return nullptr; }
  friend T* begin(BorrowedRange const&) { return nullptr; }
  friend sentinel_wrapper<T*> end(BorrowedRange&) { return sentinel_wrapper<T*>(nullptr); }
  friend sentinel_wrapper<T*> end(BorrowedRange const&) { return sentinel_wrapper<T*>(nullptr); }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange<BorrowedRange<int>>> = true;

void testCTAD() {
    View<View<int>> v;
    Range<Range<int>> r;
    BorrowedRange<BorrowedRange<int>> br;

    static_assert(std::same_as<
        decltype(std::ranges::join_view(v)),
        std::ranges::join_view<View<View<int>>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::join_view(r)),
        std::ranges::join_view<std::ranges::ref_view<Range<Range<int>>>>
    >);
    // std::ranges::join_view(std::move(r)) invalid. RValue range must be borrowed.
    static_assert(std::same_as<
        decltype(std::ranges::join_view(br)),
        std::ranges::join_view<std::ranges::ref_view<BorrowedRange<BorrowedRange<int>>>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::join_view(std::move(br))),
        std::ranges::join_view<std::ranges::subrange<BorrowedRange<int> *,
                               sentinel_wrapper<BorrowedRange<int> *>,
                               std::ranges::subrange_kind::unsized>>
    >);
}
