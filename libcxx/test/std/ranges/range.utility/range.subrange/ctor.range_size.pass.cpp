//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<borrowed_range R>
//   requires convertible-to-non-slicing<iterator_t<R>, I> &&
//            convertible_to<sentinel_t<R>, S>
// constexpr subrange(R&& r, make-unsigned-like-t<iter_difference_t<I>> n)
//   requires (K == subrange_kind::sized);

#include <ranges>
#include <cassert>

struct BorrowedRange {
  constexpr explicit BorrowedRange(int* b, int* e) : begin_(b), end_(e) { }
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

private:
  int* begin_;
  int* end_;
};

namespace std::ranges {
  template <>
  inline constexpr bool enable_borrowed_range<::BorrowedRange> = true;
}

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};
  using Subrange = std::ranges::subrange<int*, int*, std::ranges::subrange_kind::sized>;

  // Test with an empty range
  {
    BorrowedRange range(buff, buff);
    Subrange subrange(range, 0);
    assert(subrange.size() == 0);
  }

  // Test with non-empty ranges
  {
    BorrowedRange range(buff, buff + 1);
    Subrange subrange(range, 1);
    assert(subrange.size() == 1);
  }
  {
    BorrowedRange range(buff, buff + 2);
    Subrange subrange(range, 2);
    assert(subrange[0] == 1);
    assert(subrange[1] == 2);
    assert(subrange.size() == 2);
  }
  {
    BorrowedRange range(buff, buff + 8);
    Subrange subrange(range, 8);
    assert(subrange[0] == 1);
    assert(subrange[1] == 2);
    // ...
    assert(subrange[7] == 8);
    assert(subrange.size() == 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
