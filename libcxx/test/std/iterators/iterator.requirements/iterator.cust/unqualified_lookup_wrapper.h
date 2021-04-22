//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBCPP_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER
#define LIBCPP_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER

#include <iterator>
#include <utility>

namespace check_unqualified_lookup {
// Wrapper around an iterator for testing unqualified calls to `iter_move` and `iter_swap`.
template <typename I>
class unqualified_lookup_wrapper {
public:
  unqualified_lookup_wrapper() = default;

  constexpr explicit unqualified_lookup_wrapper(I i) noexcept : base_(std::move(i)) {}

  [[nodiscard]] constexpr decltype(auto) operator*() const noexcept { return *base_; }

  constexpr unqualified_lookup_wrapper& operator++() noexcept {
    ++base_;
    return *this;
  }

  constexpr void operator++(int) noexcept { ++base_; }

  [[nodiscard]] constexpr bool operator==(unqualified_lookup_wrapper const& other) const noexcept {
    return base_ == other.base_;
  }

  // Delegates `std::ranges::iter_move` for the underlying iterator. `noexcept(false)` will be used
  // to ensure that the unqualified-lookup overload is chosen.
  [[nodiscard]] friend constexpr decltype(auto) iter_move(unqualified_lookup_wrapper& i) noexcept(false) {
    return std::ranges::iter_move(i.base_);
  }

private:
  I base_ = I{};
};

enum unscoped_enum { a, b, c };
constexpr unscoped_enum iter_move(unscoped_enum& e) noexcept(false) { return e; }

enum class scoped_enum { a, b, c };
constexpr scoped_enum* iter_move(scoped_enum&) noexcept { return nullptr; }

union some_union {
  int x;
  double y;
};
constexpr int iter_move(some_union& u) noexcept(false) { return u.x; }

} // namespace check_unqualified_lookup

#endif // LIBCPP_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER
