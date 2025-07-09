//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//   template<input_range V>
//     requires view<V>
//   template<bool Const>
//   class as_input_view<V>::iterator

//     constexpr iterator_t<Base> base() &&;
//     constexpr const iterator_t<Base>& base() const & noexcept;

#include <cassert>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <utility>

#include "test_iterators.h"

class CopyConstructibleView : public std::ranges::view_interface<CopyConstructibleView> {
  int* data_;
  std::size_t size_;

public:
  constexpr CopyConstructibleView(int* data, std::size_t size) : data_(data), size_(size) {}

  constexpr CopyConstructibleView(const CopyConstructibleView&)            = default;
  constexpr CopyConstructibleView& operator=(const CopyConstructibleView&) = default;

  constexpr CopyConstructibleView(CopyConstructibleView&&)            = default;
  constexpr CopyConstructibleView& operator=(CopyConstructibleView&&) = default;

  constexpr int* begin() const { return data_; }
  constexpr int* end() const { return data_ + size_; }

  constexpr std::size_t size() const { return size_; }
};

class NonCopyConstructibleView : public std::ranges::view_interface<NonCopyConstructibleView> {
  int* data_;
  std::size_t size_;

public:
  constexpr NonCopyConstructibleView(int* data, std::size_t size) : data_(data), size_(size) {}

  constexpr NonCopyConstructibleView(const NonCopyConstructibleView&)            = delete;
  constexpr NonCopyConstructibleView& operator=(const NonCopyConstructibleView&) = delete;

  constexpr NonCopyConstructibleView(NonCopyConstructibleView&&)            = default;
  constexpr NonCopyConstructibleView& operator=(NonCopyConstructibleView&&) = default;

  constexpr int* begin() const { return data_; }
  constexpr int* end() const { return data_ + size_; }
  constexpr std::size_t size() const { return size_; }
};

constexpr bool test() {
  { // base() &&
    int arr[] = {94};
    CopyConstructibleView range{arr, 1};

    std::ranges::as_input_view<CopyConstructibleView> iv{range};

    auto it = iv.begin();

    using IteratorT = std::ranges::iterator_t<CopyConstructibleView>;

    std::same_as<IteratorT> decltype(auto) base_it = std::move(it).base();
    static_assert(!noexcept(std::move(it).base()));

    assert(base(base_it) == arr);
    assert(*base_it == 94);
  }

  { // base() const & noexcept
    {
      int arr[] = {94};
      CopyConstructibleView range{arr, 1};

      std::ranges::as_input_view<CopyConstructibleView> iv{range};

      auto it = iv.begin();

      using IteratorT = std::ranges::iterator_t<CopyConstructibleView>;

      { // &
        std::same_as<const IteratorT&> decltype(auto) base_it = it.base();
        static_assert(noexcept(it.base()));

        assert(base(base_it) == arr);
        assert(*base_it == 94);
      }
      { // const &
        std::same_as<const IteratorT&> decltype(auto) base_it = std::as_const(it).base();
        static_assert(noexcept(it.base()));

        assert(base(base_it) == arr);
        assert(*base_it == 94);
      }
      { // && - selects base() &&
      }
      { // const &&
        std::same_as<const IteratorT&> decltype(auto) base_it = std::move(std::as_const(it)).base();
        static_assert(noexcept(it.base()));

        assert(base(base_it) == arr);
        assert(*base_it == 94);
      }
    }
    {
      int arr[] = {82};
      NonCopyConstructibleView range{arr, 1};

      const std::ranges::as_input_view<NonCopyConstructibleView> iv{std::move(range)};

      auto it = iv.begin();

      using IteratorT = std::ranges::iterator_t<NonCopyConstructibleView>;

      { // &
        std::same_as<const IteratorT&> decltype(auto) base_it = it.base();
        static_assert(noexcept(it.base()));

        assert(base(base_it) == arr);
        assert(*base_it == 82);
      }
      { // const &
        std::same_as<const IteratorT&> decltype(auto) base_it = std::as_const(it).base();
        static_assert(noexcept(it.base()));

        assert(base(base_it) == arr);
        assert(*base_it == 82);
      }
      { // &&
        std::same_as<IteratorT> decltype(auto) base_it = std::move(it).base();
        static_assert(noexcept(it.base()));

        assert(base(base_it) == arr);
        assert(*base_it == 82);
      }
      { // const&&
        std::same_as<const IteratorT&> decltype(auto) base_it = std::move(std::as_const(it)).base();
        static_assert(noexcept(it.base()));

        assert(base(base_it) == arr);
        assert(*base_it == 82);
      }
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
