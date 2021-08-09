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

// std::views::counted;

#include <concepts>
#include <ranges>
#include <span>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

struct Unrelated {};

struct ConvertibleToSize {
  constexpr operator std::ptrdiff_t() const { return 8; }
};

struct ImplicitlyConvertible {
    operator short();
    explicit operator std::ptrdiff_t() = delete;
};

template<class Iter, class T>
concept CountedInvocable = requires(Iter& i, T t) { std::views::counted(i, t); };

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( CountedInvocable<contiguous_iterator<int*>, ConvertibleToSize>);
    static_assert(!CountedInvocable<contiguous_iterator<int*>, ImplicitlyConvertible>);
    static_assert(!CountedInvocable<contiguous_iterator<int*>, Unrelated>);

    static_assert(std::semiregular<std::remove_const_t<decltype(std::views::counted)>>);
  }

  {
    {
      contiguous_iterator<int*> iter(buffer);
      std::span<int> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.data() == buffer);

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)), std::span<int>);
    }
    {
      const contiguous_iterator<int*> iter(buffer);
      std::span<int> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.data() == buffer);

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)), std::span<int>);
    }
    {
      contiguous_iterator<const int*> iter(buffer);
      std::span<const int> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.data() == buffer);

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)), std::span<const int>);
    }
    {
      const contiguous_iterator<const int*> iter(buffer);
      std::span<const int> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.data() == buffer);

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)), std::span<const int>);
    }
  }

  {
    {
      random_access_iterator<int*> iter(buffer);
      std::ranges::subrange<random_access_iterator<int*>> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.begin() == iter);

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)), std::ranges::subrange<random_access_iterator<int*>>);
    }
    {
      const random_access_iterator<int*> iter(buffer);
      std::ranges::subrange<random_access_iterator<int*>> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.begin() == iter);

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)), std::ranges::subrange<random_access_iterator<int*>>);
    }
    {
      random_access_iterator<const int*> iter(buffer);
      std::ranges::subrange<random_access_iterator<const int*>> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.begin() == iter);

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)), std::ranges::subrange<random_access_iterator<const int*>>);
    }
    {
      const random_access_iterator<const int*> iter(buffer);
      std::ranges::subrange<random_access_iterator<const int*>> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.begin() == iter);

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)), std::ranges::subrange<random_access_iterator<const int*>>);
    }
  }

  {
    {
      bidirectional_iterator<int*> iter(buffer);
      std::ranges::subrange<
        std::counted_iterator<bidirectional_iterator<int*>>,
        std::default_sentinel_t> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.begin() == std::counted_iterator(iter, 8));

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)),
                       std::ranges::subrange<
                         std::counted_iterator<bidirectional_iterator<int*>>,
                         std::default_sentinel_t>);
    }
    {
      const bidirectional_iterator<int*> iter(buffer);
      std::ranges::subrange<
        std::counted_iterator<bidirectional_iterator<int*>>,
        std::default_sentinel_t> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.begin() == std::counted_iterator(iter, 8));

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)),
                       std::ranges::subrange<
                         std::counted_iterator<bidirectional_iterator<int*>>,
                         std::default_sentinel_t>);
    }
    {
      output_iterator<const int*> iter(buffer);
      std::ranges::subrange<
        std::counted_iterator<output_iterator<const int*>>,
        std::default_sentinel_t> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.begin() == std::counted_iterator(iter, 8));

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)),
                       std::ranges::subrange<
                         std::counted_iterator<output_iterator<const int*>>,
                         std::default_sentinel_t>);
    }
    {
      const output_iterator<const int*> iter(buffer);
      std::ranges::subrange<
        std::counted_iterator<output_iterator<const int*>>,
        std::default_sentinel_t> s = std::views::counted(iter, 8);
      assert(s.size() == 8);
      assert(s.begin() == std::counted_iterator(iter, 8));

      ASSERT_SAME_TYPE(decltype(std::views::counted(iter, 8)),
                       std::ranges::subrange<
                         std::counted_iterator<output_iterator<const int*>>,
                         std::default_sentinel_t>);
    }
    {
      cpp20_input_iterator<int*> iter(buffer);
      std::ranges::subrange<
        std::counted_iterator<cpp20_input_iterator<int*>>,
        std::default_sentinel_t> s = std::views::counted(std::move(iter), 8);
      assert(s.size() == 8);
      assert(s.begin().base().base() == buffer);

      ASSERT_SAME_TYPE(decltype(std::views::counted(std::move(iter), 8)),
                       std::ranges::subrange<
                         std::counted_iterator<cpp20_input_iterator<int*>>,
                         std::default_sentinel_t>);
    }
    {
      std::ranges::subrange<
        std::counted_iterator<cpp20_input_iterator<int*>>,
        std::default_sentinel_t> s = std::views::counted(cpp20_input_iterator<int*>(buffer), 8);
      assert(s.size() == 8);
      assert(s.begin().base().base() == buffer);

      ASSERT_SAME_TYPE(decltype(std::views::counted(cpp20_input_iterator<int*>(buffer), 8)),
                       std::ranges::subrange<
                         std::counted_iterator<cpp20_input_iterator<int*>>,
                         std::default_sentinel_t>);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
