//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::counted;

#include <ranges>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>

#include "test_macros.h"
#include "test_iterators.h"

struct RvalueConvertible {
  RvalueConvertible(const RvalueConvertible&) = delete;
  operator int() &&;
};

struct LvalueConvertible {
  LvalueConvertible(const LvalueConvertible&) = delete;
  operator int() &;
};

struct OnlyExplicitlyConvertible {
  explicit operator int() const;
};

template<class... Ts>
concept CountedInvocable = requires (Ts&&... ts) {
  std::views::counted(std::forward<Ts>(ts)...);
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert(std::addressof(std::views::counted) == std::addressof(std::ranges::views::counted));

    static_assert( CountedInvocable<int*, std::size_t>);
    static_assert(!CountedInvocable<int*, LvalueConvertible>);
    static_assert( CountedInvocable<int*, LvalueConvertible&>);
    static_assert( CountedInvocable<int*, RvalueConvertible>);
    static_assert(!CountedInvocable<int*, RvalueConvertible&>);
    static_assert(!CountedInvocable<int*, OnlyExplicitlyConvertible>);
    static_assert(!CountedInvocable<int*, int*>);
    static_assert(!CountedInvocable<int*>);
    static_assert(!CountedInvocable<std::size_t>);
    static_assert(!CountedInvocable<>);
  }

  {
    auto c1 = std::views::counted(buffer, 3);
    auto c2 = std::views::counted(std::as_const(buffer), 3);

    ASSERT_SAME_TYPE(decltype(c1), std::span<int>);
    ASSERT_SAME_TYPE(decltype(c2), std::span<const int>);

    assert(c1.data() == buffer && c1.size() == 3);
    assert(c2.data() == buffer && c2.size() == 3);
  }

  {
    auto it = contiguous_iterator<int*>(buffer);
    auto cit = contiguous_iterator<const int*>(buffer);

    auto c1 = std::views::counted(it, 3);
    auto c2 = std::views::counted(std::as_const(it), 3);
    auto c3 = std::views::counted(std::move(it), 3);
    auto c4 = std::views::counted(contiguous_iterator<int*>(buffer), 3);
    auto c5 = std::views::counted(cit, 3);
    auto c6 = std::views::counted(std::as_const(cit), 3);
    auto c7 = std::views::counted(std::move(cit), 3);
    auto c8 = std::views::counted(contiguous_iterator<const int*>(buffer), 3);

    ASSERT_SAME_TYPE(decltype(c1), std::span<int>);
    ASSERT_SAME_TYPE(decltype(c2), std::span<int>);
    ASSERT_SAME_TYPE(decltype(c3), std::span<int>);
    ASSERT_SAME_TYPE(decltype(c4), std::span<int>);
    ASSERT_SAME_TYPE(decltype(c5), std::span<const int>);
    ASSERT_SAME_TYPE(decltype(c6), std::span<const int>);
    ASSERT_SAME_TYPE(decltype(c7), std::span<const int>);
    ASSERT_SAME_TYPE(decltype(c8), std::span<const int>);

    assert(c1.data() == buffer && c1.size() == 3);
    assert(c2.data() == buffer && c2.size() == 3);
    assert(c3.data() == buffer && c3.size() == 3);
    assert(c4.data() == buffer && c4.size() == 3);
    assert(c5.data() == buffer && c5.size() == 3);
    assert(c6.data() == buffer && c6.size() == 3);
    assert(c7.data() == buffer && c7.size() == 3);
    assert(c8.data() == buffer && c8.size() == 3);
  }

  {
    auto it = random_access_iterator<int*>(buffer);
    auto cit = random_access_iterator<const int*>(buffer);

    auto c1 = std::views::counted(it, 3);
    auto c2 = std::views::counted(std::as_const(it), 3);
    auto c3 = std::views::counted(std::move(it), 3);
    auto c4 = std::views::counted(random_access_iterator<int*>(buffer), 3);
    auto c5 = std::views::counted(cit, 3);
    auto c6 = std::views::counted(std::as_const(cit), 3);
    auto c7 = std::views::counted(std::move(cit), 3);
    auto c8 = std::views::counted(random_access_iterator<const int*>(buffer), 3);

    ASSERT_SAME_TYPE(decltype(c1), std::ranges::subrange<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(c2), std::ranges::subrange<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(c3), std::ranges::subrange<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(c4), std::ranges::subrange<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(c5), std::ranges::subrange<random_access_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(c6), std::ranges::subrange<random_access_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(c7), std::ranges::subrange<random_access_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(c8), std::ranges::subrange<random_access_iterator<const int*>>);

    assert(c1.begin() == it && c1.end() == it + 3);
    assert(c2.begin() == it && c2.end() == it + 3);
    assert(c3.begin() == it && c3.end() == it + 3);
    assert(c4.begin() == it && c4.end() == it + 3);
    assert(c5.begin() == cit && c5.end() == cit + 3);
    assert(c6.begin() == cit && c6.end() == cit + 3);
    assert(c7.begin() == cit && c7.end() == cit + 3);
    assert(c8.begin() == cit && c8.end() == cit + 3);
  }

  {
    auto it = bidirectional_iterator<int*>(buffer);
    auto cit = bidirectional_iterator<const int*>(buffer);

    auto c1 = std::views::counted(it, 3);
    auto c2 = std::views::counted(std::as_const(it), 3);
    auto c3 = std::views::counted(std::move(it), 3);
    auto c4 = std::views::counted(bidirectional_iterator<int*>(buffer), 3);
    auto c5 = std::views::counted(cit, 3);
    auto c6 = std::views::counted(std::as_const(cit), 3);
    auto c7 = std::views::counted(std::move(cit), 3);
    auto c8 = std::views::counted(bidirectional_iterator<const int*>(buffer), 3);

    using Expected = std::ranges::subrange<std::counted_iterator<decltype(it)>, std::default_sentinel_t>;
    using ConstExpected = std::ranges::subrange<std::counted_iterator<decltype(cit)>, std::default_sentinel_t>;

    ASSERT_SAME_TYPE(decltype(c1), Expected);
    ASSERT_SAME_TYPE(decltype(c2), Expected);
    ASSERT_SAME_TYPE(decltype(c3), Expected);
    ASSERT_SAME_TYPE(decltype(c4), Expected);
    ASSERT_SAME_TYPE(decltype(c5), ConstExpected);
    ASSERT_SAME_TYPE(decltype(c6), ConstExpected);
    ASSERT_SAME_TYPE(decltype(c7), ConstExpected);
    ASSERT_SAME_TYPE(decltype(c8), ConstExpected);

    assert(c1.begin().base() == it && c1.size() == 3);
    assert(c2.begin().base() == it && c2.size() == 3);
    assert(c3.begin().base() == it && c3.size() == 3);
    assert(c4.begin().base() == it && c4.size() == 3);
    assert(c5.begin().base() == cit && c5.size() == 3);
    assert(c6.begin().base() == cit && c6.size() == 3);
    assert(c7.begin().base() == cit && c7.size() == 3);
    assert(c8.begin().base() == cit && c8.size() == 3);
  }

  {
    auto it = cpp17_output_iterator<int*>(buffer);

    auto c1 = std::views::counted(it, 3);
    auto c2 = std::views::counted(std::as_const(it), 3);
    auto c3 = std::views::counted(std::move(it), 3);
    auto c4 = std::views::counted(cpp17_output_iterator<int*>(buffer), 3);

    using Expected = std::ranges::subrange<std::counted_iterator<decltype(it)>, std::default_sentinel_t>;

    ASSERT_SAME_TYPE(decltype(c1), Expected);
    ASSERT_SAME_TYPE(decltype(c2), Expected);
    ASSERT_SAME_TYPE(decltype(c3), Expected);
    ASSERT_SAME_TYPE(decltype(c4), Expected);

    assert(base(c1.begin().base()) == buffer && c1.size() == 3);
    assert(base(c2.begin().base()) == buffer && c2.size() == 3);
    assert(base(c3.begin().base()) == buffer && c3.size() == 3);
    assert(base(c4.begin().base()) == buffer && c4.size() == 3);
  }

  {
    auto it = cpp17_input_iterator<int*>(buffer);

    auto c1 = std::views::counted(it, 3);
    auto c2 = std::views::counted(std::as_const(it), 3);
    auto c3 = std::views::counted(std::move(it), 3);
    auto c4 = std::views::counted(cpp17_input_iterator<int*>(buffer), 3);

    using Expected = std::ranges::subrange<std::counted_iterator<decltype(it)>, std::default_sentinel_t>;

    ASSERT_SAME_TYPE(decltype(c1), Expected);
    ASSERT_SAME_TYPE(decltype(c2), Expected);
    ASSERT_SAME_TYPE(decltype(c3), Expected);
    ASSERT_SAME_TYPE(decltype(c4), Expected);

    assert(base(c1.begin().base()) == buffer && c1.size() == 3);
    assert(base(c2.begin().base()) == buffer && c2.size() == 3);
    assert(base(c3.begin().base()) == buffer && c3.size() == 3);
    assert(base(c4.begin().base()) == buffer && c4.size() == 3);
  }

  {
    auto it = cpp20_input_iterator<int*>(buffer);

    static_assert(!std::copyable<cpp20_input_iterator<int*>>);
    static_assert(!CountedInvocable<cpp20_input_iterator<int*>&, int>);
    static_assert(!CountedInvocable<const cpp20_input_iterator<int*>&, int>);
    auto c3 = std::views::counted(std::move(it), 3);
    auto c4 = std::views::counted(cpp20_input_iterator<int*>(buffer), 3);

    using Expected = std::ranges::subrange<std::counted_iterator<decltype(it)>, std::default_sentinel_t>;

    ASSERT_SAME_TYPE(decltype(c3), Expected);
    ASSERT_SAME_TYPE(decltype(c4), Expected);

    assert(base(c3.begin().base()) == buffer && c3.size() == 3);
    assert(base(c4.begin().base()) == buffer && c4.size() == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
