//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// class std::ranges::subrange;

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

template<size_t I, class S>
concept HasGet = requires {
  std::get<I>(std::declval<S>());
};

static_assert( HasGet<0, std::ranges::subrange<int*>>);
static_assert( HasGet<1, std::ranges::subrange<int*>>);
static_assert(!HasGet<2, std::ranges::subrange<int*>>);
static_assert(!HasGet<3, std::ranges::subrange<int*>>);

constexpr bool test() {
  {
    using It = int*;
    using Sent = sentinel_wrapper<int*>;
    int a[] = {1, 2, 3};
    using R = std::ranges::subrange<It, Sent, std::ranges::subrange_kind::unsized>;
    R r = R(It(a), Sent(It(a + 3)));
    ASSERT_SAME_TYPE(decltype(std::get<0>(r)), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(r)), Sent);
    ASSERT_SAME_TYPE(decltype(std::get<0>(static_cast<R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<R&&>(r))), Sent);
    ASSERT_SAME_TYPE(decltype(std::get<0>(static_cast<const R&>(r))), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<const R&>(r))), Sent);
    ASSERT_SAME_TYPE(decltype(std::get<0>(static_cast<const R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<const R&&>(r))), Sent);
    assert(base(std::get<0>(r)) == a);                      // copy from It
    assert(base(base(std::get<1>(r))) == a + 3);            // copy from Sent
    assert(base(std::get<0>(std::move(r))) == a);           // copy from It
    assert(base(base(std::get<1>(std::move(r)))) == a + 3); // copy from Sent
  }
  {
    using It = int*;
    using Sent = sentinel_wrapper<int*>;
    int a[] = {1, 2, 3};
    using R = std::ranges::subrange<It, Sent, std::ranges::subrange_kind::sized>;
    R r = R(It(a), Sent(It(a + 3)), 3);
    ASSERT_SAME_TYPE(decltype(std::get<0>(r)), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(r)), Sent);
    ASSERT_SAME_TYPE(decltype(std::get<0>(static_cast<R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<R&&>(r))), Sent);
    ASSERT_SAME_TYPE(decltype(std::get<0>(static_cast<const R&>(r))), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<const R&>(r))), Sent);
    ASSERT_SAME_TYPE(decltype(std::get<0>(static_cast<const R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<const R&&>(r))), Sent);
    assert(base(std::get<0>(r)) == a);                      // copy from It
    assert(base(base(std::get<1>(r))) == a + 3);            // copy from Sent
    assert(base(std::get<0>(std::move(r))) == a);           // copy from It
    assert(base(base(std::get<1>(std::move(r)))) == a + 3); // copy from Sent
  }
  {
    // Test the fix for LWG 3589.
    using It = cpp20_input_iterator<int*>;
    using Sent = sentinel_wrapper<It>;
    int a[] = {1, 2, 3};
    using R = std::ranges::subrange<It, Sent>;
    R r = R(It(a), Sent(It(a + 3)));
    static_assert(!HasGet<0, R&>);
    ASSERT_SAME_TYPE(decltype(std::get<1>(r)), Sent);
    ASSERT_SAME_TYPE(decltype(std::get<0>(static_cast<R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<R&&>(r))), Sent);
    static_assert(!HasGet<0, const R&>);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<const R&>(r))), Sent);
    static_assert(!HasGet<0, const R&&>);
    ASSERT_SAME_TYPE(decltype(std::get<1>(static_cast<const R&&>(r))), Sent);
    assert(base(base(std::get<1>(r))) == a + 3);            // copy from Sent
    assert(base(std::get<0>(std::move(r))) == a);           // move from It
    assert(base(base(std::get<1>(std::move(r)))) == a + 3); // copy from Sent
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
