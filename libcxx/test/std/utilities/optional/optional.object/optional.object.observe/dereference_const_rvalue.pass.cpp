//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr T&& optional<T>::operator*() const &&;

#include <cassert>
#include <memory>
#include <optional>
#include <utility>

#include "test_macros.h"
#if TEST_STD_VER >= 26
#  include "copy_move_types.h"
#endif

using std::optional;

struct X
{
    constexpr int test() const& {return 3;}
    int test() & {return 4;}
    constexpr int test() const&& {return 5;}
    int test() && {return 6;}
};

struct Y
{
    int test() const && {return 2;}
};

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  { // const&&
    TracedCopyMove x{};
    const std::optional<TracedCopyMove&> opt(x);
    ASSERT_NOEXCEPT(*std::move(opt));
    ASSERT_SAME_TYPE(decltype(*std::move(opt)), TracedCopyMove&);

    assert(std::addressof(*(std::move(opt))) == std::addressof(x));
    assert((*std::move(opt)).constMove == 0);
    assert((*std::move(opt)).nonConstMove == 0);
    assert((*std::move(opt)).constCopy == 0);
    assert((*std::move(opt)).nonConstCopy == 0);
  }

  {
    TracedCopyMove x{};
    const std::optional<const TracedCopyMove&> opt(x);
    ASSERT_NOEXCEPT(*std::move(opt));
    ASSERT_SAME_TYPE(decltype(*std::move(opt)), const TracedCopyMove&);

    assert(std::addressof(*(std::move(opt))) == std::addressof(x));
    assert((*std::move(opt)).constMove == 0);
    assert((*std::move(opt)).nonConstMove == 0);
    assert((*std::move(opt)).constCopy == 0);
    assert((*std::move(opt)).nonConstCopy == 0);
  }

  return true;
}
#endif

int main(int, char**)
{
    {
        const optional<X> opt; ((void)opt);
        ASSERT_SAME_TYPE(decltype(*std::move(opt)), X const &&);
        ASSERT_NOEXCEPT(*std::move(opt));
    }
    {
        constexpr optional<X> opt(X{});
        static_assert((*std::move(opt)).test() == 5, "");
    }
    {
        constexpr optional<Y> opt(Y{});
        assert((*std::move(opt)).test() == 2);
    }

#if TEST_STD_VER >= 26
    assert(test_ref());
    static_assert(test_ref());
#endif

    return 0;
}
