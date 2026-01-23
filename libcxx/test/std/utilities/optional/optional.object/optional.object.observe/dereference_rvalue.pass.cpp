//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr T&& optional<T>::operator*() &&;

#include <cassert>
#include <optional>

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
    constexpr int test() && {return 7;}
};

constexpr int
test()
{
    optional<Y> opt{Y{}};
    return (*std::move(opt)).test();
}

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  // ensure underlying value isn't moved from
  {
    TracedCopyMove x{};
    std::optional<TracedCopyMove&> opt(x);
    ASSERT_NOEXCEPT(*std::move(opt));
    ASSERT_SAME_TYPE(decltype(*std::move(opt)), TracedCopyMove&);

    assert(std::addressof(*std::move(opt)) == std::addressof(x));
    assert((*std::move(opt)).constMove == 0);
    assert((*std::move(opt)).nonConstMove == 0);
    assert((*std::move(opt)).constCopy == 0);
    assert((*std::move(opt)).nonConstCopy == 0);
  }

  {
    TracedCopyMove x{};
    std::optional<const TracedCopyMove&> opt(x);
    ASSERT_NOEXCEPT(*std::move(opt));
    ASSERT_SAME_TYPE(decltype(*std::move(opt)), const TracedCopyMove&);

    assert(std::addressof(*std::move(opt)) == std::addressof(x));
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
        optional<X> opt; ((void)opt);
        ASSERT_SAME_TYPE(decltype(*std::move(opt)), X&&);
        ASSERT_NOEXCEPT(*std::move(opt));
    }
    {
        optional<X> opt(X{});
        assert((*std::move(opt)).test() == 6);
    }
    static_assert(test() == 7, "");

#if TEST_STD_VER >= 26
    assert(test_ref());
    static_assert(test_ref());
#endif

    return 0;
}
