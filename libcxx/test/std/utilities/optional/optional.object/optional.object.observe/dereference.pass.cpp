//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr T& optional<T>::operator*() &;

#include <cassert>
#include <memory>
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
    constexpr int test() {return 7;}
};

constexpr int
test()
{
    optional<Y> opt{Y{}};
    return (*opt).test();
}

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  {
    TracedCopyMove x{};
    std::optional<TracedCopyMove&> opt(x);
    static_assert(noexcept(*opt));
    ASSERT_SAME_TYPE(decltype(*opt), TracedCopyMove&);

    assert(std::addressof(*opt) == std::addressof(x));
    assert(x.constMove == 0);
    assert(x.nonConstMove == 0);
    assert(x.constCopy == 0);
    assert(x.nonConstCopy == 0);
  }

  {
    TracedCopyMove x{};
    std::optional<const TracedCopyMove&> opt(x);
    static_assert(noexcept(*opt));
    ASSERT_SAME_TYPE(decltype(*opt), const TracedCopyMove&);

    assert(std::addressof(*opt) == std::addressof(x));
    assert(x.constMove == 0);
    assert(x.nonConstMove == 0);
    assert(x.constCopy == 0);
    assert(x.nonConstCopy == 0);
  }

  return true;
}
#endif

int main(int, char**)
{
    {
        optional<X> opt; ((void)opt);
        ASSERT_SAME_TYPE(decltype(*opt), X&);
        ASSERT_NOEXCEPT(*opt);
    }
    {
        optional<X> opt(X{});
        assert((*opt).test() == 4);
    }
#if TEST_STD_VER >= 26
    {
      X x{};
      optional<X&> opt(x);
      ASSERT_SAME_TYPE(decltype(*opt), X&);
      ASSERT_NOEXCEPT(*opt);
    }
    {
      X x{};
      optional<X&> opt(x);
      assert((*opt).test() == 4);
    }
#endif
    static_assert(test() == 7, "");
    return 0;
}
