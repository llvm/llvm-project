//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr const T& optional<T>::operator*() const &;

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
    int test() const {return 2;}
};

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  using T = TracedCopyMove;
  {
    T x{};
    const std::optional<T&> opt(x);
    ASSERT_NOEXCEPT(*opt);
    ASSERT_SAME_TYPE(decltype(*opt), TracedCopyMove&);

    assert(std::addressof(*opt) == std::addressof(x));
    assert((*opt).constMove == 0);
    assert((*opt).nonConstMove == 0);
    assert((*opt).constCopy == 0);
    assert((*opt).nonConstCopy == 0);
  }

  {
    T x{};
    const std::optional<const T&> opt(x);
    ASSERT_NOEXCEPT(*opt);
    ASSERT_SAME_TYPE(decltype(*opt), const TracedCopyMove&);

    assert(std::addressof(*opt) == std::addressof(x));
    assert((*opt).constMove == 0);
    assert((*opt).nonConstMove == 0);
    assert((*opt).constCopy == 0);
    assert((*opt).nonConstCopy == 0);
  }

  return true;
}
#endif

int main(int, char**)
{
    {
        const optional<X> opt; ((void)opt);
        ASSERT_SAME_TYPE(decltype(*opt), X const&);
        ASSERT_NOEXCEPT(*opt);
    }
    {
        constexpr optional<X> opt(X{});
        static_assert((*opt).test() == 3, "");
    }
    {
      constexpr optional<Y> opt(Y{});
      assert((*opt).test() == 2);
    }
#if TEST_STD_VER >= 26
    {
      assert(test_ref());
      static_assert(test_ref());
    }
#endif
    return 0;
}
