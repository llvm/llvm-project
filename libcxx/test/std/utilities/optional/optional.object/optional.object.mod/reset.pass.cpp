//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// void reset() noexcept;

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;

struct X
{
    static bool dtor_called;
    X() = default;
    X(const X&) = default;
    X& operator=(const X&) = default;
    ~X() {dtor_called = true;}
};

bool X::dtor_called = false;

TEST_CONSTEXPR_CXX20 bool check_reset() {
  {
    optional<int> opt;
    static_assert(noexcept(opt.reset()) == true, "");
    opt.reset();
    assert(static_cast<bool>(opt) == false);
  }
  {
    optional<int> opt(3);
    opt.reset();
    assert(static_cast<bool>(opt) == false);
  }
#if TEST_STD_VER >= 20
  {
    // https://llvm.org/PR192852
    // Verify that a disengaged optional<T> can also be constexpr, where T is not trivially destructible.

    struct NonTriviallyDestructible {
      constexpr NonTriviallyDestructible() {}
      constexpr ~NonTriviallyDestructible() {}
    };

    struct Derived : optional<NonTriviallyDestructible> {
      using Base = optional<NonTriviallyDestructible>;

      constexpr Derived() : Base(std::in_place) { Base::reset(); }
    };

    [[maybe_unused]] constexpr Derived d;
  }
#endif
  return true;
}

int main(int, char**)
{
    check_reset();
#if TEST_STD_VER >= 20
    static_assert(check_reset());
#endif
    {
        optional<X> opt;
        static_assert(noexcept(opt.reset()) == true, "");
        assert(X::dtor_called == false);
        opt.reset();
        assert(X::dtor_called == false);
        assert(static_cast<bool>(opt) == false);
    }
    {
        optional<X> opt(X{});
        X::dtor_called = false;
        opt.reset();
        assert(X::dtor_called == true);
        assert(static_cast<bool>(opt) == false);
        X::dtor_called = false;
    }

#if TEST_STD_VER >= 26
    {
      X x{};
      optional<X&> opt(x);
      X::dtor_called = false;
      opt.reset();
      ASSERT_NOEXCEPT(opt.reset());
      assert(X::dtor_called == false);
      assert(static_cast<bool>(opt) == false);
    }
#endif

    return 0;
}
