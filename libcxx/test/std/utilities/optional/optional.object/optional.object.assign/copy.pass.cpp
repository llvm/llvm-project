//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr optional<T>& operator=(const optional<T>& rhs);

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "archetypes.h"

using std::optional;

struct X
{
    static bool throw_now;

    X() = default;
    X(const X&)
    {
        if (throw_now)
            TEST_THROW(6);
    }
    X& operator=(X const&) = default;
};

bool X::throw_now = false;

template <class Tp>
constexpr bool assign_empty(optional<Tp>&& lhs) {
    const optional<Tp> rhs;
    lhs = rhs;
    return !lhs.has_value() && !rhs.has_value();
}

template <class Tp>
constexpr bool assign_value(optional<Tp>&& lhs) {
    const optional<Tp> rhs(101);
    lhs = rhs;
    return lhs.has_value() && rhs.has_value() && *lhs == *rhs;
}

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  struct TraceCopyAssign {
    int copyAssign              = 0;
    mutable int constCopyAssign = 0;

    constexpr TraceCopyAssign() = default;
    constexpr TraceCopyAssign(TraceCopyAssign& r) : copyAssign(r.copyAssign + 1), constCopyAssign(r.constCopyAssign) {}
    constexpr TraceCopyAssign(const TraceCopyAssign& r)
        : copyAssign(r.copyAssign), constCopyAssign(r.constCopyAssign + 1) {}
    constexpr TraceCopyAssign& operator=(const TraceCopyAssign&) {
      copyAssign++;
      return *this;
    }
    constexpr const TraceCopyAssign& operator=(const TraceCopyAssign&) const {
      constCopyAssign++;
      return *this;
    }
  };
  using T = TraceCopyAssign;
  {
    T t{};
    std::optional<T&> o1{t};
    std::optional<T&> o2 = o1;

    assert(&(*o2) == &t);
    assert(&(*o2) == &(*o1));
    assert(t.constCopyAssign == 0);
    assert(t.copyAssign == 0);
    assert(o1.has_value());
    assert(o2.has_value());
  }
  {
    T t{};
    std::optional<T&> o1{t};
    std::optional<T> o2 = o1;

    assert(&(*o2) != &t);
    assert(&(*o2) != &(*o1));
    assert(o2->constCopyAssign == 0);
    assert(o2->copyAssign == 1);
    assert(o1.has_value());
    assert(o2.has_value());
  }

  return true;
}
#endif

int main(int, char**)
{
    {
        using O = optional<int>;
        static_assert(assign_empty(O{42}));
        static_assert(assign_value(O{42}));
        assert(assign_empty(O{42}));
        assert(assign_value(O{42}));
    }
    {
        using O = optional<TrivialTestTypes::TestType>;
        static_assert(assign_empty(O{42}));
        static_assert(assign_value(O{42}));
        assert(assign_empty(O{42}));
        assert(assign_value(O{42}));
    }
    {
        using O = optional<TestTypes::TestType>;
        assert(assign_empty(O{42}));
        assert(assign_value(O{42}));
    }
    {
        using T = TestTypes::TestType;
        T::reset();
        optional<T> opt(3);
        const optional<T> opt2;
        assert(T::alive == 1);
        opt = opt2;
        assert(T::alive == 0);
        assert(!opt2.has_value());
        assert(!opt.has_value());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        optional<X> opt;
        optional<X> opt2(X{});
        assert(static_cast<bool>(opt2) == true);
        try
        {
            X::throw_now = true;
            opt = opt2;
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
            assert(static_cast<bool>(opt) == false);
        }
    }
#endif

#if TEST_STD_VER >= 26
    assert(test_ref());
    static_assert(test_ref());
#endif

    return 0;
}
