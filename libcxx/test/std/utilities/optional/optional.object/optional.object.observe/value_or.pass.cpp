//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// template <class U> constexpr T optional<T>::value_or(U&& v) &&;

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;
using std::in_place_t;
using std::in_place;

struct Y
{
    int i_;

    constexpr Y(int i) : i_(i) {}
};

struct X
{
    int i_;

    constexpr X(int i) : i_(i) {}
    constexpr X(X&& x) : i_(x.i_) {x.i_ = 0;}
    constexpr X(const Y& y) : i_(y.i_) {}
    constexpr X(Y&& y) : i_(y.i_+1) {}
    friend constexpr bool operator==(const X& x, const X& y)
        {return x.i_ == y.i_;}
};

struct Z {
  int i_, j_;
  constexpr Z(int i, int j) : i_(i), j_(j) {}
  friend constexpr bool operator==(const Z& z1, const Z& z2) { return z1.i_ == z2.i_ && z1.j_ == z2.j_; }
};

constexpr int test()
{
    {
        optional<X> opt(in_place, 2);
        Y y(3);
        assert(std::move(opt).value_or(y) == 2);
        assert(*opt == 0);
    }
    {
        optional<X> opt(in_place, 2);
        assert(std::move(opt).value_or(Y(3)) == 2);
        assert(*opt == 0);
    }
    {
        optional<X> opt;
        Y y(3);
        assert(std::move(opt).value_or(y) == 3);
        assert(!opt);
    }
    {
        optional<X> opt;
        assert(std::move(opt).value_or(Y(3)) == 4);
        assert(!opt);
    }
    {
      optional<X> opt;
      assert(std::move(opt).value_or({Y(3)}) == 4);
      assert(!opt);
    }
    {
      optional<Z> opt;
      assert((std::move(opt).value_or({2, 3}) == Z{2, 3}));
      assert(!opt);
    }
#if TEST_STD_VER >= 26
    {
      int y = 2;
      optional<int&> opt;
      ASSERT_SAME_TYPE(decltype(std::move(opt).value_or(y)), int);
      assert(std::move(opt).value_or(y) == 2);
      assert(!opt);
    }
#endif
    return 0;
}

int main(int, char**)
{
  assert(test() == 0);
  static_assert(test() == 0);

  return 0;
}
