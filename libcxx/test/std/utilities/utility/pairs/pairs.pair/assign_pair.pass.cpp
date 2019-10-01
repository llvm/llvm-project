//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include <utility>
#include <memory>
#include <cassert>

#include "test_macros.h"


struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  CopyAssignable() = default;
  CopyAssignable(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
};
struct MoveAssignable {
  MoveAssignable() = default;
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};

struct CountAssign {
  static int copied;
  static int moved;
  static void reset() { copied = moved = 0; }
  CountAssign() = default;
  CountAssign& operator=(CountAssign const&) { ++copied; return *this; }
  CountAssign& operator=(CountAssign&&) { ++moved; return *this; }
};
int CountAssign::copied = 0;
int CountAssign::moved = 0;

struct Incomplete;
extern Incomplete inc_obj;

int main(int, char**)
{
    {
        typedef std::pair<CopyAssignable, short> P;
        const P p1(CopyAssignable(), 4);
        P p2;
        p2 = p1;
        assert(p2.second == 4);
    }
    {
        using P = std::pair<int&, int&&>;
        int x = 42;
        int y = 101;
        int x2 = -1;
        int y2 = 300;
        P p1(x, std::move(y));
        P p2(x2, std::move(y2));
        p1 = p2;
        assert(p1.first == x2);
        assert(p1.second == y2);
    }
    {
        using P = std::pair<int, NonAssignable>;
        static_assert(!std::is_copy_assignable<P>::value, "");
    }
    {
        CountAssign::reset();
        using P = std::pair<CountAssign, CopyAssignable>;
        static_assert(std::is_copy_assignable<P>::value, "");
        P p;
        P p2;
        p = p2;
        assert(CountAssign::copied == 1);
        assert(CountAssign::moved == 0);
    }
    {
        using P = std::pair<int, MoveAssignable>;
        static_assert(!std::is_copy_assignable<P>::value, "");
    }
    {
        using P = std::pair<int, Incomplete&>;
        static_assert(!std::is_copy_assignable<P>::value, "");
        P p(42, inc_obj);
        assert(&p.second == &inc_obj);
    }

  return 0;
}

struct Incomplete {};
Incomplete inc_obj;
