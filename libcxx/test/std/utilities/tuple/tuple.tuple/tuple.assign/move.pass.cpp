//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// tuple& operator=(tuple&& u);

// UNSUPPORTED: c++03

#include <memory>
#include <tuple>
#include <utility>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
};
static_assert(std::is_copy_assignable<CopyAssignable>::value, "");
struct MoveAssignable {
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


int main(int, char**)
{
    {
        typedef std::tuple<> T;
        T t0;
        T t;
        t = std::move(t0);
    }
    {
        typedef std::tuple<MoveOnly> T;
        T t0(MoveOnly(0));
        T t;
        t = std::move(t0);
        assert(std::get<0>(t) == 0);
    }
    {
        typedef std::tuple<MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1));
        T t;
        t = std::move(t0);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
    }
    {
        typedef std::tuple<MoveOnly, MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
        T t;
        t = std::move(t0);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
        assert(std::get<2>(t) == 2);
    }
    {
        // test reference assignment.
        using T = std::tuple<int&, int&&>;
        int x = 42;
        int y = 100;
        int x2 = -1;
        int y2 = 500;
        T t(x, std::move(y));
        T t2(x2, std::move(y2));
        t = std::move(t2);
        assert(std::get<0>(t) == x2);
        assert(&std::get<0>(t) == &x);
        assert(std::get<1>(t) == y2);
        assert(&std::get<1>(t) == &y);
    }
    {
        // test that the implicitly generated move assignment operator
        // is properly deleted
        using T = std::tuple<std::unique_ptr<int>>;
        static_assert(std::is_move_assignable<T>::value, "");
        static_assert(!std::is_copy_assignable<T>::value, "");

    }
    {
        using T = std::tuple<int, NonAssignable>;
        static_assert(!std::is_move_assignable<T>::value, "");
    }
    {
        using T = std::tuple<int, MoveAssignable>;
        static_assert(std::is_move_assignable<T>::value, "");
    }
    {
        // The move should decay to a copy.
        CountAssign::reset();
        using T = std::tuple<CountAssign, CopyAssignable>;
        static_assert(std::is_move_assignable<T>::value, "");
        T t1;
        T t2;
        t1 = std::move(t2);
        assert(CountAssign::copied == 1);
        assert(CountAssign::moved == 0);
    }

  return 0;
}
