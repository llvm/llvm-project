//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   tuple& operator=(tuple<UTypes...>&& u);

// UNSUPPORTED: c++03

#include <tuple>
#include <string>
#include <memory>
#include <utility>
#include <cassert>

#include "test_macros.h"

struct B
{
    int id_;

    explicit B(int i= 0) : id_(i) {}

    virtual ~B() {}
};

struct D
    : B
{
    explicit D(int i) : B(i) {}
};

struct E {
  E() = default;
  E& operator=(int) {
      return *this;
  }
};

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};

struct NothrowMoveAssignable
{
    NothrowMoveAssignable& operator=(NothrowMoveAssignable&&) noexcept { return *this; }
};

struct PotentiallyThrowingMoveAssignable
{
    PotentiallyThrowingMoveAssignable& operator=(PotentiallyThrowingMoveAssignable&&) { return *this; }
};

int main(int, char**)
{
    {
        typedef std::tuple<long> T0;
        typedef std::tuple<long long> T1;
        T0 t0(2);
        T1 t1;
        t1 = std::move(t0);
        assert(std::get<0>(t1) == 2);
    }
    {
        typedef std::tuple<long, char> T0;
        typedef std::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1;
        t1 = std::move(t0);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
    }
    {
        typedef std::tuple<long, char, D> T0;
        typedef std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1;
        t1 = std::move(t0);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1).id_ == 3);
    }
    {
        D d(3);
        D d2(2);
        typedef std::tuple<long, char, D&> T0;
        typedef std::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d2);
        T1 t1(1, 'b', d);
        t1 = std::move(t0);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1).id_ == 2);
    }
    {
        typedef std::tuple<long, char, std::unique_ptr<D>> T0;
        typedef std::tuple<long long, int, std::unique_ptr<B>> T1;
        T0 t0(2, 'a', std::unique_ptr<D>(new D(3)));
        T1 t1;
        t1 = std::move(t0);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1)->id_ == 3);
    }
    {
        // Test that tuple evaluates correctly applies an lvalue reference
        // before evaluating is_assignable (i.e. 'is_assignable<int&, int&&>')
        // instead of evaluating 'is_assignable<int&&, int&&>' which is false.
        int x = 42;
        int y = 43;
        std::tuple<int&&, E> t(std::move(x), E{});
        std::tuple<int&&, int> t2(std::move(y), 44);
        t = std::move(t2);
        assert(std::get<0>(t) == 43);
        assert(&std::get<0>(t) == &x);
    }
    {
      using T = std::tuple<int, NonAssignable>;
      using U = std::tuple<NonAssignable, int>;
      static_assert(!std::is_assignable<T, U&&>::value, "");
      static_assert(!std::is_assignable<U, T&&>::value, "");
    }
    {
        typedef std::tuple<NothrowMoveAssignable, long> T0;
        typedef std::tuple<NothrowMoveAssignable, int> T1;
        static_assert(std::is_nothrow_assignable<T0&, T1&&>::value, "");
    }
    {
        typedef std::tuple<PotentiallyThrowingMoveAssignable, long> T0;
        typedef std::tuple<PotentiallyThrowingMoveAssignable, int> T1;
        static_assert(!std::is_nothrow_assignable<T0&, T1&&>::value, "");
    }

    return 0;
}
