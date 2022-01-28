//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Investigation needed
// XFAIL: gcc

// <memory>

// template <class T, class ...Args>
// constexpr T* construct_at(T* location, Args&& ...args);

#include <memory>
#include <cassert>
#include <utility>

#include "test_iterators.h"

struct Foo {
    int a;
    char b;
    double c;
    constexpr Foo() { }
    constexpr Foo(int a, char b, double c) : a(a), b(b), c(c) { }
    constexpr Foo(int a, char b, double c, int* count) : Foo(a, b, c) { *count += 1; }
    constexpr bool operator==(Foo const& other) const {
        return a == other.a && b == other.b && c == other.c;
    }
};

struct Counted {
    int& count_;
    constexpr Counted(int& count) : count_(count) { ++count; }
    constexpr Counted(Counted const& that) : count_(that.count_) { ++count_; }
    constexpr ~Counted() { --count_; }
};

constexpr bool test()
{
    {
        int i = 99;
        int* res = std::construct_at(&i);
        assert(res == &i);
        assert(*res == 0);
    }

    {
        int i = 0;
        int* res = std::construct_at(&i, 42);
        assert(res == &i);
        assert(*res == 42);
    }

    {
        Foo foo = {};
        int count = 0;
        Foo* res = std::construct_at(&foo, 42, 'x', 123.89, &count);
        assert(res == &foo);
        assert(*res == Foo(42, 'x', 123.89));
        assert(count == 1);
    }

    {
        std::allocator<Counted> a;
        Counted* p = a.allocate(2);
        int count = 0;
        std::construct_at(p, count);
        assert(count == 1);
        std::construct_at(p+1, count);
        assert(count == 2);
        (p+1)->~Counted();
        assert(count == 1);
        p->~Counted();
        assert(count == 0);
        a.deallocate(p, 2);
    }

    {
        std::allocator<Counted const> a;
        Counted const* p = a.allocate(2);
        int count = 0;
        std::construct_at(p, count);
        assert(count == 1);
        std::construct_at(p+1, count);
        assert(count == 2);
        (p+1)->~Counted();
        assert(count == 1);
        p->~Counted();
        assert(count == 0);
        a.deallocate(p, 2);
    }

    return true;
}

template <class ...Args, class = decltype(std::construct_at(std::declval<Args>()...))>
constexpr bool can_construct_at(Args&&...) { return true; }

template <class ...Args>
constexpr bool can_construct_at(...) { return false; }

// Check that SFINAE works.
static_assert( can_construct_at((int*)nullptr, 42));
static_assert( can_construct_at((Foo*)nullptr, 1, '2', 3.0));
static_assert(!can_construct_at((Foo*)nullptr, 1, '2'));
static_assert(!can_construct_at((Foo*)nullptr, 1, '2', 3.0, 4));
static_assert(!can_construct_at(nullptr, 1, '2', 3.0));
static_assert(!can_construct_at((int*)nullptr, 1, '2', 3.0));
static_assert(!can_construct_at(contiguous_iterator<Foo*>(), 1, '2', 3.0));
// Can't construct function pointers.
static_assert(!can_construct_at((int(*)())nullptr));
static_assert(!can_construct_at((int(*)())nullptr, nullptr));

int main(int, char**)
{
    test();
    static_assert(test());
    return 0;
}
