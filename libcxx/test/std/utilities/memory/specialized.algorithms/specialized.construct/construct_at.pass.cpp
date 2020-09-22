//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <class T, class ...Args>
// constexpr T* construct_at(T* location, Args&& ...args);

#include <memory>
#include <cassert>


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
        int ints[1] = {0};
        int* res = std::construct_at(&ints[0], 42);
        assert(res == &ints[0]);
        assert(*res == 42);
    }

    {
        Foo foos[1] = {};
        int count = 0;
        Foo* res = std::construct_at(&foos[0], 42, 'x', 123.89, &count);
        assert(res == &foos[0]);
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

// Make sure std::construct_at SFINAEs out based on the validity of calling
// the constructor, instead of hard-erroring.
template <typename T, typename = decltype(
    std::construct_at((T*)nullptr, 1, 2) // missing arguments for Foo(...)
)>
constexpr bool test_sfinae(int) { return false; }
template <typename T>
constexpr bool test_sfinae(...) { return true; }
static_assert(test_sfinae<Foo>(int()));

int main(int, char**)
{
    test();
    static_assert(test());
    return 0;
}
