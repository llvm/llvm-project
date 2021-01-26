//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <functional>

#include <functional>
#include <cassert>

#include "test_macros.h"

// Prevent warning on the `const NonCopyable()` function type.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif

struct NonCopyable {
    NonCopyable() = default;
    NonCopyable(NonCopyable&&) = delete;
    friend bool operator==(NonCopyable, NonCopyable) { return true; }
};

struct LargeLambda {
    int a[100];
    NonCopyable operator()() const { return NonCopyable(); }
    NonCopyable operator()(int) const { return NonCopyable(); }
    NonCopyable f() const { return NonCopyable(); }
};

void test()
{
    std::function<NonCopyable()> f1a = []() { return NonCopyable(); };
    std::function<NonCopyable()> f2a = +[]() { return NonCopyable(); };
    std::function<NonCopyable()> f3a = LargeLambda();
    std::function<NonCopyable()> f4a = std::ref(f1a);
    std::function<NonCopyable(int)> f1b = [](int) { return NonCopyable(); };
    std::function<NonCopyable(int)> f2b = +[](int) { return NonCopyable(); };
    std::function<NonCopyable(int)> f3b = LargeLambda();
    std::function<NonCopyable(int)> f4b = std::ref(f1b);

    assert(f1a() == f2a());
    assert(f3a() == f4a());
    assert(f1b(1) == f2b(1));
    assert(f3b(1) == f4b(1));
}

void const_test()
{
    std::function<const NonCopyable()> f1a = []() { return NonCopyable(); };
    std::function<const NonCopyable()> f2a = +[]() { return NonCopyable(); };
    std::function<const NonCopyable()> f3a = LargeLambda();
    std::function<const NonCopyable()> f4a = std::ref(f1a);
    std::function<const NonCopyable(int)> f1b = [](int) { return NonCopyable(); };
    std::function<const NonCopyable(int)> f2b = +[](int) { return NonCopyable(); };
    std::function<const NonCopyable(int)> f3b = LargeLambda();
    std::function<const NonCopyable(int)> f4b = std::ref(f1b);

    assert(f1a() == f2a());
    assert(f3a() == f4a());
    assert(f1b(1) == f2b(1));
    assert(f3b(1) == f4b(1));
}

void void_test()
{
    std::function<void()> f1a = []() { return NonCopyable(); };
    std::function<void()> f2a = +[]() { return NonCopyable(); };
    std::function<void()> f3a = LargeLambda();
    std::function<void()> f4a = std::ref(f1a);
    std::function<void(int)> f1b = [](int) { return NonCopyable(); };
    std::function<void(int)> f2b = +[](int) { return NonCopyable(); };
    std::function<void(int)> f3b = LargeLambda();
    std::function<void(int)> f4b = std::ref(f1b);
}

void const_void_test()
{
    std::function<const void()> f1a = []() { return NonCopyable(); };
    std::function<const void()> f2a = +[]() { return NonCopyable(); };
    std::function<const void()> f3a = LargeLambda();
    std::function<const void()> f4a = std::ref(f1a);
    std::function<const void(int)> f1b = [](int) { return NonCopyable(); };
    std::function<const void(int)> f2b = +[](int) { return NonCopyable(); };
    std::function<const void(int)> f3b = LargeLambda();
    std::function<const void(int)> f4b = std::ref(f1b);
}

void member_pointer_test()
{
    std::function<NonCopyable(LargeLambda*)> f1a = &LargeLambda::f;
    std::function<NonCopyable(LargeLambda&)> f2a = &LargeLambda::f;
    LargeLambda ll;
    assert(f1a(&ll) == f2a(ll));

    static_assert(std::is_convertible_v<NonCopyable (LargeLambda::*)(), std::function<NonCopyable(LargeLambda*)>>);
    static_assert(std::is_convertible_v<NonCopyable (LargeLambda::*)(), std::function<NonCopyable(LargeLambda&)>>);
    static_assert(std::is_convertible_v<NonCopyable (LargeLambda::*)() const, std::function<NonCopyable(LargeLambda*)>>);
    static_assert(std::is_convertible_v<NonCopyable (LargeLambda::*)() const, std::function<NonCopyable(LargeLambda&)>>);
    static_assert(std::is_convertible_v<NonCopyable (LargeLambda::*)() const, std::function<NonCopyable(const LargeLambda*)>>);
    static_assert(std::is_convertible_v<NonCopyable (LargeLambda::*)() const, std::function<NonCopyable(const LargeLambda&)>>);

    // Verify we have SFINAE against invoking a pointer-to-data-member in a way that would have to copy the NonCopyable.
    static_assert(!std::is_convertible_v<NonCopyable LargeLambda::*, std::function<NonCopyable(LargeLambda*)>>);
    static_assert(!std::is_convertible_v<NonCopyable LargeLambda::*, std::function<NonCopyable(LargeLambda&)>>);
    static_assert(!std::is_convertible_v<NonCopyable LargeLambda::*, std::function<NonCopyable&(const LargeLambda&)>>);
    static_assert(std::is_convertible_v<NonCopyable LargeLambda::*, std::function<NonCopyable&(LargeLambda*)>>);
    static_assert(std::is_convertible_v<NonCopyable LargeLambda::*, std::function<NonCopyable&(LargeLambda&)>>);
    static_assert(std::is_convertible_v<NonCopyable LargeLambda::*, std::function<const NonCopyable&(const LargeLambda&)>>);
}

void ctad_test()
{
    std::function f1a = []() { return NonCopyable(); };
    std::function f2a = +[]() { return NonCopyable(); };
    std::function f1b = [](int) { return NonCopyable(); };
    std::function f2b = +[](int) { return NonCopyable(); };
    static_assert(std::is_same_v<decltype(f1a), std::function<NonCopyable()>>);
    static_assert(std::is_same_v<decltype(f2a), std::function<NonCopyable()>>);
    static_assert(std::is_same_v<decltype(f1b), std::function<NonCopyable(int)>>);
    static_assert(std::is_same_v<decltype(f2b), std::function<NonCopyable(int)>>);
}

int main(int, char**)
{
    test();
    const_test();
    void_test();
    const_void_test();
    member_pointer_test();
    ctad_test();
    return 0;
}
