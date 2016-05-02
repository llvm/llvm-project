//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// type_traits

// is_nothrow_callable

#include <type_traits>
#include <functional>

#include "test_macros.h"

struct Tag {};

struct Implicit {
  Implicit(int) noexcept {}
};

struct ThrowsImplicit {
  ThrowsImplicit(int) {}
};

struct Explicit {
  explicit Explicit(int) noexcept {}
};

template <bool IsNoexcept, class Ret, class ...Args>
struct CallObject {
  Ret operator()(Args&&...) const noexcept(IsNoexcept);
};

template <class Fn>
constexpr bool throws_callable() {
    return std::is_callable<Fn>::value &&
        !std::is_nothrow_callable<Fn>::value;
}

template <class Fn, class Ret>
constexpr bool throws_callable() {
    return std::is_callable<Fn, Ret>::value &&
        !std::is_nothrow_callable<Fn, Ret>::value;
}

// FIXME(EricWF) Don't test the where noexcept is *not* part of the type system
// once implementations have caught up.
void test_noexcept_function_pointers()
{
    struct Dummy { void foo() noexcept {} static void bar() noexcept {} };
#if !defined(__cpp_noexcept_function_type)
    {
        // Check that PMF's and function pointers *work*. is_nothrow_callable will always
        // return false because 'noexcept' is not part of the function type.
        static_assert(throws_callable<decltype(&Dummy::foo)(Dummy&)>());
        static_assert(throws_callable<decltype(&Dummy::bar)()>());
    }
#else
    {
        // Check that PMF's and function pointers actually work and that
        // is_nothrow_callable returns true for noexcept PMF's and function
        // pointers.
        static_assert(std::is_nothrow_callable<decltype(&Dummy::foo)(Dummy&)>::value);
        static_assert(std::is_nothrow_callable<decltype(&Dummy::bar)()>::value);
    }
#endif
}

int main()
{
    {
        // Check that the conversion to the return type is properly checked
        using Fn = CallObject<true, int>;
        static_assert(std::is_nothrow_callable<Fn(), Implicit>::value);
        static_assert(std::is_nothrow_callable<Fn(), double>::value);
        static_assert(std::is_nothrow_callable<Fn(), const volatile void>::value);
        static_assert(throws_callable<Fn(), ThrowsImplicit>());
        static_assert(!std::is_nothrow_callable<Fn(), Explicit>());
    }
    {
        // Check that the conversion to the parameters is properly checked
        using Fn = CallObject<true, void, const Implicit&, const ThrowsImplicit&>;
        static_assert(std::is_nothrow_callable<Fn(Implicit&, ThrowsImplicit&)>::value);
        static_assert(std::is_nothrow_callable<Fn(int, ThrowsImplicit&)>::value);
        static_assert(throws_callable<Fn(int, int)>());
        static_assert(!std::is_nothrow_callable<Fn()>::value);
    }
    {
        // Check that the noexcept-ness of function objects is checked.
        using Fn = CallObject<true, void>;
        using Fn2 = CallObject<false, void>;
        static_assert(std::is_nothrow_callable<Fn()>::value);
        static_assert(throws_callable<Fn2()>());
    }
    {
        // Check that PMD derefs are noexcept
        using Fn = int (Tag::*);
        static_assert(std::is_nothrow_callable<Fn(Tag&)>::value);
        static_assert(std::is_nothrow_callable<Fn(Tag&), Implicit>::value);
        static_assert(throws_callable<Fn(Tag&), ThrowsImplicit>());
    }
    {
        // Check for is_nothrow_callable_v
        using Fn = CallObject<true, int>;
        static_assert(std::is_nothrow_callable_v<Fn()>);
        static_assert(!std::is_nothrow_callable_v<Fn(int)>);
    }
    test_noexcept_function_pointers();
}
