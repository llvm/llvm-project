//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <functional>

// template<class R, class F, class... Args>
// constexpr R invoke_r(F&& f, Args&&... args)              // C++23
//     noexcept(is_nothrow_invocable_r_v<R, F, Args...>);

#include <cassert>
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility> // declval

template <class R, class F, class ...Args>
concept can_invoke_r = requires {
    { std::invoke_r<R>(std::declval<F>(), std::declval<Args>()...) } -> std::same_as<R>;
};

constexpr bool test() {
    // Make sure basic functionality works (i.e. we actually call the function and
    // return the right result).
    {
        auto f = [](int i) { return i + 3; };
        assert(std::invoke_r<int>(f, 4) == 7);
    }

    // Make sure invoke_r is SFINAE-friendly
    {
        auto f = [](int) -> char* { return nullptr; };
        static_assert( can_invoke_r<char*, decltype(f), int>);
        static_assert( can_invoke_r<void*, decltype(f), int>);
        static_assert( can_invoke_r<void,  decltype(f), int>);   // discard return type
        static_assert(!can_invoke_r<char*, decltype(f), void*>); // wrong argument type
        static_assert(!can_invoke_r<char*, decltype(f)>);        // missing argument
        static_assert(!can_invoke_r<int*,  decltype(f), int>);   // incompatible return type
        static_assert(!can_invoke_r<void,  decltype(f), void*>); // discard return type, invalid argument type
    }

    // Make sure invoke_r has the right noexcept specification
    {
        auto f = [](int) noexcept(true) -> char* { return nullptr; };
        auto g = [](int) noexcept(false) -> char* { return nullptr; };
        struct ConversionNotNoexcept {
            constexpr ConversionNotNoexcept(char*) noexcept(false) { }
        };
        static_assert( noexcept(std::invoke_r<char*>(f, 0)));
        static_assert(!noexcept(std::invoke_r<char*>(g, 0)));                 // function call is not noexcept
        static_assert(!noexcept(std::invoke_r<ConversionNotNoexcept>(f, 0))); // function call is noexcept, conversion isn't
        static_assert(!noexcept(std::invoke_r<ConversionNotNoexcept>(g, 0))); // function call and conversion are both not noexcept
    }

    // Make sure invoke_r works with cv-qualified void return type
    {
        auto check = []<class CV_Void> {
            bool was_called = false;
            auto f = [&](int) -> char* { was_called = true; return nullptr; };
            std::invoke_r<CV_Void>(f, 3);
            assert(was_called);
            static_assert(std::is_void_v<decltype(std::invoke_r<CV_Void>(f, 3))>);
        };
        check.template operator()<void>();
        check.template operator()<void const>();
        // volatile void is deprecated, so not testing it
        // const volatile void is deprecated, so not testing it
    }

    // Make sure invoke_r forwards its arguments
    {
        struct NonCopyable {
            NonCopyable() = default;
            NonCopyable(NonCopyable const&) = delete;
            NonCopyable(NonCopyable&&) = default;
        };
        // Forward argument, with void return
        {
            bool was_called = false;
            auto f = [&](NonCopyable) { was_called = true; };
            std::invoke_r<void>(f, NonCopyable());
            assert(was_called);
        }
        // Forward argument, with non-void return
        {
            bool was_called = false;
            auto f = [&](NonCopyable) -> int { was_called = true; return 0; };
            std::invoke_r<int>(f, NonCopyable());
            assert(was_called);
        }
        // Forward function object, with void return
        {
            struct MoveOnlyVoidFunction {
                bool& was_called;
                constexpr void operator()() && { was_called = true; }
            };
            bool was_called = false;
            std::invoke_r<void>(MoveOnlyVoidFunction{was_called});
            assert(was_called);
        }
        // Forward function object, with non-void return
        {
            struct MoveOnlyIntFunction {
                bool& was_called;
                constexpr int operator()() && { was_called = true; return 0; }
            };
            bool was_called = false;
            std::invoke_r<int>(MoveOnlyIntFunction{was_called});
            assert(was_called);
        }
    }

    // Make sure invoke_r performs an implicit conversion of the result
    {
        struct Convertible {
            constexpr operator int() const { return 42; }
        };
        auto f = []() -> Convertible { return Convertible{}; };
        int result = std::invoke_r<int>(f);
        assert(result == 42);
    }

    // Note: We don't test that `std::invoke_r` works with all kinds of callable types here,
    //       since that is extensively tested in the `std::invoke` tests.

    return true;
}

int main(int, char**) {
    test();
    static_assert(test());
    return 0;
}
