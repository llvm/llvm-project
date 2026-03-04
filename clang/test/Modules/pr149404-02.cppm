// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -o %t/format.pcm %t/format.cppm
// RUN: %clang_cc1 -std=c++20  -emit-module-interface -o %t/includes_in_gmf.pcm %t/includes_in_gmf.cppm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/test.cpp -verify -fsyntax-only

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface -o %t/format.pcm %t/format.cppm
// RUN: %clang_cc1 -std=c++20  -emit-reduced-module-interface -o %t/includes_in_gmf.pcm %t/includes_in_gmf.cppm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/test.cpp -verify -fsyntax-only

//--- format.h
#pragma once

namespace test {

template <class _Tp>
struct type_identity {
    typedef _Tp type;
};

template <class _Tp>
using type_identity_t = typename type_identity<_Tp>::type;


template <class _Tp, class _CharT>
struct formatter
{
    formatter() = delete;
};

template <>
struct formatter<char, char>
{};

template <class _CharT, class... _Args>
struct basic_format_string {
    static inline const int __handles_{ [] {
        formatter<char, _CharT> f;
        (void)f;
        return 0;
        }() };
    
    consteval basic_format_string(const _CharT*) {
        (void)__handles_;
    }
};

template <class... _Args>
using wformat_string = basic_format_string<wchar_t, type_identity_t<_Args>...>;

template <class... _Args>
using format_string = basic_format_string<char, type_identity_t<_Args>...>;

template <class... _Args>
void format(format_string<_Args...> __fmt, _Args&&... __args) {}

template <class... _Args>
void format(wformat_string<_Args...> __fmt, _Args&&... __args) {}

}

//--- format.cppm
module;
#include "format.h"
export module format;

export namespace test {
	using test::format;
	using test::formatter;
	using test::format_string;
}

auto something() -> void
{
	auto a = 'a';
	test::format("{}", a);
}

//--- includes_in_gmf.cppm
module;
#include "format.h"
export module includes_in_gmf;

namespace test {
	using test::format;
	using test::formatter;
	using test::format_string;
}

//--- test.cpp
// expected-no-diagnostics
import format;
import includes_in_gmf;

auto what() -> void
{
    auto a = 'a';
    test::format("{}", a);

    constexpr auto fs = "{}"; // test::format_string<char>{ "{}" }; // <- same result even passing exact param type
    test::format(fs, 'r');
}
