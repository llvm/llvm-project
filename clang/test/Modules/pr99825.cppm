// RUN: %clang_cc1 -std=c++20 %s -fsyntax-only -verify
// expected-no-diagnostics
export module mod;

extern "C++"
{
    export constexpr auto x = 10;
}
