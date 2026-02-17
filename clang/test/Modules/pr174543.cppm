// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics
export module m;
template < typename T >
void fun(T)
{
    fun(9);
}
