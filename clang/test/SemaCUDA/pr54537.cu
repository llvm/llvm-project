// Regression test for the crash in
// https://github.com/llvm/llvm-project/issues/54537
//
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

template< class T > inline constexpr bool test_v = true;

template <typename T>
struct A {
    A(const T = 1 ) requires test_v<T>;
};

struct B :  A<int> {
    using A<int>::A;
};
