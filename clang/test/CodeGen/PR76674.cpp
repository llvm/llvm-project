// RUN: %clang_cc1 -std=c++20 -emit-llvm -o - %s
// expected-no-diagnostics

template <class>
struct A {
    template <class U>
    using Func = decltype([] {return U{};});
};

A<int>::Func<int> f{};
int i{f()};
