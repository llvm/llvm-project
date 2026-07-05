// RUN: %clang_cc1 -std=c++20 -fms-compatibility -fms-compatibility-version=19.33 -emit-llvm %s -o - -triple=x86_64-windows-msvc | FileCheck %s

template <typename T>
concept C = requires
{
    { T::test([](){}) };
};

template<typename T>
struct Widget {};

template <C T>
struct Widget<T> {};

struct Baz
{
    template<typename F>
    static constexpr decltype(auto) test(F&&) {}
};

void test()
{
    Widget<Baz> w;
}
// CHECK: @"?test@@YAXXZ"
