// RUN: %clang_cc1 -ast-print -std=c++17 -Wno-implicitly-unsigned-literal %s -o - | FileCheck %s

template <unsigned long long N>
struct Foo {};

Foo<9223372036854775810> x;
// CHECK: template<> struct Foo<9223372036854775810ULL> {
