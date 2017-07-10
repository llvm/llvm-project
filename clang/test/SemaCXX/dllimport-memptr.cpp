// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -verify -std=c++11 %s

// expected-no-diagnostics

struct __declspec(dllimport) Foo { int get_a(); };
template <int (Foo::*Getter)()> struct HasValue { };
HasValue<&Foo::get_a> hv;
