// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Regression test for GH192846: the default TransformOpaqueValueExpr
// asserts on OVEs bound by __builtin_dump_struct when the printing
// callback is immediate-escalated. ComplexRemove must not reach that
// path.

struct S {};

consteval void F(S &out, const char *fmt, ...) {}

template <class T>
class C { T value = {}; };

constexpr C<int> g_c{};

void bar() {
  S s;
  __builtin_dump_struct(&g_c, F, s);
}
