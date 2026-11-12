// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fdelayed-template-parsing -verify %s
// expected-no-diagnostics

// Dependent builtin calls used to assert during auto NTTP deduction.

template <auto> struct S {};

template <typename T> using Alias = S<__builtin_constant_p(T::x)>;
template <typename T> using Alias2 = S<__builtin_ffs(T::x)>;

struct HasX {
  static constexpr int x = 42;
};

using Inst = Alias<HasX>;
using Inst2 = Alias2<HasX>;

using Control = S<__builtin_constant_p(3)>;
