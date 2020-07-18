// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

consteval int Fun() { return; } // expected-error {{non-void constexpr function 'Fun' should return a value}}

// FIXME: The diagnostic is wrong; should be "consteval".

template <typename T> consteval int FunT1() { return; } // expected-error {{non-void constexpr function 'FunT1' should return a value}}
template <typename T> consteval int FunT2() { return 0; }
template <> consteval int FunT2<double>() { return 0; }
template <> consteval int FunT2<int>() { return; } // expected-error {{non-void constexpr function 'FunT2' should return a value}}
