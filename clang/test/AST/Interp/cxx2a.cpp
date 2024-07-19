// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fcxx-exceptions -verify=ref,both %s
// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fcxx-exceptions -verify=expected,both %s -fexperimental-new-constant-interpreter

template <unsigned N>
struct S {
  S() requires (N==1) = default;
  S() requires (N==2) {} // both-note {{declared here}}
  consteval S() requires (N==3) = default;
};

consteval int aConstevalFunction() { // both-error {{consteval function never produces a constant expression}}
  S<2> s4; // both-note {{non-constexpr constructor 'S' cannot be used in a constant expression}}
  return 0;
}
/// We're NOT calling the above function. The diagnostics should appear anyway.
