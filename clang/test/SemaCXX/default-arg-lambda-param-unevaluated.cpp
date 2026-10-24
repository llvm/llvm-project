// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// Using a function parameter in the body of a lambda requires capturing it,
// even if the lambda appears in an unevaluated operand.

int p(int x, int = sizeof(x));
int q(int x, int = sizeof([] { return x; }())); // expected-error {{default argument references parameter 'x'}}
int r(int x, int = decltype([] { return x; }())()); // expected-error {{default argument references parameter 'x'}}
int s(int x, bool = noexcept([] { return x; }())); // expected-error {{default argument references parameter 'x'}}
