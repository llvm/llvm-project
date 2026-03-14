// RUN: %clang_cc1 -fsyntax-only %s -verify

// Shouldn't crash here
// Reported by https://github.com/llvm/llvm-project/issues/57008
template<class... Ts> bool b = __is_constructible(Ts...); // expected-error{{type trait requires 1 or more arguments}}
bool x = b<>; // expected-note{{in instantiation of variable template specialization}}
