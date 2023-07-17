// RUN: %clang_cc1 -verify -fsyntax-only %s
// expected-no-diagnostics
// <rdar://problem/13153516> - This previously triggered an assertion failure.
template<class T>
struct X {
 T array;
};

int foo(X<int[1]> x0) {
 return x0.array[17];
}
