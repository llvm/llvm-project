// RUN: %clang_cc1 -verify -fsyntax-only %s
// expected-no-diagnostics
// This previously triggered an assertion failure.
template<class T>
struct X {
 T array;
};

int foo(X<int[1]> x0) {
 return x0.array[17];
}
