// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test that a very basic variation of generalized initializer returns (that
// required for libstdc++ 4.5) is supported in C++98.

int test0(int i) {
  return { i }; // expected-warning{{generalized initializer lists are a C++11 extension}} expected-warning {{scalar}}
}

template<typename T, typename U>
T test1(U u) {
  return { u }; // expected-warning{{generalized initializer lists are a C++11 extension}}
}

template int test1(char);
template long test1(int);
