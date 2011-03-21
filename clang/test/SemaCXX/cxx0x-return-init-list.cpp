// RUN: %clang_cc1 -fsyntax-only -verify %s

// This test checks for a teeny tiny subset of the functionality in
// the C++0x generalized initializer lists feature, which happens to
// be used in libstdc++ 4.5. We accept only this syntax so that Clang
// can handle the libstdc++ 4.5 headers.

int test0(int i) {
  return { i }; // expected-warning{{generalized initializer lists are a C++0x extension unsupported in Clang}}
}

template<typename T, typename U>
T test1(U u) {
  return { u }; // expected-warning{{generalized initializer lists are a C++0x extension unsupported in Clang}}
}

template int test1(char);
template long test1(int);
