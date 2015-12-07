// RUN: %clang_cc1 -std=c++11 -verify -emit-llvm-only %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify %s -DCPP98

namespace std {
  template <class _E>
  class initializer_list
  {};
}

template<class E> int f(std::initializer_list<E> il);
	

int F = f({1, 2, 3});
#ifdef CPP98
//expected-error@-2{{expected expression}}
#else
//expected-error@-4{{cannot compile}}
#endif


