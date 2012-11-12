// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
struct X1 {
  static void member() { T* x = 1; } // expected-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
};

template<void(*)()> struct instantiate { };

template<typename T>
struct X2 {
  typedef instantiate<&X1<int>::member> i; // expected-note{{in instantiation of}}
};

X2<int> x; 
