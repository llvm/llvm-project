// RUN: %clang_cc1 %s -cl-std=c++ -pedantic -verify -fsyntax-only

struct C {
  kernel void m(); //expected-error{{kernel functions cannot be class members}}
};

template <typename T>
kernel void templ(T par) { //expected-error{{kernel functions cannot be used in a template declaration, instantiation or specialization}}
}

template <int>
kernel void bar(int par) { //expected-error{{kernel functions cannot be used in a template declaration, instantiation or specialization}}
}

kernel void foo(int); //expected-note{{previous declaration is here}}

kernel void foo(float); //expected-error{{conflicting types for 'foo'}}
