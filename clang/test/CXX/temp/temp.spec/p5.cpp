// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> inline void f(T) { }
template void f(int); // expected-note{{previous explicit instantiation}}
template void f(int); // expected-error{{duplicate explicit instantiation}}

template<typename T>
struct X0 {
  union Inner { };
  
  void f(T) { }
  
  static T value;
};

template<typename T>
T X0<T>::value = 3.14; // expected-warning{{implicit conversion from 'double' to 'int' changes value from 3.14 to 3}}

template struct X0<int>; // expected-note{{previous explicit instantiation}} \
                            expected-note{{requested here}}
template struct X0<int>; // expected-error{{duplicate explicit instantiation}}

template void X0<float>::f(float); // expected-note{{previous explicit instantiation}}
template void X0<float>::f(float); // expected-error{{duplicate explicit instantiation}}

template union X0<float>::Inner; // expected-note{{previous explicit instantiation}}
template union X0<float>::Inner; // expected-error{{duplicate explicit instantiation}}

template float X0<float>::value; // expected-note{{previous explicit instantiation}}
template float X0<float>::value; // expected-error{{duplicate explicit instantiation}}

// Make sure that we don't get tricked by redeclarations of nested classes.
namespace NestedClassRedecls {
  template<typename T>
  struct X {
    struct Nested;
    friend struct Nested;

    struct Nested { 
      Nested() {}
    } nested;
  };

  X<int> xi;

  template struct X<int>;
}
