// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test class template partial specializations of member templates.
template<typename T>
struct X0 {
  template<typename U> struct Inner0 {
    static const unsigned value = 0;
  };
  
  template<typename U> struct Inner0<U*> { 
    static const unsigned value = 1;
  };
};

template<typename T> template<typename U>
struct X0<T>::Inner0<const U*> {
  static const unsigned value = 2;
};

int array0[X0<int>::Inner0<int>::value == 0? 1 : -1];
int array1[X0<int>::Inner0<int*>::value == 1? 1 : -1];
int array2[X0<int>::Inner0<const int*>::value == 2? 1 : -1];

// Make sure we can provide out-of-line class template partial specializations
// for member templates (and instantiate them).
template<class T> struct A { 
  struct C {
    template<class T2> struct B;
  };
};

// partial specialization of A<T>::C::B<T2> 
template<class T> template<class T2> struct A<T>::C::B<T2*> { }; 

A<short>::C::B<int*> absip;

// Check for conflicts during template instantiation. 
template<typename T, typename U>
struct Outer {
  template<typename X, typename Y> struct Inner;
  template<typename Y> struct Inner<T, Y> {}; // expected-note{{previous}}
  template<typename Y> struct Inner<U, Y> {}; // expected-error{{cannot be redeclared}}
};

Outer<int, int> outer; // expected-note{{instantiation}}

// Test specialization of class template partial specialization members.
template<> template<typename Z>
struct X0<float>::Inner0<Z*> {
  static const unsigned value = 3;
};

int array3[X0<float>::Inner0<int>::value == 0? 1 : -1];
int array4[X0<float>::Inner0<int*>::value == 3? 1 : -1];
int array5[X0<float>::Inner0<const int*>::value == 2? 1 : -1];

namespace rdar8651930 {
  template<typename OuterT>
  struct Outer {
    template<typename T, typename U>
    struct Inner;

    template<typename T>
    struct Inner<T, T> { 
      static const bool value = true;
    };

    template<typename T, typename U>
    struct Inner { 
      static const bool value = false;
    };
  };

  int array0[Outer<int>::Inner<int, int>::value? 1 : -1];
  int array1[Outer<int>::Inner<int, float>::value? -1 : 1];
}
