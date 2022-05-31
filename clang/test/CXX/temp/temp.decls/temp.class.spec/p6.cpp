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
  template<typename Y> struct Inner<T, Y> {}; // expected-note{{previous declaration of class template partial specialization 'Inner<int, type-parameter-0-0>' is here}}
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

namespace print_dependent_TemplateSpecializationType {

template <class T, class U> struct Foo {
  template <unsigned long, class X, class Y> struct Bar;
  template <class Y> struct Bar<0, T, Y> {};
  // expected-note-re@-1 {{previous declaration {{.*}} 'Bar<0UL, int, type-parameter-0-0>' is here}}
  template <class Y> struct Bar<0, U, Y> {};
  // expected-error@-1 {{partial specialization 'Bar<0, int, Y>' cannot be redeclared}}
};
template struct Foo<int, int>; // expected-note {{requested here}}

} // namespace print_dependent_TemplateSpecializationType
