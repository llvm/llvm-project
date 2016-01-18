// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T> struct vector;

// C++ [temp.class.spec]p6:
namespace N {
  namespace M {
    template<typename T> struct A; // expected-note{{here}}
  }
}

template<typename T>
struct N::M::A<T*> { }; // expected-warning{{C++11 extension}}

// C++ [temp.class.spec]p9
//   bullet 1
template <int I, int J> struct A {}; 
template <int I> struct A<I+5, I*2> {}; // expected-error{{depends on}} 
template <int I, int J> struct B {}; 
template <int I> struct B<I, I> {}; //OK 

//   bullet 2
template <class T, T t> struct C {};  // expected-note{{declared here}}
template <class T> struct C<T, 1>; // expected-error{{specializes}}
template <class T, T* t> struct C<T*, t>; // okay

template< int X, int (*array_ptr)[X] > class A2 {}; // expected-note{{here}}
int array[5]; 
template< int X > class A2<X, &array> { }; // expected-error{{specializes}}

template<typename T, int N, template<typename X> class TT>
struct Test0;

//   bullet 3
template<typename T, int N, template<typename X> class TT>
struct Test0<T, N, TT>; // expected-error{{does not specialize}}

// C++ [temp.class.spec]p10
template<typename T = int, // expected-error{{default template argument}}
         int N = 17, // expected-error{{default template argument}}
         template<typename X> class TT = ::vector> // expected-error{{default template argument}}
  struct Test0<T*, N, TT> { };

template<typename T> struct Test1;
template<typename T, typename U>  // expected-note{{non-deducible}}
  struct Test1<T*> { }; // expected-warning{{never be used}}
