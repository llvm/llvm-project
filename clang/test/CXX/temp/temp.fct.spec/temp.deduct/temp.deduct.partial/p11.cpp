// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T> T* f(int);	// #1 
template <class T, class U> T& f(U); // #2 

void g() {
  int *ip = f<int>(1);	// calls #1
}

template<typename T>
struct identity {
  typedef T type;
};

template <class T> 
  T* f2(int, typename identity<T>::type = 0);
template <class T, class U> 
  T& f2(U, typename identity<T>::type = 0);

void g2() {
  int* ip = f2<int>(1);
}

template<class T, class U> struct A { };

template<class T, class U> inline int *f3( U, A<U,T>* p = 0 ); // #1 expected-note{{candidate function [with T = int, U = int]}}
template<         class U> inline float *f3( U, A<U,U>* p = 0 ); // #2 expected-note{{candidate function [with U = int]}}

void g3() {
   float *fp = f3<int>( 42, (A<int,int>*)0 );  // Ok, picks #2.
   f3<int>( 42 );                  // expected-error{{call to 'f3' is ambiguous}}
   
}

namespace PR9006 {
  struct X {
    template <class Get>
    int &f(char const* name, Get fget, char const* docstr = 0);
  
    template <class Get, class Set>
    float &f(char const* name, Get fget, Set fset, char const* docstr = 0);
  };

  void test(X x) {
    int &ir = x.f("blah", 0, "blah");
  }
}
