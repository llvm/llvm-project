// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %s
// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -fms-compatibility-version=12.0 -verify %s

struct C {
  template <typename T> static int foo(T);
};

template <typename T> static int C::foo(T) { 
  //expected-warning@-1 {{'static' can only be specified inside the class definition}}
  return 0;
}

template <class T> struct S { 
  void f();
};

template <class T> static void S<T>::f() {}
#if _MSC_VER >= 1900
  //expected-error@-2 {{'static' can only be specified inside the class definition}}
#else
  //expected-warning@-4 {{'static' can only be specified inside the class definition}}
#endif
