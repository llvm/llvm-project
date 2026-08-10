// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR5681
template <class T> struct Base {
  struct foo {};
  int foo;
};

template <class T> struct Derived : Base<T> {
  typedef struct Base<T>::foo type;
};

template struct Derived<int>;
