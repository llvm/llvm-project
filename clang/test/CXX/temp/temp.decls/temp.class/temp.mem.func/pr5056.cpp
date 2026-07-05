// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

extern "C" void * malloc(int);

template <typename T> struct A {
  void *malloc(int);
};

template <typename T>
inline void *A<T>::malloc(int)
{
  return 0;
}

void f() {
  malloc(10);
}
