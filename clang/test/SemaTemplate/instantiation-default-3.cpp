// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T> struct A { };

template<typename T, typename U = A<T*> >
  struct B : U { };

template<>
struct A<int*> { 
  void foo();
};

template<>
struct A<float*> { 
  void bar();
};

void test(B<int> *b1, B<float> *b2) {
  b1->foo();
  b2->bar();
}
