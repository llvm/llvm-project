// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5868
struct T0 {
  int x;
  union {
    void *m0;
  };
};
template <typename T>
struct T1 : public T0, public T { //expected-warning{{direct base 'T0' is inaccessible due to ambiguity:\n    struct T1<struct A> -> T0\n    struct T1<struct A> -> A -> T0}}
  void f0() { 
    m0 = 0; // expected-error{{ambiguous conversion}}
  } 
};

struct A : public T0 { };

void f1(T1<A> *S) { S->f0(); } // expected-note{{instantiation of member function}} expected-note{{in instantiation of template class 'T1<A>' requested here}}

namespace rdar8635664 {
  template<typename T>
  struct X {
    struct inner;
  
    struct inner {
      union {
        int x;
        float y;
      };

      typedef T type;
    };
  };

  void test() {
    X<int>::inner i;
    i.x = 0;
  }
}
