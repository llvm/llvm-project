// RUN: %clang_cc1 -fsyntax-only -verify %s
template<class T> struct A {
  static T t; // expected-error{{static data member instantiated with function type 'int ()'}}
};
typedef int function();
A<function> a; // expected-note{{instantiation of}}

template<typename T> struct B {
  B() { T t; } // expected-error{{variable instantiated with function type 'int ()'}}
};
B<function> b; // expected-note{{instantiation of}}

template <typename T> int f0(void *, const T&); // expected-note{{candidate template ignored: substitution failure}}
enum {e}; // expected-note{{unnamed type used in template argument was declared here}}

void test_f0(int n) {
  int i = f0(0, e); // expected-warning{{template argument uses unnamed type}}
  int vla[n];
  f0(0, vla); // expected-error{{no matching function for call to 'f0'}}
}

namespace N0 {
  template <typename R, typename A1> void f0(R (*)(A1));
  template <typename T> int f1(T);
  template <typename T, typename U> int f1(T, U);
  enum {e1}; // expected-note 2{{unnamed type used in template argument was declared here}}
  enum {e2}; // expected-note 2{{unnamed type used in template argument was declared here}}
  enum {e3}; // expected-note{{unnamed type used in template argument was declared here}}

  template<typename T> struct X;
  template<typename T> struct X<T*> { };

  void f() {
    f0( // expected-warning{{template argument uses unnamed type}}
       &f1<__typeof__(e1)>); // expected-warning{{template argument uses unnamed type}}
    int (*fp1)(int, __typeof__(e2)) = f1; // expected-warning{{template argument uses unnamed type}}
    f1(e2); // expected-warning{{template argument uses unnamed type}}
    f1(e2);

    X<__typeof__(e3)*> x; // expected-warning{{template argument uses unnamed type}}
  }
}
