// RUN: %clang_cc1 -fsyntax-only -verify -Wvla %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

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
enum {e};
#if __cplusplus <= 199711L
// expected-note@-2 {{unnamed type used in template argument was declared here}}
#endif

void test_f0(int n) { // #here
  int i = f0(0, e);
#if __cplusplus <= 199711L
  // expected-warning@-2 {{template argument uses unnamed type}}
#endif

  int vla[n]; // expected-warning {{variable length arrays in C++ are a Clang extension}}
#if __cplusplus > 199711L
  // expected-note@-2 {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  // expected-note@#here {{declared here}}
#endif
  f0(0, vla); // expected-error{{no matching function for call to 'f0'}}
}

namespace N0 {
  template <typename R, typename A1> void f0(R (*)(A1));
  template <typename T> int f1(T);
  template <typename T, typename U> int f1(T, U);
  enum {e1};
#if __cplusplus <= 199711L
  // expected-note@-2 2{{unnamed type used in template argument was declared here}}
#endif

  enum {e2};
#if __cplusplus <= 199711L
  // expected-note@-2 2{{unnamed type used in template argument was declared here}}
#endif

  enum {e3};
#if __cplusplus <= 199711L
 // expected-note@-2 {{unnamed type used in template argument was declared here}}
#endif

  template<typename T> struct X;
  template<typename T> struct X<T*> { };

  void f() {
    f0(
#if __cplusplus <= 199711L
    // expected-warning@-2 {{template argument uses unnamed type}}
#endif

       &f1<__typeof__(e1)>);
#if __cplusplus <= 199711L
 // expected-warning@-2 {{template argument uses unnamed type}}
#endif

    int (*fp1)(int, __typeof__(e2)) = f1;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{template argument uses unnamed type}}
#endif

    f1(e2);
#if __cplusplus <= 199711L
    // expected-warning@-2 {{template argument uses unnamed type}}
#endif

    f1(e2);

    X<__typeof__(e3)*> x;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{template argument uses unnamed type}}
#endif
  }
}
