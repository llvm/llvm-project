// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

class C {
  friend class D;
};

class A {
public:
  void f();
};

friend int x; // expected-error {{'friend' used outside of class}}

friend class D {}; // expected-error {{'friend' used outside of class}}

union U {
  int u1;
};

class B {
  // 'A' here should refer to the declaration above.  
  friend class A;

  friend C;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{unelaborated friend declaration is a C++11 extension; specify 'class' to befriend 'C'}}
#endif

  friend U;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{unelaborated friend declaration is a C++11 extension; specify 'union' to befriend 'U'}}
#endif

  friend int;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{non-class friend type 'int' is a C++11 extension}}
#endif

  friend void myfunc();

  void f(A *a) { a->f(); }
};

inline void bar() {} // expected-note {{previous definition is here}}
class E {
  friend void bar() {} // expected-error {{redefinition of 'bar'}}
};




template <typename t1, typename t2> class some_template;
friend   // expected-error {{'friend' used outside of class}}
some_template<foo, bar>&  // expected-error {{use of undeclared identifier 'foo'}}
  ;  // expected-error {{expected unqualified-id}}
