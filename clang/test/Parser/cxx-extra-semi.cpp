// RUN: %clang_cc1 -fsyntax-only -pedantic -verify -DPEDANTIC %s
// RUN: %clang_cc1 -fsyntax-only -Wextra-semi -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wextra-semi -verify -std=c++11 %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wextra-semi -fixit -DFIXIT %t
// RUN: %clang_cc1 -x c++ -Wextra-semi -Werror -DFIXIT %t

class A {
  void A1();
  void A2() { };
#ifndef PEDANTIC
  // This warning is only produced if we specify -Wextra-semi, and not if only
  // -pedantic is specified, since one semicolon is technically permitted.
  // expected-warning@-4{{extra ';' after member function definition}}
#endif
  void A2b() { };; // expected-warning{{extra ';' after member function definition}}
  ; // expected-warning{{extra ';' inside a class}}
  void A2c() { }
  ;
#ifndef PEDANTIC
  // expected-warning@-2{{extra ';' after member function definition}}
#endif
  void A3() { };  ;; // expected-warning{{extra ';' after member function definition}}
  ;;;;;;; // expected-warning{{extra ';' inside a class}}
  ; // expected-warning{{extra ';' inside a class}}
  ; ;;		 ;  ;;; // expected-warning{{extra ';' inside a class}}
    ;  ; 	;	;  ;; // expected-warning{{extra ';' inside a class}}
  void A4();
};

union B {
  int a1;
  int a2;; // expected-warning{{extra ';' inside a union}}
};

;
; ;;
#if __cplusplus < 201103L
// expected-warning@-3{{extra ';' outside of a function is a C++11 extension}}
// expected-warning@-3{{extra ';' outside of a function is a C++11 extension}}
#elif !defined(PEDANTIC)
// expected-warning@-6{{extra ';' outside of a function is incompatible with C++98}}
// expected-warning@-6{{extra ';' outside of a function is incompatible with C++98}}
#endif

#ifndef FIXIT
namespace GH112377 {
struct B { };
struct Base {
  Base(struct B;); // expected-error {{unexpected ';' before ')'}} \
                   // expected-note {{member is declared here}}
};

class Forward : Base { // expected-note {{constrained by implicitly private inheritance here}} \
                       // expected-note {{candidate constructor (the implicit copy constructor) not viable: cannot convert argument of incomplete type 'B' to 'const Forward' for 1st argument}} \
                       // expected-note {{candidate constructor (the implicit move constructor) not viable: cannot convert argument of incomplete type 'B' to 'Forward' for 1st argument}} \
                       // expected-note {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  using Base::Base;
};

class A : Forward {
  A();
};

A::A() : Forward(B()) { } // expected-error {{'B' is a private member of 'GH112377::Base'}} \
                          // expected-error {{no matching constructor for initialization of 'Forward'}}
}
#endif
