// RUN: %clang_cc1 -fsyntax-only -pedantic -verify -DPEDANTIC %s
// RUN: %clang_cc1 -fsyntax-only -Wextra-semi -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wextra-semi -verify -std=c++11 %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wextra-semi -fixit %t
// RUN: %clang_cc1 -x c++ -Wextra-semi -Werror %t

#if __cplusplus >= 201103L && defined(PEDANTIC)
// In C++11 extra semicolons inside classes are allowed via defect reports.
// expected-no-diagnostics
class A {
  void A1();
  void A2() { };
  void A2b() { };; 
  ; 
  void A2c() { }
  ;
  void A3() { };  ;; 
  ;;;;;;; 
  ; 
  ; ;;		 ;  ;;; 
    ;  ; 	;	;  ;; 
  void A4();

  union {
    ;
    int a;
    ;
  };
};

union B {
  int a1;
  int a2;; 
};

;
; ;;

#else

class A {
  void A1();
  void A2() { };
#ifndef PEDANTIC
  // This warning is only produced if we specify -Wextra-semi, and not if only
  // -pedantic is specified, since one semicolon is technically permitted.
  // expected-warning@-4{{extra ';' after member function definition}}
#endif
  void A2b() { };; // expected-warning{{multiple extra ';' after member function definition}}
  ; // expected-warning{{extra ';' inside a class}}
  void A2c() { }
  ;
#ifndef PEDANTIC
  // expected-warning@-2{{extra ';' after member function definition}}
#endif
  void A3() { };  ;; // expected-warning{{multiple extra ';' after member function definition}}
  ;;;;;;; // expected-warning{{extra ';' inside a class}}
  ; // expected-warning{{extra ';' inside a class}}
  ; ;;		 ;  ;;; // expected-warning{{extra ';' inside a class}}
    ;  ; 	;	;  ;; // expected-warning{{extra ';' inside a class}}
  void A4();
  
  union {
    ; // expected-warning{{extra ';' inside a union}}
    int a;
    ; // expected-warning{{extra ';' inside a union}}
  };
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

#endif
