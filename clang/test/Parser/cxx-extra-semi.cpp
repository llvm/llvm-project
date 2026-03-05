// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=compat98 -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=cxx11_pedantic -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -Wextra-semi -verify=wextra,compat98 -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -Wextra-semi -verify=wextra,compat11 -std=c++11 %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wextra-semi -fixit %t
// RUN: %clang_cc1 -x c++ -Wextra-semi -Werror %t

// In C++11 extra semicolons inside classes are allowed via defect reports.
// cxx11_pedantic-no-diagnostics
 
class A {
  void A1();
  void A2() { };
  // This warning is only produced if we specify -Wextra-semi, and not if only
  // -pedantic is specified, since one semicolon is technically permitted.
  // wextra-warning@-3{{extra ';' after member function definition}}
  void A2b() { };;
  // compat98-warning@-1{{multiple extra ';' after member function definition is a C++11 extension}}
  // compat11-warning@-2{{multiple extra ';' after member function definition is incompatible with C++98}}
  ;
  // compat98-warning@-1{{extra ';' inside a class is a C++11 extension}}
  // compat11-warning@-2{{extra ';' inside a class is incompatible with C++98}}
  void A2c() { }
  ; // wextra-warning{{extra ';' after member function definition}}
  void A3() { };  ;;
  // compat98-warning@-1{{multiple extra ';' after member function definition is a C++11 extension}}
  // compat11-warning@-2{{multiple extra ';' after member function definition is incompatible with C++98}}
  ;;;;;;;
  // compat98-warning@-1{{extra ';' inside a class is a C++11 extension}}
  // compat11-warning@-2{{extra ';' inside a class is incompatible with C++98}}
  ;
  // compat98-warning@-1{{extra ';' inside a class is a C++11 extension}}
  // compat11-warning@-2{{extra ';' inside a class is incompatible with C++98}}
  ; ;;		 ;  ;;;
  // compat98-warning@-1{{extra ';' inside a class is a C++11 extension}}
  // compat11-warning@-2{{extra ';' inside a class is incompatible with C++98}}
    ;  ; 	;	;  ;;
  // compat98-warning@-1{{extra ';' inside a class is a C++11 extension}}
  // compat11-warning@-2{{extra ';' inside a class is incompatible with C++98}}
  void A4();
  
  union {
    ;
    // compat98-warning@-1{{extra ';' inside a union is a C++11 extension}}
    // compat11-warning@-2{{extra ';' inside a union is incompatible with C++98}}
    int a;
    ;
    // compat98-warning@-1{{extra ';' inside a union is a C++11 extension}}
    // compat11-warning@-2{{extra ';' inside a union is incompatible with C++98}}
  };

  virtual void f() = 0;
  virtual void g() = 0;;
  // compat11-warning@-1{{extra ';' inside a class is incompatible with C++98}}
  // compat98-warning@-2{{extra ';' inside a class is a C++11 extension}}

#if __cplusplus >= 201103L
  void h() = delete;
  void i() = delete;; // compat11-warning{{extra ';' inside a class is incompatible with C++98}}
  void j() = delete;;; // compat11-warning{{extra ';' inside a class is incompatible with C++98}}
  
  A(const A&) = default;
  A(A&&) = default;;  // compat11-warning{{extra ';' inside a class is incompatible with C++98}}
  A(A&) = default;;;  // compat11-warning{{extra ';' inside a class is incompatible with C++98}}
#endif
};

union B {
  int a1;
  int a2;;
  // compat11-warning@-1{{extra ';' inside a union is incompatible with C++98}}
  // compat98-warning@-2{{extra ';' inside a union is a C++11 extension}}
};

;
; ;;
// compat98-warning@-2{{extra ';' outside of a function is a C++11 extension}}
// compat98-warning@-2{{extra ';' outside of a function is a C++11 extension}}
// compat11-warning@-4{{extra ';' outside of a function is incompatible with C++98}}
// compat11-warning@-4{{extra ';' outside of a function is incompatible with C++98}}

