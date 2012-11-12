// RUN: %clang_cc1  -fsyntax-only -verify %s

#define NULL (void*)0

#define ATTR __attribute__ ((__sentinel__)) 

void foo1 (int x, ...) ATTR; // expected-note 3 {{function has been explicitly marked sentinel here}}
void foo5 (int x, ...) __attribute__ ((__sentinel__(1))); // expected-note {{function has been explicitly marked sentinel here}}
void foo6 (int x, ...) __attribute__ ((__sentinel__(5))); // expected-note {{function has been explicitly marked sentinel here}}
void foo7 (int x, ...) __attribute__ ((__sentinel__(0))); // expected-note {{function has been explicitly marked sentinel here}}
void foo10 (int x, ...) __attribute__ ((__sentinel__(1,1)));
void foo12 (int x, ... ) ATTR; // expected-note {{function has been explicitly marked sentinel here}}

#define FOOMACRO(...) foo1(__VA_ARGS__)

void test1() {
  foo1(1, NULL); // OK
  foo1(1, 0) ; // expected-warning {{missing sentinel in function call}}
  foo5(1, NULL, 2);  // OK
  foo5(1,2,NULL, 1); // OK
  foo5(1, NULL, 2, 1); // expected-warning {{missing sentinel in function call}}

  foo6(1,2,3,4,5,6,7); // expected-warning {{missing sentinel in function call}}
  foo6(1,NULL,3,4,5,6,7); // OK
  foo7(1); // expected-warning {{not enough variable arguments in 'foo7' declaration to fit a sentinel}}
  foo7(1, NULL); // OK

  foo12(1); // expected-warning {{not enough variable arguments in 'foo12' declaration to fit a sentinel}}

  // PR 5685
  struct A {};
  struct A a, b, c;
  foo1(3, &a, &b, &c); // expected-warning {{missing sentinel in function call}}
  foo1(3, &a, &b, &c, (struct A*) 0);

  // PR11002
  FOOMACRO(1, 2); // expected-warning {{missing sentinel in function call}}
}
 


void (*e) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (1,1)));

void test2() {
  void (*b) (int arg, const char * format, ...) __attribute__ ((__sentinel__));  // expected-note {{function has been explicitly marked sentinel here}}
  void (*z) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (2))); // expected-note {{function has been explicitly marked sentinel here}}
  
  
  void (*y) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (5))); // expected-note {{function has been explicitly marked sentinel here}}
  
  b(1, "%s", (void*)0); // OK
  b(1, "%s", 0);  // expected-warning {{missing sentinel in function call}}
  z(1, "%s",4 ,1,0);  // expected-warning {{missing sentinel in function call}}
  z(1, "%s", (void*)0, 1, 0); // OK
  
  y(1, "%s", 1,2,3,4,5,6,7);  // expected-warning {{missing sentinel in function call}}
  
  y(1, "%s", (void*)0,3,4,5,6,7); // OK
}
