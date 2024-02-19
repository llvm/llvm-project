// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -fblocks %s

// PR2241
float test2241[2] = { 
  1e,            // expected-error {{exponent}}
  1ee0           // expected-error {{exponent}}
};


// Testcase derived from PR2692
static void f (char * (*g) (char **, int), char **p, ...) {
  char *s;
  va_list v;                              // expected-error {{identifier}}
  s = g (p, __builtin_va_arg(v, int));    // expected-error {{identifier}}
}


// PR3172
} // expected-error {{extraneous closing brace ('}')}}


void test(int a) {
  struct { int i; } x;
  
  if (x.hello)   // expected-error {{no member named 'hello'}}
    test(0);
  else
    ;
  
  if (x.hello == 0)   // expected-error {{no member named 'hello'}}
    test(0);
  else
    ;
  
  if ((x.hello == 0))   // expected-error {{no member named 'hello'}}
    test(0);
  else
    ;

  // PR12595
  if (x.i == 0))   // expected-error {{extraneous ')' after condition, expected a statement}}
    test(0);
  else
    ;
}



char ((((                       /* expected-note {{to match this '('}} */
         *X x ] ))));                    /* expected-error {{expected ')'}} */

;   // expected-warning {{extra ';' outside of a function}}




struct S { void *X, *Y; };

struct S A = {
&BADIDENT, 0     /* expected-error {{use of undeclared identifier}} */
};

void test6248081(void) { 
  [10]  // expected-error {{expected expression}}
}

struct forward; // expected-note{{forward declaration of 'struct forward'}}
void x(struct forward* x) {switch(x->a) {}} // expected-error {{incomplete definition of type}}

// PR3410
void foo(void) {
  int X;
  X = 4 // expected-error{{expected ';' after expression}}
}

void test9045701(int x) {
#define VALUE 0
  x = VALUE // expected-error{{expected ';' after expression}}
}

typedef int intptr_t;  // expected-note {{'intptr_t' declared here}}
void bar(intptr y);     // expected-error {{unknown type name 'intptr'; did you mean 'intptr_t'?}}

void test1(void) {
  int x = 2: // expected-error {{expected ';' at end of declaration}}
  int y = x;
  int z = y;
}

void test2(int x) {
#define VALUE2 VALUE+VALUE
#define VALUE3 VALUE+0
#define VALUE4(x) x+0
  x = VALUE2 // expected-error{{expected ';' after expression}}
  x = VALUE3 // expected-error{{expected ';' after expression}}
  x = VALUE4(0) // expected-error{{expected ';' after expression}}
}
