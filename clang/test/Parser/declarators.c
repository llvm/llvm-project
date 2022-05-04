// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -std=c99

extern int a1[];

void f0(); /* expected-warning {{a function declaration without a prototype is deprecated in all versions of C}} */
void f1(int [*]);
void f2(int [const *]);
void f3(int [volatile const*]);
int f4(*XX)(void); /* expected-error {{cannot return}} expected-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} */
int f5(int [static]); /* expected-error {{'static' may not be used without an array size}} */

char ((((*X))));

void (*signal(int, void (*)(int)))(int);

int aaaa, ***C, * const D, B(int);

int *A;

struct str;

void test2(int *P, int A) {
  struct str;

  // Hard case for array decl, not Array[*].
  int Array[*(int*)P+A];
}

typedef int atype;
void test3(x,            /* expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}} */
           atype         /* expected-error {{unexpected type name 'atype': expected identifier}} */
          ) int x, atype; {}

void test4(x, x) int x; {} // expected-error {{redefinition of parameter 'x'}} \
                           // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}


// PR3031
int (test5), ;  // expected-error {{expected identifier or '('}}



// PR3963 & rdar://6759604 - test error recovery for mistyped "typenames".

foo_t *d;      // expected-error {{unknown type name 'foo_t'}}
foo_t a;   // expected-error {{unknown type name 'foo_t'}}
int test6() { /* expected-warning {{a function declaration without a prototype is deprecated in all versions of C}} */
  return a; // a should be declared.
}

// Use of tagged type without tag. rdar://6783347
struct xyz { int y; };
enum myenum { ASDFAS };
xyz b;         // expected-error {{must use 'struct' tag to refer to type 'xyz'}}
myenum c;      // expected-error {{must use 'enum' tag to refer to type 'myenum'}}

float *test7(void) {
  // We should recover 'b' by parsing it with a valid type of "struct xyz", which
  // allows us to diagnose other bad things done with y, such as this.
  return &b.y;   // expected-warning {{incompatible pointer types returning 'int *' from a function with result type 'float *'}}
}

struct xyz test8(void) { return a; }  // a should be be marked invalid, no diag.


// Verify that implicit int still works.
static f;      // expected-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
static g = 4;  // expected-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
static h        // expected-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
      __asm__("foo");


struct test9 {
  int x  // expected-error {{expected ';' at end of declaration list}}
  int y;
  int z  // expected-warning {{expected ';' at end of declaration list}}
};

// PR6208
struct test10 { int a; } static test10x;
struct test11 { int a; } const test11x;

// PR6216
void test12(void) {
  (void)__builtin_offsetof(struct { char c; int i; }, i);
}

// rdar://7608537
struct test13 { int a; } (test13x);

// <rdar://problem/8044088>
struct X<foo::int> { }; // expected-error{{expected identifier or '('}}


// PR7617 - error recovery on missing ;.

void test14(void)  // expected-error {{expected ';' after top level declarator}}

void test14a(void);
void *test14b = (void*)test14a; // Make sure test14a didn't get skipped.

// rdar://problem/8358508
long struct X { int x; } test15(void); // expected-error {{'long struct' is invalid}}

void test16(i) int i j; { } // expected-error {{expected ';' at end of declaration}} \
                            // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}
void test17(i, j) int i, j k; { } // expected-error {{expected ';' at end of declaration}} \
                                  // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}
void knrNoSemi(i) int i { } // expected-error {{expected ';' at end of declaration}} \
                            // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}

// PR12595
void test18(void) {
  int x = 4+(5-12));  // expected-error {{extraneous ')' before ';'}}
}

enum E1 { e1 }: // expected-error {{expected ';'}}
struct EnumBitfield { // expected-warning {{struct without named members is a GNU extension}}
  enum E2 { e2 } : 4; // ok
  struct S { int n; }: // expected-error {{expected ';'}}
                       // expected-warning@-1 {{declaration does not declare anything}}

};

// PR10982
enum E11 {
  A1 = 1,
};

enum E12 {
  ,  // expected-error{{expected identifier}}
  A2
};
void func_E12(enum E12 *p) { *p = A2; }

enum E13 {
  1D,  // expected-error{{expected identifier}}
  A3
};
void func_E13(enum E13 *p) { *p = A3; }

enum E14 {
  A4 12,  // expected-error{{expected '= constant-expression' or end of enumerator definition}}
  A4a
};
void func_E14(enum E14 *p) { *p = A4a; }

enum E15 {
  A5=12 4,  // expected-error{{expected '}' or ','}}
  A5a
};
void func_E15(enum E15 *p) { *p = A5a; }

enum E16 {
  A6;  // expected-error{{expected '= constant-expression' or end of enumerator definition}}
  A6a
};

int PR20634 = sizeof(struct { int n; } [5]);
