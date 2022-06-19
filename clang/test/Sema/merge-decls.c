// RUN: %clang_cc1 %s -verify -fsyntax-only -Wno-strict-prototypes

void foo(void);
void foo(void) {} 
void foo(void);
void foo(void); // expected-note {{previous declaration is here}}

void foo(int); // expected-error {{conflicting types for 'foo'}}

int funcdef()
{
 return 0;
}

int funcdef();

int funcdef2() { return 0; } // expected-note {{previous definition is here}}
int funcdef2() { return 0; } // expected-error {{redefinition of 'funcdef2'}}

// PR2502
void (*f)(void);
void (*f)() = 0;

typedef __attribute__(( ext_vector_type(2) )) int Vi2;
typedef __attribute__(( ext_vector_type(2) )) float Vf2;

Vf2 g0; // expected-note {{previous definition is here}}
Vi2 g0; // expected-error {{redefinition of 'g0'}}

_Complex int g1; // expected-note {{previous definition is here}}
_Complex float g1; // expected-error {{redefinition of 'g1'}}

// rdar://6096412
extern char i6096412[10];
extern char i6096412[];
void foo6096412(void) {
  int x = sizeof(i6096412);
}


typedef int test1_IA[];
typedef int test1_A10[10];
static test1_A10 *test1_f(void);
void test1_g(void)
{
  {
    extern test1_IA  *test1_f(void);
  }
  (void)sizeof(*test1_f());
}

typedef int test2_IA[];
typedef int test2_A10[10];

static test2_A10 *test2_f(void);
static test2_IA  *test2_f(void);

void test2_g(void)
{
  (void)sizeof(*test2_f());
}

int (*test3_f())[10];
int (*test3_f())[];
int test3_k = sizeof(*test3_f());

void test4_f(int);
void test4_f(a)
  char a;
{
  int v[sizeof(a) == 1 ? 1 : -1];
}

int test5_f(int (*)[10]);
int test5_f(int (*x)[]) {
  return sizeof(*x); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int[]'}}
}

void test6_f(int (*a)[11]);
void test6_f(a)
   int (*a)[];
{}
void test6_g() {
  int arr[10];
  test6_f(&arr); // expected-warning {{incompatible pointer types passing 'int (*)[10]' to parameter of type 'int (*)[11]}}
}

void test7_f(int (*)[10]);
void test7_f(int (*)[]); // expected-note {{passing argument to parameter here}}
void test7_g() {
  int x[5];
  test7_f(&x); // expected-warning {{incompatible pointer types passing 'int (*)[5]' to parameter of type 'int (*)[10]}}
}
