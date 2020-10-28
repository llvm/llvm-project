/* RUN: %clang_cc1 %s -std=c89 -pedantic -fsyntax-only -verify -Wimplicit-function-declaration
 */
void test1() {
  {
    int i;
    i = i + 1;
    int j;          /* expected-warning {{mixing declarations and code}} */
  }
  {
    __extension__ int i;
    i = i + 1;
    int j;          /* expected-warning {{mixing declarations and code}} */
  }
  {
    int i;
    i = i + 1;
    __extension__ int j; /* expected-warning {{mixing declarations and code}} */
  }
}

long long test2;   /* expected-warning {{extension}} */


void test3(int i) {
  int A[i];        /* expected-warning {{variable length array}} */
}

int test4 = 0LL;   /* expected-warning {{long long}} */

/* PR1999 */
void test5(register);

/* PR2041 */
int *restrict;
int *__restrict;  /* expected-error {{expected identifier}} */


/* Implicit int, always ok */
test6() { return 0; }

/* PR2012 */
test7;  /* expected-warning {{declaration specifier missing, defaulting to 'int'}} */

void test8(int, x);  /* expected-warning {{declaration specifier missing, defaulting to 'int'}} */

typedef int sometype;
int a(sometype, y) {return 0;}  /* expected-warning {{declaration specifier missing, defaulting to 'int'}} \
                                   expected-warning {{omitting the parameter name in a function definition is a C2x extension}}*/




void bar (void *); 
void f11 (z)       /* expected-error {{may not have 'void' type}} */
void z; 
{ bar (&z); }

typedef void T;
void foo(T); /* typedef for void is allowed */

void foo(void) {}

/* PR2759 */
void test10 (int x[*]); /* expected-warning {{variable length arrays are a C99 feature}} */
void test11 (int x[static 4]); /* expected-warning {{static array size is a C99 feature}} */

void test12 (int x[const 4]) { /* expected-warning {{qualifier in array size is a C99 feature}} */
  int Y[x[1]]; /* expected-warning {{variable length arrays are a C99 feature}} expected-note {{parameter 'x'}} */
}

/* PR4074 */
struct test13 {
  int X[23];
} test13a();

void test13b() {
  int a = test13a().X[1]; /* expected-warning {{ISO C90 does not allow subscripting non-lvalue array}} */
  int b = 1[test13a().X]; /* expected-warning {{ISO C90 does not allow subscripting non-lvalue array}} */
}

/* Make sure we allow *test14 as a "function designator" */
int test14() { return (&*test14)(); }

int test15[5] = { [2] = 1 }; /* expected-warning {{designated initializers are a C99 feature}} */

extern int printf(__const char *__restrict __format, ...);

/* Warn, but don't suggest typo correction. */
void test16() {
  printg("Hello, world!\n"); /* expected-warning {{implicit declaration of function 'printg'}} */
}

struct x { int x,y[]; }; /* expected-warning {{flexible array members are a C99 feature}} */

/* Duplicated type-qualifiers aren't allowed by C90 */
const const int c_i; /* expected-warning {{duplicate 'const' declaration specifier}} */
typedef volatile int vol_int;
volatile vol_int volvol_i; /* expected-warning {{duplicate 'volatile' declaration specifier}} */
typedef volatile vol_int volvol_int; /* expected-warning {{duplicate 'volatile' declaration specifier}} */
const int * const c;

typedef const int CI;

const CI mine1[5][5]; /* expected-warning {{duplicate 'const' declaration specifier}} */

typedef CI array_of_CI[5];
const array_of_CI mine2; /* expected-warning {{duplicate 'const' declaration specifier}} */

typedef CI *array_of_pointer_to_CI[5];
const array_of_pointer_to_CI mine3;

void main() {} /* expected-error {{'main' must return 'int'}} */

const int main() {} /* expected-error {{'main' must return 'int'}} */

long long ll1 = /* expected-warning {{'long long' is an extension when C99 mode is not enabled}} */
         -42LL; /* expected-warning {{'long long' is an extension when C99 mode is not enabled}} */
unsigned long long ull1 = /* expected-warning {{'long long' is an extension when C99 mode is not enabled}} */
                   42ULL; /* expected-warning {{'long long' is an extension when C99 mode is not enabled}} */

struct Test17 { int a; };
struct Test17 test17_aux(void);

void test17(int v, int w) {
  int a[2] = { v, w }; /* expected-warning {{initializer for aggregate is not a compile-time constant}} */
  struct Test17 t0 = { v }; /* expected-warning {{initializer for aggregate is not a compile-time constant}} */
  struct Test17 t1 = test17_aux(); /* this is allowed */
}

