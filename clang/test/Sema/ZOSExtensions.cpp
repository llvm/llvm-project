// RUN: %clang_cc1 -triple s390x-ibm-zos %s -fsyntax-only -fzos-extensions -verify
// RUN: %clang_cc1 -triple s390x-ibm-zos %s -fsyntax-only -verify

struct A {
  int a;
  short b;
  float q;
  double z;
};

union B {
  int a;
  short b;
  float q;
  double z;
};

class C {
  int a;
  short b;
  float q;
  double z;
};

// ************************
// INCORRECT DECLARATION
// ************************
int * __ptr64 p64; // expected-error {{expected ';' after top level declarator}}
int *wrong_var3 __ptr32; // expected-error {{expected ';' after top level declarator}} expected-warning {{declaration does not declare anything}}

// **************************
// INCORRECT USAGES OF PTR32
// **************************
struct D {
  int __ptr32 *a; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
};

union E {
  int __ptr32 *b; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
};

char __ptr32 *a; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
signed char __ptr32 *b; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
unsigned char __ptr32 *c; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
int __ptr32 *d; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
signed int __ptr32 *e; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
unsigned int __ptr32 *f; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
short int __ptr32 *g; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
signed short int __ptr32 *h; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
unsigned short int __ptr32 *i; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
long int __ptr32 *j; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
signed long int __ptr32 *k; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
unsigned long int __ptr32 *l; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
long long int __ptr32 *m; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
signed long long int __ptr32 *n; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
unsigned long long int __ptr32 *o; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
float __ptr32 *p;                  // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
double __ptr32 *q;                 // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
int __ptr32 **r;                   // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
int __ptr32 *__ptr32 *s;           // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
int __ptr32 *__ptr32 *__ptr32 t;   // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
int __ptr32 *__ptr32 *__ptr32 *__ptr32 u; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
int __ptr32 __ptr32 **v_i;                // expected-error {{'__ptr32' attribute only applies to pointer arguments}} expected-error {{'__ptr32' attribute only applies to pointer arguments}}
int __ptr32 __ptr32 __ptr32 w_i;          // expected-error {{'__ptr32' attribute only applies to pointer arguments}} expected-error {{'__ptr32' attribute only applies to pointer arguments}} expected-error {{'__ptr32' attribute only applies to pointer arguments}}

__ptr32 int wrong_var; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}

struct A __ptr32 *c1;                  // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
struct A __ptr32 **e1;                 // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
struct A __ptr32 *__ptr32 *f1;         // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
struct A __ptr32 *__ptr32 *__ptr32 g1; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
union B __ptr32 *d1; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
union B __ptr32 **h1; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
union B __ptr32 * __ptr32 *i1; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
union B __ptr32 * __ptr32 * __ptr32 j1; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}

C __ptr32 **k1; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
C __ptr32 * __ptr32 *l1; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
C __ptr32 * __ptr32 * __ptr32 m1; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}

struct D n1;
union E o1;

int incorrect_func() {
  int __ptr32 = 1; // expected-error {{expected unqualified-id}}
  return __ptr32; // expected-error {{expected expression}}
}

typedef int __ptr32; // expected-warning {{typedef requires a name}}
int incorrect_func2() {
  return 1;
}

typedef int __ptr32 *v; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}
int incorrect_func3() {
  v v1;
  return 0;
}

int *__ptr32 a_ptr; //expected-note {{previous definition is here}}
int *a_ptr;         // expected-error {{redefinition of 'a_ptr' with a different type: 'int *' vs 'int * __ptr32'}}

// *******************************************************
// FUNCTION OVERLOADING BETWEEN PTR32 AND REGULAR POINTERS
// *******************************************************
void func(int * __ptr32 p32) {} // expected-note {{previous definition is here}}
void func(int *p64) {}          // expected-error {{redefinition of 'func'}}

// Overloads between ptr32 and other non-pointer types are permissible
void func1(int *__ptr32 p32) {}
void func1(int p64) {}

// ******
// MISC
// ******
void func2() {
  char * __ptr32 v = ((char * __ptr32 *)1028)[0];
  char *v1 = ((char ** __ptr32 *)1028)[0][1];
}

