// RUN: %clang_cc1 -fsyntax-only -verify %s -Wincompatible-pointer-types

int var __attribute__((overloadable)); // expected-error{{'overloadable' attribute only applies to functions}}
void params(void) __attribute__((overloadable(12))); // expected-error {{'overloadable' attribute takes no arguments}}

int *f(int) __attribute__((overloadable)); // expected-note 2{{previous overload of function is here}}
float *f(float); // expected-error{{overloaded function 'f' must have the 'overloadable' attribute}}
int *f(int); // expected-error{{redeclaration of 'f' must have the 'overloadable' attribute}} \
             // expected-note{{previous declaration is here}}
double *f(double) __attribute__((overloadable)); // okay, new

void test_f(int iv, float fv, double dv) {
  int *ip = f(iv);
  float *fp = f(fv);
  double *dp = f(dv);
}

int *accept_funcptr(int (*)()) __attribute__((overloadable)); //         \
  // expected-note{{candidate function}}
float *accept_funcptr(int (*)(int, double)) __attribute__((overloadable)); //  \
  // expected-note{{candidate function}}

void test_funcptr(int (*f1)(int, double),
                  int (*f2)(int, float)) {
  float *fp = accept_funcptr(f1);
  accept_funcptr(f2); // expected-error{{no matching function for call to 'accept_funcptr'}}
}

struct X { int x; float y; };
struct Y { int x; float y; };
int* accept_struct(struct X x) __attribute__((__overloadable__));
float* accept_struct(struct Y y) __attribute__((overloadable));

void test_struct(struct X x, struct Y y) {
  int *ip = accept_struct(x);
  float *fp = accept_struct(y);
}

double *f(int) __attribute__((overloadable)); // expected-error{{conflicting types for 'f'}}

double promote(float) __attribute__((__overloadable__)); // expected-note {{candidate}}
double promote(double) __attribute__((__overloadable__)); // expected-note {{candidate}}
long double promote(long double) __attribute__((__overloadable__)); // expected-note {{candidate}}

void promote(...) __attribute__((__overloadable__, __unavailable__)); // \
    // expected-note{{candidate function}}

void test_promote(short* sp) {
  promote(1.0);
  promote(sp); // expected-error{{call to unavailable function 'promote'}}
}

// PR6600
typedef double Double;
typedef Double DoubleVec __attribute__((vector_size(16)));
typedef int Int;
typedef Int IntVec __attribute__((vector_size(16)));
double magnitude(DoubleVec) __attribute__((__overloadable__));
double magnitude(IntVec) __attribute__((__overloadable__));
double test_p6600(DoubleVec d) {
  return magnitude(d) * magnitude(d);
}

// PR7738
extern int __attribute__((overloadable)) f0(); // expected-error{{'overloadable' function 'f0' must have a prototype}}
typedef int f1_type();
f1_type __attribute__((overloadable)) f1; // expected-error{{'overloadable' function 'f1' must have a prototype}}

void test() {
  f0();
  f1();
}

void before_local_1(int) __attribute__((overloadable)); // expected-note {{here}}
void before_local_2(int); // expected-note {{here}}
void before_local_3(int) __attribute__((overloadable));
void local() {
  void before_local_1(char); // expected-error {{must have the 'overloadable' attribute}}
  void before_local_2(char) __attribute__((overloadable)); // expected-error {{conflicting types}}
  void before_local_3(char) __attribute__((overloadable));
  void after_local_1(char); // expected-note {{here}}
  void after_local_2(char) __attribute__((overloadable)); // expected-note {{here}}
  void after_local_3(char) __attribute__((overloadable));
}
void after_local_1(int) __attribute__((overloadable)); // expected-error {{conflicting types}}
void after_local_2(int); // expected-error {{must have the 'overloadable' attribute}}
void after_local_3(int) __attribute__((overloadable));

// Make sure we allow C-specific conversions in C.
void conversions() {
  void foo(char *c) __attribute__((overloadable));
  void foo(char *c) __attribute__((overloadable, enable_if(c, "nope.jpg")));

  void *ptr;
  foo(ptr);

  void multi_type(unsigned char *c) __attribute__((overloadable));
  void multi_type(signed char *c) __attribute__((overloadable));
  unsigned char *c;
  multi_type(c);
}

// Ensure that we allow C-specific type conversions in C
void fn_type_conversions() {
  void foo(void *c) __attribute__((overloadable));
  void foo(char *c) __attribute__((overloadable));
  void (*ptr1)(void *) = &foo;
  void (*ptr2)(char *) = &foo;
  void (*ambiguous)(int *) = &foo; // expected-error{{initializing 'void (*)(int *)' with an expression of incompatible type '<overloaded function type>'}} expected-note@105{{candidate function}} expected-note@106{{candidate function}}
  void *vp_ambiguous = &foo; // expected-error{{initializing 'void *' with an expression of incompatible type '<overloaded function type>'}} expected-note@105{{candidate function}} expected-note@106{{candidate function}}

  void (*specific1)(int *) = (void (*)(void *))&foo; // expected-warning{{incompatible pointer types initializing 'void (*)(int *)' with an expression of type 'void (*)(void *)'}}
  void *specific2 = (void (*)(void *))&foo;

  void disabled(void *c) __attribute__((overloadable, enable_if(0, "")));
  void disabled(int *c) __attribute__((overloadable, enable_if(c, "")));
  void disabled(char *c) __attribute__((overloadable, enable_if(1, "The function name lies.")));
  // To be clear, these should all point to the last overload of 'disabled'
  void (*dptr1)(char *c) = &disabled;
  void (*dptr2)(void *c) = &disabled; // expected-warning{{incompatible pointer types initializing 'void (*)(void *)' with an expression of type '<overloaded function type>'}} expected-note@115{{candidate function made ineligible by enable_if}} expected-note@116{{candidate function made ineligible by enable_if}} expected-note@117{{candidate function has type mismatch at 1st parameter (expected 'void *' but has 'char *')}}
  void (*dptr3)(int *c) = &disabled; // expected-warning{{incompatible pointer types initializing 'void (*)(int *)' with an expression of type '<overloaded function type>'}} expected-note@115{{candidate function made ineligible by enable_if}} expected-note@116{{candidate function made ineligible by enable_if}} expected-note@117{{candidate function has type mismatch at 1st parameter (expected 'int *' but has 'char *')}}

  void *specific_disabled = &disabled;
}
