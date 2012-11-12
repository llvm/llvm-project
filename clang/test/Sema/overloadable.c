// RUN: %clang_cc1 -fsyntax-only -verify %s

int var __attribute__((overloadable)); // expected-error{{'overloadable' attribute can only be applied to a function}}

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
