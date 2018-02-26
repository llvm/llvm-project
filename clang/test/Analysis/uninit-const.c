// RUN: %clang_analyze_cc1 -analyzer-checker=unix.Malloc,core,alpha.core.CallAndMessageUnInitRefArg,debug.ExprInspection -analyzer-output=text -verify %s

void clang_analyzer_warnIfReached();

// Passing uninitialized const data to function
#include "Inputs/system-header-simulator.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *valloc(size_t);
void free(void *);


void doStuff3(const int y){}
void doStuff2(int g){}
void doStuff_pointerToConstInt(const int *u){};
void doStuff_arrayOfConstInt(const int a[]){};

void doStuff_constPointerToConstInt              (int const * const u){};
void doStuff_constPointerToConstPointerToConstInt(int const * const * const u){};
void doStuff_pointerToConstPointerToConstInt(int const * const * u){};
void doStuff_pointerToPointerToConstInt       (int const **u){};
void doStuff_constStaticSizedArray(const int a[static 10]) {}
void doStuff_variadic(const int *u, ...){};

void f_1(void) {
  int t;
  int* tp = &t;        // expected-note {{'tp' initialized here}}
  doStuff_pointerToConstInt(tp);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                       // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

void f_1_1(void) {
  int t;
  int* tp1 = &t;
  int* tp2 = tp1;        // expected-note {{'tp2' initialized here}}
  doStuff_pointerToConstInt(tp2);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                       // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}


int *f_2_sub(int *p) {
  return p;
}

void f_2(void) {
  int t;
  int* p = f_2_sub(&t);
  int* tp = p; // expected-note {{'tp' initialized here}}
  doStuff_pointerToConstInt(tp); // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                      // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

int z;
void f_3(void) {
      doStuff_pointerToConstInt(&z);  // no warning
}

void f_4(void) {
      int x=5;
      doStuff_pointerToConstInt(&x);  // no warning
}

void f_5(void) {
  int ta[5];
  int* tp = ta;        // expected-note {{'tp' initialized here}}
  doStuff_pointerToConstInt(tp);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                       // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

void f_5_1(void) {
  int ta[5];        // expected-note {{'ta' initialized here}}
  doStuff_pointerToConstInt(ta);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                       // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

void f_6(void) {
  int ta[5] = {1,2,3,4,5};
  int* tp = ta;
  doStuff_pointerToConstInt(tp); // no-warning
}

void f_6_1(void) {
  int ta[5] = {1,2,3,4,5};
  doStuff_pointerToConstInt(ta); // no-warning
}

void f_7(void) {
      int z;        // expected-note {{'z' declared without an initial value}}
      int y=z;      // expected-warning {{Assigned value is garbage or undefined}}
                    // expected-note@-1 {{Assigned value is garbage or undefined}}
      doStuff3(y);
}

void f_8(void) {
      int g;       // expected-note {{'g' declared without an initial value}}
      doStuff2(g); // expected-warning {{1st function call argument is an uninitialized value}}
                   // expected-note@-1 {{1st function call argument is an uninitialized value}}
}

void f_9(void) {
  int  a[6];
  int const *ptau = a;             // expected-note {{'ptau' initialized here}}
  doStuff_arrayOfConstInt(ptau);    // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                                   // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

void f_10(void) {
  int  a[6];                     // expected-note {{'a' initialized here}}
  doStuff_arrayOfConstInt(a);    // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                                 // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

void f_11(void) {
  int t[10];                    //expected-note {{'t' initialized here}}
  doStuff_constStaticSizedArray(t);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                                // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

void f_12(void) {
  int t[10] = {0,1,2,3,4,5,6,7,8,9};
  doStuff_constStaticSizedArray(t);  // no-warning

}

// https://bugs.llvm.org/show_bug.cgi?id=35419
void f11_0(void) {
  int x; // expected-note {{'x' declared without an initial value}}
  x++; // expected-warning {{The expression is an uninitialized value. The computed value will also be garbage}}
       // expected-note@-1 {{The expression is an uninitialized value. The computed value will also be garbage}}
  clang_analyzer_warnIfReached(); // no-warning
}
void f11_1(void) {
  int x; // expected-note {{'x' declared without an initial value}}
  ++x; // expected-warning {{The expression is an uninitialized value. The computed value will also be garbage}}
       // expected-note@-1 {{The expression is an uninitialized value. The computed value will also be garbage}}
  clang_analyzer_warnIfReached(); // no-warning
}
void f11_2(void) {
  int x; // expected-note {{'x' declared without an initial value}}
  x--; // expected-warning {{The expression is an uninitialized value. The computed value will also be garbage}}
       // expected-note@-1 {{The expression is an uninitialized value. The computed value will also be garbage}}
  clang_analyzer_warnIfReached(); // no-warning
}
void f11_3(void) {
  int x; // expected-note {{'x' declared without an initial value}}
  --x; // expected-warning {{The expression is an uninitialized value. The computed value will also be garbage}}
       // expected-note@-1 {{The expression is an uninitialized value. The computed value will also be garbage}}
  clang_analyzer_warnIfReached(); // no-warning
}

int f_malloc_1(void) {
  int *ptr;

  ptr = (int *)malloc(sizeof(int)); // expected-note {{Value assigned to 'ptr'}}

  doStuff_pointerToConstInt(ptr); // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                       // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
  free(ptr);
  return 0;
}

int f_malloc_2(void) {
  int *ptr;

  ptr = (int *)malloc(sizeof(int));
  *ptr = 25;

  doStuff_pointerToConstInt(ptr); // no warning
  free(ptr);
  return 0;
}

// uninit pointer, uninit val
void f_variadic_unp_unv(void) {
  int t;
  int v;
  int* tp = &t;           // expected-note {{'tp' initialized here}}
  doStuff_variadic(tp,v);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                          // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}
// uninit pointer, init val
void f_variadic_unp_inv(void) {
  int t;
  int v = 3;
  int* tp = &t;           // expected-note {{'tp' initialized here}}
  doStuff_variadic(tp,v);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                          // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

// init pointer, uninit val
void f_variadic_inp_unv(void) {
  int t=5;
  int v;                  // expected-note {{'v' declared without an initial value}}
  int* tp = &t;
  doStuff_variadic(tp,v);// expected-warning {{2nd function call argument is an uninitialized value}}
                          // expected-note@-1 {{2nd function call argument is an uninitialized value}}
}

// init pointer, init val
void f_variadic_inp_inv(void) {
  int t=5;
  int v = 3;
  int* tp = &t;
  doStuff_variadic(tp,v); // no-warning
}

// init pointer, init pointer
void f_variadic_inp_inp(void) {
  int t=5;
  int u=3;
  int *vp = &u ;
  int *tp = &t;
  doStuff_variadic(tp,vp); // no-warning
}

//uninit pointer, init pointer
void f_variadic_unp_inp(void) {
  int t;
  int u=3;
  int *vp = &u ;
  int *tp = &t;             // expected-note {{'tp' initialized here}}
  doStuff_variadic(tp,vp); // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                            // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}

//init pointer, uninit pointer
void f_variadic_inp_unp(void) {
  int t=5;
  int u;
  int *vp = &u ;
  int *tp = &t;
  doStuff_variadic(tp,vp); // no-warning
}

//uninit pointer, uninit pointer
void f_variadic_unp_unp(void) {
  int t;
  int u;
  int *vp = &u ;
  int *tp = &t;             // expected-note {{'tp' initialized here}}
  doStuff_variadic(tp,vp); // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                            // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}
