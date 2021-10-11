// RUN: %clang_analyze_cc1 -std=c++20 \
// RUN:  -analyzer-checker=core,unix,cplusplus,debug.ExprInspection \
// RUN:  -triple x86_64-unknown-linux-gnu \
// RUN:  -verify %s

#include "Inputs/system-header-simulator-cxx.h"

typedef __SIZE_TYPE__ size_t;
void *malloc(size_t);
void *alloca(size_t);
void *realloc(void *ptr, size_t size);
void *calloc(size_t number, size_t size);
void free(void *);

struct S {
  int f;
};

void clang_analyzer_dump(int);
void clang_analyzer_dump(const void *);
void clang_analyzer_dumpExtent(int);
void clang_analyzer_dumpExtent(const void *);
void clang_analyzer_dumpElementCount(int);
void clang_analyzer_dumpElementCount(const void *);

int clang_analyzer_getExtent(void *);
void clang_analyzer_eval(bool);

void var_simple_ref() {
  int a = 13;
  clang_analyzer_dump(&a);             // expected-warning {{a}}
  clang_analyzer_dumpExtent(&a);       // expected-warning {{4 S64b}}
  clang_analyzer_dumpElementCount(&a); // expected-warning {{1 S64b}}
}

void var_simple_ptr(int *a) {
  clang_analyzer_dump(a);             // expected-warning {{SymRegion{reg_$0<int * a>}}}
  clang_analyzer_dumpExtent(a);       // expected-warning {{extent_$1{SymRegion{reg_$0<int * a>}}}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{(extent_$1{SymRegion{reg_$0<int * a>}}) / 4}}
}

void var_array() {
  int a[] = {1, 2, 3};
  clang_analyzer_dump(a);             // expected-warning {{Element{a,0 S64b,int}}}
  clang_analyzer_dumpExtent(a);       // expected-warning {{12 S64b}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{3 S64b}}
}

void string() {
  clang_analyzer_dump("foo");             // expected-warning {{Element{"foo",0 S64b,char}}}
  clang_analyzer_dumpExtent("foo");       // expected-warning {{4 S64b}}
  clang_analyzer_dumpElementCount("foo"); // expected-warning {{4 S64b}}
}

void struct_simple_ptr(S *a) {
  clang_analyzer_dump(a);             // expected-warning {{SymRegion{reg_$0<S * a>}}}
  clang_analyzer_dumpExtent(a);       // expected-warning {{extent_$1{SymRegion{reg_$0<S * a>}}}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{(extent_$1{SymRegion{reg_$0<S * a>}}) / 4}}
}

void field_ref(S a) {
  clang_analyzer_dump(&a.f);             // expected-warning {{a.f}}
  clang_analyzer_dumpExtent(&a.f);       // expected-warning {{4 S64b}}
  clang_analyzer_dumpElementCount(&a.f); // expected-warning {{1 S64b}}
}

void field_ptr(S *a) {
  clang_analyzer_dump(&a->f);             // expected-warning {{SymRegion{reg_$0<S * a>}.f}}
  clang_analyzer_dumpExtent(&a->f);       // expected-warning {{4 S64b}}
  clang_analyzer_dumpElementCount(&a->f); // expected-warning {{1 S64b}}
}

void symbolic_array() {
  int *a = new int[3];
  clang_analyzer_dump(a);             // expected-warning {{Element{HeapSymRegion{conj}}
  clang_analyzer_dumpExtent(a);       // expected-warning {{12 S64b}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{3 S64b}}
  delete[] a;
}

void symbolic_placement_new() {
  char *buf = new char[sizeof(int) * 3];
  int *a = new (buf) int(12);
  clang_analyzer_dump(a);             // expected-warning {{Element{HeapSymRegion{conj}}
  clang_analyzer_dumpExtent(a);       // expected-warning {{12 S64b}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{3 S64b}}
  delete[] buf;
}

void symbolic_malloc() {
  int *a = (int *)malloc(12);
  clang_analyzer_dump(a);             // expected-warning {{Element{HeapSymRegion{conj}}
  clang_analyzer_dumpExtent(a);       // expected-warning {{12 U64b}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{3 S64b}}
  free(a);
}

void symbolic_alloca() {
  int *a = (int *)alloca(12);
  clang_analyzer_dump(a);             // expected-warning {{Element{HeapSymRegion{conj}}
  clang_analyzer_dumpExtent(a);       // expected-warning {{12 U64b}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{3 S64b}}
}

void symbolic_complex() {
  int *a = (int *)malloc(4);
  clang_analyzer_dumpExtent(a);       // expected-warning {{4 U64b}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{1 S64b}}

  int *b = (int *)realloc(a, sizeof(int) * 2);
  clang_analyzer_dumpExtent(b);       // expected-warning {{8 U64b}}
  clang_analyzer_dumpElementCount(b); // expected-warning {{2 S64b}}
  free(b);

  int *c = (int *)calloc(3, 4);
  clang_analyzer_dumpExtent(c);       // expected-warning {{12 U64b}}
  clang_analyzer_dumpElementCount(c); // expected-warning {{3 S64b}}
  free(c);
}

void signedness_equality() {
  char *a = new char[sizeof(char) * 13];
  char *b = (char *)malloc(13);

  clang_analyzer_dump(clang_analyzer_getExtent(a)); // expected-warning {{13 S64b}}
  clang_analyzer_dump(clang_analyzer_getExtent(b)); // expected-warning {{13 U64b}}
  clang_analyzer_eval(clang_analyzer_getExtent(a) ==
                      clang_analyzer_getExtent(b));
  // expected-warning@-2 {{TRUE}}

  delete[] a;
  free(b);
}

void default_new_aligned() {
  struct alignas(32) S {};

  S *a = new S[10];

  clang_analyzer_dump(a);             // expected-warning {{Element{HeapSymRegion{conj}}
  clang_analyzer_dumpExtent(a);       // expected-warning {{320 S64b}}
  clang_analyzer_dumpElementCount(a); // expected-warning {{10 S64b}}

  delete[] a;
}

void *operator new[](std::size_t, std::align_val_t, bool hack) throw();

void user_defined_new() {
  int *a = new (std::align_val_t(32), true) int[10];

  clang_analyzer_dump(a);             // expected-warning {{Element{SymRegion{conj}}
  clang_analyzer_dumpExtent(a);       // expected-warning-re {{{{^extent_\$[0-9]\{SymRegion{conj}}}}
  clang_analyzer_dumpElementCount(a); // expected-warning-re {{{{^\(extent_\$[0-9]\{SymRegion{conj.*\) / 4}}}}

  operator delete[](a, std::align_val_t(32));
}
