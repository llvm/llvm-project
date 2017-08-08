// RUN: %clang_analyze_cc1 -std=c++11 -Wno-array-bounds -analyzer-checker=unix,core,alpha.security.ArrayBoundV2 -verify %s

// Tests doing an out-of-bounds access after the end of an array using:
// - constant integer index
// - constant integer size for buffer
void test1(int x) {
  int *buf = new int[100];
  buf[100] = 1; // expected-warning{{Out of bound memory access}}
}

void test1_ok(int x) {
  int *buf = new int[100];
  buf[99] = 1; // no-warning
}

// Tests doing an out-of-bounds access after the end of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test1_ptr(int x) {
  int *buf = new int[100];
  int *p = buf;
  p[101] = 1; // expected-warning{{Out of bound memory access}}
}

void test1_ptr_ok(int x) {
  int *buf = new int[100];
  int *p = buf;
  p[99] = 1; // no-warning
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test1_ptr_arith(int x) {
  int *buf = new int[100];
  int *p = buf;
  p = p + 100;
  p[0] = 1; // expected-warning{{Out of bound memory access}}
}

void test1_ptr_arith_ok(int x) {
  int *buf = new int[100];
  int *p = buf;
  p = p + 99;
  p[0] = 1; // no-warning
}

void test1_ptr_arith_bad(int x) {
  int *buf = new int[100];
  int *p = buf;
  p = p + 99;
  p[1] = 1; // expected-warning{{Out of bound memory access}}
}

void test1_ptr_arith_ok2(int x) {
  int *buf = new int[100];
  int *p = buf;
  p = p + 99;
  p[-1] = 1; // no-warning
}

// Tests doing an out-of-bounds access before the start of an array using:
// - constant integer index
// - constant integer size for buffer
void test2(int x) {
  int *buf = new int[100];
  buf[-1] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test2_ptr(int x) {
  int *buf = new int[100];
  int *p = buf;
  p[-1] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test2_ptr_arith(int x) {
  int *buf = new int[100];
  int *p = buf;
  --p;
  p[0] = 1; // expected-warning {{Out of bound memory access (accessed memory precedes memory block)}}
}

// Tests under-indexing
// of a multi-dimensional array
void test2_multi(int x) {
  auto buf = new int[100][100];
  buf[0][-1] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests under-indexing
// of a multi-dimensional array
void test2_multi_b(int x) {
  auto buf = new int[100][100];
  buf[-1][0] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests over-indexing
// of a multi-dimensional array
void test2_multi_c(int x) {
  auto buf = new int[100][100];
  buf[100][0] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests over-indexing
// of a multi-dimensional array
void test2_multi_2(int x) {
  auto buf = new int[100][100];
  buf[99][100] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests normal access of
// a multi-dimensional array
void test2_multi_ok(int x) {
  auto buf = new int[100][100];
  buf[0][0] = 1; // no-warning
}

// Tests over-indexing using different types
// array
void test_diff_types(int x) {
  int *buf = new int[10]; //10*sizeof(int) Bytes allocated
  char *cptr = (char *)buf;
  cptr[sizeof(int) * 9] = 1;  // no-warning
  cptr[sizeof(int) * 10] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests over-indexing
//if the allocated area is non-array
void test_non_array(int x) {
  int *ip = new int;
  ip[0] = 1; // no-warning
  ip[1] = 2; // expected-warning{{Out of bound memory access}}
}

//Tests over-indexing
//if the allocated area size is a runtime parameter
void test_dynamic_size(int s) {
  int *buf = new int[s];
  buf[0] = 1; // no-warning
}
//Tests complex arithmetic
//in new expression
void test_dynamic_size2(unsigned m,unsigned n){
  unsigned *U = nullptr;
  U = new unsigned[m + n + 1];
}
