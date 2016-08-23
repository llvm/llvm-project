// RUN: %clang_cc1 -Wno-array-bounds -analyze -analyzer-checker=core,alpha.security.ArrayBoundV2,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

// Tests doing an out-of-bounds access after the end of an array using:
// - constant integer index
// - constant integer size for buffer
void test1(int x) {
  int buf[100];
  buf[100] = 1; // expected-warning{{Out of bound memory access}}
}

void test1_ok(int x) {
  int buf[100];
  buf[99] = 1; // no-warning
}

const char test1_strings_underrun(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[-1]; // expected-warning{{Out of bound memory access}}
}

const char test1_strings_overrun(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[1000];  // expected-warning{{Out of bound memory access}}
}

const char test1_strings_ok(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[5]; // no-warning
}

// Tests doing an out-of-bounds access after the end of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test1_ptr(int x) {
  int buf[100];
  int *p = buf;
  p[101] = 1; // expected-warning{{Out of bound memory access}}
}

void test1_ptr_ok(int x) {
  int buf[100];
  int *p = buf;
  p[99] = 1; // no-warning
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test1_ptr_arith(int x) {
  int buf[100];
  int *p = buf;
  p = p + 100;
  p[0] = 1; // expected-warning{{Out of bound memory access}}
}

void test1_ptr_arith_ok(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[0] = 1; // no-warning
}

void test1_ptr_arith_bad(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[1] = 1; // expected-warning{{Out of bound memory access}}
}

void test1_ptr_arith_ok2(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[-1] = 1; // no-warning
}

// Tests doing an out-of-bounds access before the start of an array using:
// - constant integer index
// - constant integer size for buffer
void test2(int x) {
  int buf[100];
  buf[-1] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test2_ptr(int x) {
  int buf[100];
  int *p = buf;
  p[-1] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test2_ptr_arith(int x) {
  int buf[100];
  int *p = buf;
  --p;
  p[0] = 1; // expected-warning {{Out of bound memory access (accessed memory precedes memory block)}}
}

// Tests doing an out-of-bounds access before the start of a multi-dimensional
// array using:
// - constant integer indices
// - constant integer sizes for the array
void test2_multi(int x) {
  int buf[100][100];
  buf[0][-1] = 1; // expected-warning{{Out of bound memory access}}
}

// Tests doing an out-of-bounds access before the start of a multi-dimensional
// array using:
// - constant integer indices
// - constant integer sizes for the array
void test2_multi_b(int x) {
  int buf[100][100];
  buf[-1][0] = 1; // expected-warning{{Out of bound memory access}}
}

void test2_multi_ok(int x) {
  int buf[100][100];
  buf[0][0] = 1; // no-warning
}

void test3(int x) {
  int buf[100];
  if (x < 0)
    buf[x] = 1; // expected-warning{{Out of bound memory access}}
}

void test4(int x) {
  int buf[100];
  if (x > 99)
    buf[x] = 1; // expected-warning{{Out of bound memory access}}
}

void test_assume_after_access(unsigned long x) {
  int buf[100];
  buf[x] = 1;
  clang_analyzer_eval(x <= 99); // expected-warning{{TRUE}}
}

// Don't warn when indexing below the start of a symbolic region's whose
// base extent we don't know.
int *get_symbolic();
void test_index_below_symboloc() {
  int *buf = get_symbolic();
  buf[-1] = 0; // no-warning;
}

void test_incomplete_struct() {
  extern struct incomplete incomplete;
  int *p = (int *)&incomplete;
  p[1] = 42; // no-warning
}

void test_extern_void() {
  extern void v;
  int *p = (int *)&v;
  p[1] = 42; // no-warning
}

void test_assume_after_access2(unsigned long x) {
  char buf[100];
  buf[x] = 1;
  clang_analyzer_eval(x <= 99); // expected-warning{{TRUE}}
}

