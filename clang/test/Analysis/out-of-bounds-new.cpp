// RUN: %clang_analyze_cc1 -std=c++11 -Wno-array-bounds -verify %s \
// RUN:   -analyzer-checker=unix,core,security.ArrayBound,debug.ExprInspection

void clang_analyzer_eval(bool);

// Tests doing an out-of-bounds access after the end of an array using:
// - constant integer index
// - constant integer size for buffer
void test1(int x) {
  int *buf = new int[100];
  buf[100] = 1; // expected-warning{{Out of bound access to memory}}
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
  p[101] = 1; // expected-warning{{Out of bound access to memory}}
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
  p[0] = 1; // expected-warning{{Out of bound access to memory}}
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
  p[1] = 1; // expected-warning{{Out of bound access to memory}}
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
  buf[-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test2_ptr(int x) {
  int *buf = new int[100];
  int *p = buf;
  p[-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test2_ptr_arith(int x) {
  int *buf = new int[100];
  int *p = buf;
  --p;
  p[0] = 1; // expected-warning {{Out of bound access to memory preceding}}
}

// Tests under-indexing
// of a multi-dimensional array
void test2_multi(int x) {
  auto buf = new int[100][100];
  buf[0][-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests under-indexing
// of a multi-dimensional array
void test2_multi_b(int x) {
  auto buf = new int[100][100];
  buf[-1][0] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests over-indexing
// of a multi-dimensional array
void test2_multi_c(int x) {
  auto buf = new int[100][100];
  buf[100][0] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests over-indexing
// of a multi-dimensional array
void test2_multi_2(int x) {
  auto buf = new int[100][100];
  buf[99][100] = 1; // expected-warning{{Out of bound access to memory}}
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
  cptr[sizeof(int) * 10] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests over-indexing
//if the allocated area is non-array
void test_non_array(int x) {
  int *ip = new int;
  ip[0] = 1; // no-warning
  ip[1] = 2; // expected-warning{{Out of bound access to memory}}
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

//Test creating invalid references, which break the invariant that a reference
//is always holding a value, and could lead to nasty runtime errors.
//(This is not related to operator new, but placed in this file because the
//other test files are not C++.)
int array[10] = {0};

void test_after_the_end_reference() {
  int &ref = array[10]; // expected-warning{{Out of bound access to memory}}
}

void test_after_after_the_end_reference() {
  int &ref = array[11]; // expected-warning{{Out of bound access to memory}}
}

int test_reference_that_might_be_after_the_end(int idx) {
  // This TC produces no warning because separate analysis of (idx == 10) is
  // only introduced _after_ the creation of the reference ref.
  if (idx < 0 || idx > 10)
    return -2;
  int &ref = array[idx];
  if (idx == 10)
    return -1;
  return ref;
}

// From: https://github.com/llvm/llvm-project/issues/100762
extern int arrOf10[10];
void using_builtin(int x) {
  __builtin_assume(x > 101); // CallExpr
  arrOf10[x] = 404; // expected-warning {{Out of bound access to memory}}
}

void using_assume_attr(int ax) {
  [[assume(ax > 100)]]; // NullStmt with an "assume" attribute.
  arrOf10[ax] = 405; // expected-warning {{Out of bound access to memory}}
}

void using_many_assume_attr(int yx) {
  [[assume(yx > 104), assume(yx > 200), assume(yx < 300)]]; // NullStmt with an attribute
  arrOf10[yx] = 406; // expected-warning{{Out of bound access to memory}}
}


int using_builtin_assume_has_no_sideeffects(int y) {
  // We should not apply sideeffects of the argument of [[assume(...)]].
  // "y" should not get incremented;
  __builtin_assume(++y == 43); // expected-warning {{assumption is ignored because it contains (potential) side-effects}}
  clang_analyzer_eval(y == 42); // expected-warning {{FALSE}}
  return y;
}



int using_assume_attr_has_no_sideeffects(int y) {

  // We should not apply sideeffects of the argument of [[assume(...)]].
  // "y" should not get incremented;
  [[assume(++y == 43)]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}}
 
  clang_analyzer_eval(y == 42); // expected-warning {{TRUE}} expected-warning {{FALSE}} FIXME: This should be only TRUE.

  clang_analyzer_eval(y == 43); // expected-warning {{FALSE}} expected-warning {{TRUE}} FIXME: This should be only FALSE.

  return y;
}


int using_builtinassume_has_no_sideeffects(int u) {
  // We should not apply sideeffects of the argument of __builtin_assume(...)
  // "u" should not get incremented;
  __builtin_assume(++u == 43); // expected-warning {{assumption is ignored because it contains (potential) side-effects}}
 
  // FIXME: evaluate this to true
  clang_analyzer_eval(u == 42); // expected-warning {{FALSE}}  current behavior 

  // FIXME: evaluate this to false
  clang_analyzer_eval(u == 43); // expected-warning {{TRUE}}  current behavior 

  return u;
}
