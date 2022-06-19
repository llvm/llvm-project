// RUN: %clang_analyze_cc1 -verify %s \
// RUN: -analyzer-checker=core,alpha.unix.cstring


// This file is generally for the alpha.unix.cstring.UninitializedRead Checker, the reason for putting it into
// the separate file because the checker is break the some existing test cases in bstring.c file , so we don't 
// wanna mess up with some existing test case so it's better to create separate file for it, this file also include 
// the broken test for the reference in future about the broken tests.


typedef typeof(sizeof(int)) size_t;

void clang_analyzer_eval(int);

void *memcpy(void *restrict s1, const void *restrict s2, size_t n);

void top(char *dst) {
  char buf[10];
  memcpy(dst, buf, 10); // expected-warning{{Bytes string function accesses uninitialized/garbage values}}
  (void)buf;
}

//===----------------------------------------------------------------------===
// mempcpy()
//===----------------------------------------------------------------------===

void *mempcpy(void *restrict s1, const void *restrict s2, size_t n);

void mempcpy14() {
  int src[] = {1, 2, 3, 4};
  int dst[5] = {0};
  int *p;

  p = mempcpy(dst, src, 4 * sizeof(int)); // expected-warning{{Bytes string function accesses uninitialized/garbage values}}
   // FIXME: This behaviour is actually surprising and needs to be fixed, 
   // mempcpy seems to consider the very last byte of the src buffer uninitialized
   // and returning undef unfortunately. It should have returned unknown or a conjured value instead.

  clang_analyzer_eval(p == &dst[4]); // no-warning (above is fatal)
}

struct st {
  int i;
  int j;
};


void mempcpy15() {
  struct st s1 = {0};
  struct st s2;
  struct st *p1;
  struct st *p2;

  p1 = (&s2) + 1;
  p2 = mempcpy(&s2, &s1, sizeof(struct st)); // expected-warning{{Bytes string function accesses uninitialized/garbage values}}
  // FIXME: It seems same as mempcpy14() case.
  
  clang_analyzer_eval(p1 == p2); // no-warning (above is fatal)
}
