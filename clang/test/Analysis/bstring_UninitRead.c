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
  memcpy(dst, buf, 10); // expected-warning{{The first element of the 2nd argument is undefined}}
                        // expected-note@-1{{Other elements might also be undefined}}
  (void)buf;
}

void top2(char *dst) {
  char buf[10];
  buf[0] = 'i';
  memcpy(dst, buf, 10); // expected-warning{{The last element of the 2nd argument to access (the 10th) is undefined}}
                        // expected-note@-1{{Other elements might also be undefined}}
  (void)buf;
}

void top3(char *dst) {
  char buf[10];
  buf[0] = 'i';
  memcpy(dst, buf, 1);
  (void)buf;
}

void top4(char *dst) {
  char buf[10];
  buf[0] = 'i';
  memcpy(dst, buf, 2); // expected-warning{{The last element of the 2nd argument to access (the 2nd) is undefined}}
                        // expected-note@-1{{Other elements might also be undefined}}
  (void)buf;
}

//===----------------------------------------------------------------------===
// mempcpy()
//===----------------------------------------------------------------------===

void *mempcpy(void *restrict s1, const void *restrict s2, size_t n);

void mempcpy13() {
  int src[] = {1, 2, 3, 4};
  int dst[5] = {0};
  int *p;

  p = mempcpy(dst, src, 4 * sizeof(int));
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

  p2 = mempcpy(&s2, &s1, sizeof(struct st));

  clang_analyzer_eval(p1 == p2); // no-warning (above is fatal)
}

void mempcpy16() {
  struct st s1;
  struct st s2;

  // FIXME: Maybe ask UninitializedObjectChecker whether s1 is fully
  // initialized?
  mempcpy(&s2, &s1, sizeof(struct st));
}

void initialized(int *dest) {
  int t[] = {1, 2, 3};
  memcpy(dest, t, sizeof(t));
}

// Creduced crash.

void *ga_copy_strings_from_0;
void *memmove();
void alloc();
void ga_copy_strings() {
  int i = 0;
  for (;; ++i)
    memmove(alloc, ((char **)ga_copy_strings_from_0)[i], 1);
}

