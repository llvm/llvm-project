// RUN: %clang_analyze_cc1 -analyzer-checker=unix.Malloc -analyzer-inline-max-stack-depth=5 -verify %s

#include "Inputs/system-header-simulator.h"

void *malloc(size_t);
void *valloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *reallocf(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);

void exit(int) __attribute__ ((__noreturn__));
void *memcpy(void * restrict s1, const void * restrict s2, size_t n);
size_t strlen(const char *);

static void my_malloc1(void **d, size_t size) {
  *d = malloc(size);
}

static void *my_malloc2(int elevel, size_t size) {
  void     *data;
  data = malloc(size);
  if (data == 0)
    exit(0);
  return data;
}

static void my_free1(void *p) {
  free(p);
}

static void test1(void) {
  void *data = 0;
  my_malloc1(&data, 4);
} // expected-warning {{Potential leak of memory pointed to by 'data'}}

static void test11(void) {
  void *data = 0;
  my_malloc1(&data, 4);
  my_free1(data);
}

static void testUniqueingByallocationSiteInTopLevelFunction(void) {
  void *data = my_malloc2(1, 4);
  data = 0;
  int x = 5;// expected-warning {{Potential leak of memory pointed to by 'data'}}
  data = my_malloc2(1, 4);
} // expected-warning {{Potential leak of memory pointed to by 'data'}}

static void test3(void) {
  void *data = my_malloc2(1, 4);
  free(data);
  data = my_malloc2(1, 4);
  free(data);
}

int test4(void) {
  int *data = (int*)my_malloc2(1, 4);
  my_free1(data);
  data = (int *)my_malloc2(1, 4);
  my_free1(data);
  return *data; // expected-warning {{Use of memory after it is freed}}
}

void test6(void) {
  int *data = (int *)my_malloc2(1, 4);
  my_free1((int*)data);
  my_free1((int*)data); // expected-warning{{Use of memory after it is freed}}
}

// TODO: We should warn here.
void test5(void) {
  int *data;
  my_free1((int*)data);
}

static char *reshape(char *in) {
    return 0;
}

void testThatRemoveDeadBindingsRunBeforeEachCall(void) {
    char *v = malloc(12);
    v = reshape(v);
    v = reshape(v);// expected-warning {{Potential leak of memory pointed to by 'v'}}
}

// Test that we keep processing after 'return;'
void fooWithEmptyReturn(int x) {
  if (x)
    return;
  x++;
  return;
}

int uafAndCallsFooWithEmptyReturn(void) {
  int *x = (int*)malloc(12);
  free(x);
  fooWithEmptyReturn(12);
  return *x; // expected-warning {{Use of memory after it is freed}}
}
