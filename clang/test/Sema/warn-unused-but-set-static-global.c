// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -I %S/Inputs -verify %s

// Test that header-defined static globals don't warn.
#include "warn-unused-but-set-static-global-header.h"

#define NULL (void*)0

void *set(int size);
void func_call(void *);

static int set_unused; // expected-warning {{variable 'set_unused' set but not used}}
static int set_and_used;
static int only_used;
static int addr_taken;
extern int external_var;  // No warning (external linkage).
extern int global_var;  // No warning (not static).

void f1() {
  set_unused = 1;
  set_and_used = 2;

  int x = set_and_used;
  (void)x;

  int y = only_used;
  (void)y;

  int *p = &addr_taken;
  (void)p;

  external_var = 3;
  global_var = 4;
}

// Test across multiple functions.
static int set_used1;
static int set_used2;

static int set1; // expected-warning {{variable 'set1' set but not used}}
static int set2; // expected-warning {{variable 'set2' set but not used}}

void f2() {
  set1 = 1;
  set_used1 = 1;

  int x = set_used2;
  (void)x;
}

void f3() {
  set2 = 2;
  set_used2 = 2;

  int x = set_used1;
  (void)x;
}

static volatile int vol_set; // expected-warning {{variable 'vol_set' set but not used}}
void f4() {
  vol_set = 1;
}

// Read and use
static int compound; // expected-warning{{variable 'compound' set but not used}}
static volatile int vol_compound;
static int unary; // expected-warning{{variable 'unary' set but not used}}
static volatile int vol_unary;
void f5() {
  compound += 1;
  vol_compound += 1;
  unary++;
  vol_unary++;
}

struct S {
  int i;
};
static struct S s_set;  // expected-warning{{variable 's_set' set but not used}}
static struct S s_used;
void f6() {
  struct S t;
  s_set = t;
  t = s_used;
}

// Multiple assignments
static int multi; // expected-warning{{variable 'multi' set but not used}}
void f7() {
  multi = 1;
  multi = 2;
  multi = 3;
}

// Unused pointers
static int *unused_ptr; // expected-warning{{variable 'unused_ptr' set but not used}}
static char *str_ptr; // expected-warning{{variable 'str_ptr' set but not used}}
void f8() {
  unused_ptr = set(5);
  str_ptr = "hello";
}

// Used pointers
void a(void *);
static int *used_ptr;
static int *param_ptr;
static int *null_check_ptr;
void f9() {
  used_ptr = set(5);
  *used_ptr = 5;

  param_ptr = set(5);
  func_call(param_ptr);

  null_check_ptr = set(5);
  if (null_check_ptr == NULL) {}
}

// Function pointers (unused)
static void (*unused_func_ptr)(); // expected-warning {{variable 'unused_func_ptr' set but not used}}
void SetUnusedCallback(void (*f)()) {
  unused_func_ptr = f;
}

// Function pointers (used)
static void (*used_func_ptr)();
void SetUsedCallback(void (*f)()) {
  used_func_ptr = f;
}
void CallUsedCallback() {
  if (used_func_ptr)
    used_func_ptr();
}
