// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR3592
static void* malloc(int);
static void* malloc(int size) {
  return ((void*)0); /*do not use heap in this file*/
}

void *calloc(int, int, int); // expected-warning{{incompatible redeclaration of library function 'calloc'}} \
// expected-note{{'calloc' is a builtin with type 'void *}}

void f1(void) { 
  calloc(0, 0, 0);
}

void f2() {
  int index = 1;
}

static int index;

int f3() {
  return index << 2;
}

typedef int rindex;
