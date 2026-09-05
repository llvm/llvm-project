// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,deadcode.DeadStores,debug.ExprInspection -verify %s

// Test modeling of GCC's __attribute__((cleanup(f))): the implicit f(&var)
// call at scope exit is evaluated as an implicit call, inlined when a
// definition is available and conservatively evaluated otherwise.

#include "Inputs/system-header-simulator-for-malloc.h"

void clang_analyzer_dump_int(int);
void clang_analyzer_dump_ptr(void *);
void clang_analyzer_numTimesReached(void);
void clang_analyzer_warnIfReached(void);

//===----------------------------------------------------------------------===//
// The analysis continues past a scope exit with a cleanup-attributed variable.
//===----------------------------------------------------------------------===//

static void noop_cleanup(int *p) { (void)p; }

void path_continues_after_scope(void) {
  {
    int x __attribute__((cleanup(noop_cleanup)));
    x = 42; // no dead-store warning: the value is read by the cleanup call.
  }
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

//===----------------------------------------------------------------------===//
// An inlined cleanup observes the address of the variable and the value last
// stored to it.
//===----------------------------------------------------------------------===//

static void dump_cleanup(int *p) {
  clang_analyzer_dump_ptr(p);  // expected-warning {{&x}}
  clang_analyzer_dump_int(*p); // expected-warning {{42 S32b}}
}

void inlined_cleanup_observes_value(void) {
  int x __attribute__((cleanup(dump_cleanup)));
  x = 42;
}

//===----------------------------------------------------------------------===//
// A declared-only cleanup is conservatively evaluated: the argument escapes
// and no leak is reported for memory the unknown cleanup may have released.
//===----------------------------------------------------------------------===//

void declared_only_cleanup(void *p);

void declared_only_cleanup_escapes(void) {
  void *p __attribute__((cleanup(declared_only_cleanup)));
  p = malloc(10);
} // no leak: the pointer escapes into the conservatively evaluated call.

//===----------------------------------------------------------------------===//
// An inlined cleanup that frees the pointee: no leak.
//===----------------------------------------------------------------------===//

static void free_pointer_cleanup(char **p) {
  free(*p);
}

void inlined_cleanup_frees(void) {
  char *p __attribute__((cleanup(free_pointer_cleanup)));
  p = malloc(10);
} // no leak: free_pointer_cleanup(p) frees *p at the scope exit.

//===----------------------------------------------------------------------===//
// A non-releasing cleanup still leaks.
//===----------------------------------------------------------------------===//

static void non_releasing_cleanup(char **p) {
  (void)p;
}

void non_releasing_cleanup_leaks(void) {
  char *p __attribute__((cleanup(non_releasing_cleanup)));
  p = malloc(10);
} // expected-warning {{Potential leak of memory pointed to by 'p'}}

//===----------------------------------------------------------------------===//
// A double free through a cleanup function is anchored inside the cleanup
// body.
//===----------------------------------------------------------------------===//

static void double_free_cleanup(char **p) {
  free(*p);
  free(*p); // expected-warning {{Attempt to release already released memory}}
}

void double_free_via_cleanup(void) {
  char *p __attribute__((cleanup(double_free_cleanup)));
  p = malloc(10);
}

//===----------------------------------------------------------------------===//
// Directly naming a library function is conservatively evaluated: no crash
// and no leak for the escaped memory.
//===----------------------------------------------------------------------===//

void direct_free_cleanup(void) {
  // The emitted call is free(&p) and the compiler itself warns about it at
  // the declaration; the analyzer evaluates the call conservatively and
  // stays silent (no leak for the escaped pointee).
  void *p __attribute__((cleanup(free))); // expected-warning {{attempt to call free on non-heap object 'p'}}
  p = malloc(10);
}

//===----------------------------------------------------------------------===//
// Struct, loop and early-return shapes.
//===----------------------------------------------------------------------===//

struct Wrapped {
  char *p;
};

static void struct_cleanup(struct Wrapped *w) {
  free(w->p);
}

void struct_shape(void) {
  struct Wrapped w __attribute__((cleanup(struct_cleanup)));
  w.p = malloc(10);
} // no leak: struct_cleanup(w) frees w->p at the scope exit.

static void loop_cleanup(int *p) {
  clang_analyzer_numTimesReached(); // expected-warning {{4}}
  (void)p;
}

int loop_shape(void) {
  int sum = 0;
  for (int i = 0; i < 10; ++i) {
    int x __attribute__((cleanup(loop_cleanup)));
    x = i;
    sum += x;
  }
  return sum;
}

static void early_return_cleanup(char **p) {
  free(*p);
}

int early_return_shape(void) {
  char *p __attribute__((cleanup(early_return_cleanup)));
  p = malloc(10);
  if (!p)
    return 1;
  return 0;
} // no leak on either path: the cleanup frees *p at the return.
