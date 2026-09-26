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
// A double free through a cleanup function: the first free happens in the
// caller, so the report depends on the modeled cleanup call at the scope
// exit (without the cleanup attribute there would be no second free).
//===----------------------------------------------------------------------===//

static void double_free_cleanup(char **p) {
  free(*p); // expected-warning {{Attempt to release already released memory}}
}

void double_free_via_cleanup(void) {
  char *p __attribute__((cleanup(double_free_cleanup)));
  p = malloc(10);
  free(p); // First free: the cleanup function releases the same pointer again.
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

//===----------------------------------------------------------------------===//
// Scope-exit shapes: goto, cleanup ordering and nesting.
//===----------------------------------------------------------------------===//

// The cleanup runs on every exit from the scope, including jumps.

static void goto_cleanup(int *p) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  (void)p;
}

void goto_out_of_block_scope(void) {
  {
    int x __attribute__((cleanup(goto_cleanup)));
    x = 1;
    goto out;
  }
out:;
}

static void goto_cleanup_at_function_scope(int *p) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  (void)p;
}

void goto_at_function_scope(void) {
  int x __attribute__((cleanup(goto_cleanup_at_function_scope)));
  x = 1;
  goto out;
out:;
}

// Two cleanup handlers in the same scope run in reverse declaration order,
// as in GCC.

static int order_probe_global;

static void order_probe(int *p) {
  clang_analyzer_dump_int(order_probe_global); // expected-warning {{2 S32b}}
  (void)p;
}

static void order_side_effect(int *p) {
  order_probe_global = 2;
  (void)p;
}

void cleanup_runs_in_reverse_declaration_order(void) {
  int x __attribute__((cleanup(order_probe)));
  int y __attribute__((cleanup(order_side_effect)));
  x = 1;
  y = 2;
} // order_side_effect (declared last) runs first, so the dump above prints 2.

// A cleanup handler can declare cleanup-attributed variables of its own:
// the nested cleanup runs when the inlined handler exits.

static void nested_cleanup(int *p) {
  clang_analyzer_dump_int(*p); // expected-warning {{3 S32b}}
}

static void nested_handler(int *p) {
  int z __attribute__((cleanup(nested_cleanup)));
  z = 3;
  (void)p;
}

void cleanup_nested_in_cleanup(void) {
  int x __attribute__((cleanup(nested_handler)));
  x = 42;
}

//===----------------------------------------------------------------------===//
// A cleanup-annotated variable in a function that the analyzer inlines: the
// cleanup call is processed within the inlined stack frame.
//===----------------------------------------------------------------------===//

static void inlined_function_cleanup(int *p) {
  clang_analyzer_dump_int(*p); // expected-warning {{42 S32b}}
}

static void inlined_function_with_cleanup(void) {
  int x __attribute__((cleanup(inlined_function_cleanup)));
  x = 42;
}

void cleanup_in_inlined_function(void) {
  inlined_function_with_cleanup();
}
