// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,deadcode.DeadStores,debug.ExprInspection -verify %s

// Test modeling of GCC's __attribute__((cleanup(f))): the implicit f(&var)
// call at scope exit is evaluated as an implicit call, inlined when a
// definition is available and conservatively evaluated otherwise.

#include "Inputs/system-header-simulator-for-malloc.h"

void clang_analyzer_dump(const char *);
void clang_analyzer_dump_int(int);
void clang_analyzer_dump_ptr(void *);
void clang_analyzer_numTimesReached(void);
void clang_analyzer_warnIfReached(void);

// The analysis continues past a scope exit with a cleanup-attributed variable.
static void noop_cleanup(int *p) { (void)p; }

void path_continues_after_scope(void) {
  {
    int x __attribute__((cleanup(noop_cleanup)));
    x = 42; // No dead store: the cleanup call reads x.
  }
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

// An inlined cleanup observes the address of the variable and its value.
static void dump_cleanup(int *p) {
  clang_analyzer_dump_ptr(p);  // expected-warning {{&x}}
  clang_analyzer_dump_int(*p); // expected-warning {{42 S32b}}
}

void inlined_cleanup_observes_value(void) {
  int x __attribute__((cleanup(dump_cleanup)));
  x = 42;
}

// A declared-only cleanup is conservatively evaluated: the argument escapes
// and no leak is reported for memory the unknown cleanup may have released.
void declared_only_cleanup(void *p);

void declared_only_cleanup_escapes(void) {
  void *p __attribute__((cleanup(declared_only_cleanup)));
  p = malloc(10);
} // No leak: p escapes into the conservatively evaluated call.

// An inlined cleanup that frees the pointee: no leak.
static void free_pointer_cleanup(char **p) {
  free(*p);
}

void inlined_cleanup_frees(void) {
  char *p __attribute__((cleanup(free_pointer_cleanup)));
  p = malloc(10);
} // No leak: the cleanup frees *p at scope exit.

// Regression test for https://github.com/llvm/llvm-project/issues/160527:
// the cleanup at an early return releases the memory without a false leak or
// dead-store report.
static void *my_malloc(size_t size) {
  return malloc(size);
}

static void free_pointer(char **p) {
  free(*p);
}

void process_data(void) {
  __attribute__((cleanup(free_pointer))) char *data = my_malloc(100);
  if (!data)
    return;
  return;
} // No leak on any path, and no dead-store warning on data.

// A non-releasing cleanup still leaks.
static void non_releasing_cleanup(char **p) {
  (void)p;
}

void non_releasing_cleanup_leaks(void) {
  char *p __attribute__((cleanup(non_releasing_cleanup)));
  p = malloc(10);
} // expected-warning {{Potential leak of memory pointed to by 'p'}}

// A double free through a cleanup function.
static void double_free_cleanup(char **p) {
  free(*p); // expected-warning {{Attempt to release already released memory}}
}

void double_free_via_cleanup(void) {
  char *p __attribute__((cleanup(double_free_cleanup)));
  p = malloc(10);
  free(p); // The cleanup frees the same pointer again at scope exit.
}

// Directly naming a library function is conservatively evaluated.
void direct_free_cleanup(void) {
  // The emitted call is free(&p); evaluate the call conservatively (no leak).
  void *p __attribute__((cleanup(free))); // expected-warning {{attempt to call free on non-heap object 'p'}}
  p = malloc(10);
}

// A cleanup in a loop runs on every iteration.
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

// A cleanup runs on every exit from the scope, including jumps.
static const char *goto_probe;

static void goto_cleanup(int *p) {
  goto_probe = "cleanup";
  (void)p;
}

void goto_out_of_block_scope(void) {
  {
    int x __attribute__((cleanup(goto_cleanup)));
    x = 1;
    goto_probe = "before goto";
    goto out;
  }
out:
  clang_analyzer_dump(goto_probe); // expected-warning {{"cleanup"}}
}

// A jump that stays inside the variable's scope does not trigger the cleanup.
static void goto_cleanup_at_function_scope(int *p) {
  goto_probe = "cleanup";
  (void)p;
}

void goto_at_function_scope(void) {
  int x __attribute__((cleanup(goto_cleanup_at_function_scope)));
  x = 1;
  goto_probe = "not cleaned up";
  goto out;
out:
  clang_analyzer_dump(goto_probe); // expected-warning {{"not cleaned up"}}
}

// Two cleanup handlers in the same scope run in reverse declaration order.
static const char *order_probe;

static void dump_order_probe(int *p) {
  clang_analyzer_dump(order_probe); // expected-warning {{"second cleanup"}}
  (void)p;
}

static void set_order_probe(int *p) {
  order_probe = "second cleanup";
  (void)p;
}

void cleanup_runs_in_reverse_declaration_order(void) {
  int x __attribute__((cleanup(dump_order_probe)));
  int y __attribute__((cleanup(set_order_probe)));
  order_probe = "before cleanups";
  x = 1;
  y = 2;
}

// A cleanup handler can declare cleanup-attributed variables of its own.
static const char *nested_probe;

static void nested_cleanup(int *p) {
  clang_analyzer_dump(nested_probe); // expected-warning {{"outer handler"}}
  clang_analyzer_dump_int(*p);       // expected-warning {{3 S32b}}
}

static void nested_handler(int *p) {
  int z __attribute__((cleanup(nested_cleanup)));
  nested_probe = "outer handler";
  z = 3;
  (void)p;
}

void cleanup_nested_in_cleanup(void) {
  int x __attribute__((cleanup(nested_handler)));
  nested_probe = "caller";
  x = 42;
}

// A cleanup-annotated variable in a function that the analyzer inlines: the
// cleanup is processed within the inlined stack frame.
static const char *inlined_probe;

static void inlined_function_cleanup(int *p) {
  clang_analyzer_dump(inlined_probe); // expected-warning {{"caller"}}
  clang_analyzer_dump_int(*p);        // expected-warning {{42 S32b}}
}

static void inlined_function_with_cleanup(void) {
  int x __attribute__((cleanup(inlined_function_cleanup)));
  x = 42;
}

void cleanup_in_inlined_function(void) {
  inlined_probe = "caller";
  inlined_function_with_cleanup();
}
