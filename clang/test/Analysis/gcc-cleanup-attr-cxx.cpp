// RUN: %clang_analyze_cc1 -x c++ -std=c++17 -analyzer-checker=core,unix.Malloc,deadcode.DeadStores,debug.ExprInspection -verify %s

// Test scope-exit shapes of GCC's __attribute__((cleanup(f))) that are
// specific to C++: interaction with destructors and lambdas.

#include "Inputs/system-header-simulator-for-malloc.h"

void clang_analyzer_dump_int(int);
void clang_analyzer_warnIfReached(void);

//===----------------------------------------------------------------------===//
// For a variable with both a cleanup attribute and a non-trivial destructor,
// the cleanup function runs before the destructor, matching the order in
// which clang codegen emits the two calls.
//===----------------------------------------------------------------------===//

static int g;

struct WithDtor {
  ~WithDtor() { g = 2; }
};

static void cleanup_before_dtor_probe(struct WithDtor *p) {
  clang_analyzer_dump_int(g); // expected-warning {{1 S32b}}
  (void)p;
}

void cleanup_runs_before_destructor(void) {
  struct WithDtor w __attribute__((cleanup(cleanup_before_dtor_probe)));
  g = 1;
} // The destructor would set g = 2; the dump above shows 1, so the cleanup
  // function ran first.

//===----------------------------------------------------------------------===//
// A cleanup-annotated variable inside a lambda body: the cleanup runs when
// the lambda call operator is inlined.
//===----------------------------------------------------------------------===//

static void lambda_cleanup(int *p) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  (void)p;
}

void cleanup_in_lambda_body(void) {
  auto lam = []() {
    int z __attribute__((cleanup(lambda_cleanup)));
    z = 1;
  };
  lam();
}

//===----------------------------------------------------------------------===//
// A cleanup-annotated variable in a C++ function that the analyzer inlines:
// the cleanup call is processed within the inlined stack frame.
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
