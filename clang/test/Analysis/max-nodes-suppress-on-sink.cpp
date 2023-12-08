// RUN: %clang_analyze_cc1 -x c++ -fcxx-exceptions -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config max-nodes=12 -verify %s

// Here we test how "suppress on sink" feature of certain bugtypes interacts
// with reaching analysis limits. See comments in max-nodes-suppress-on-sink.c
// for more discussion.

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

void clang_analyzer_warnIfReached(void);

// Because we don't have a better approach, we currently treat throw as
// noreturn.
void test_throw_treated_as_noreturn() {
  void *p = malloc(1); // no-warning

  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_warnIfReached(); // no-warning

  throw 0;
}

// FIXME: Handled throws shouldn't be suppressing us!
void test_handled_throw_treated_as_noreturn() {
  void *p = malloc(1); // no-warning

  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_warnIfReached(); // no-warning

  try {
    throw 0;
  } catch (int i) {
  }
}
