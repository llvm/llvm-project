// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config max-nodes=12 -verify %s

// Here we test how "suppress on sink" feature of certain bugtypes interacts
// with reaching analysis limits.

// If we report a warning of a bug-type with "suppress on sink" attribute set
// (such as MallocChecker's memory leak warning), then failing to reach the
// reason for the sink (eg. no-return function such as "exit()") due to analysis
// limits (eg. max-nodes option), we may produce a false positive.

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

extern void exit(int) __attribute__ ((__noreturn__));

void clang_analyzer_warnIfReached(void);

int coin(void);

void test_single_cfg_block_sink(void) {
  void *p = malloc(1); // no-warning (wherever the leak warning may occur here)

  // Due to max-nodes option in the run line, we should reach the first call
  // but bail out before the second call.
  // If the test on these two lines starts failing, see if modifying
  // the max-nodes run-line helps.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_warnIfReached(); // no-warning

  // Even though we do not reach this line, we should still suppress
  // the leak report.
  exit(0);
}

// A similar test with more complicated control flow before the no-return thing,
// so that the no-return thing wasn't in the same CFG block.
void test_more_complex_control_flow_before_sink(void) {
  void *p = malloc(1); // no-warning

  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_warnIfReached(); // no-warning

  if (coin())
    exit(0);
  else
    exit(1);
}

// A loop before the no-return function, to make sure that
// the dominated-by-sink analysis doesn't hang.
void test_loop_before_sink(int n) {
  void *p = malloc(1); // no-warning

  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_warnIfReached(); // no-warning

  for (int i = 0; i < n; ++i) {
  }
  exit(1);
}

// We're not sure if this is no-return.
void test_loop_with_sink(int n) {
  void *p = malloc(1); // expected-warning@+2{{Potential leak of memory}}

  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_warnIfReached(); // no-warning

  for (int i = 0; i < n; ++i)
    if (i == 0)
      exit(1);
}

// Handle unreachable blocks correctly.
void test_unreachable_successor_blocks(void) {
  void *p = malloc(1); // no-warning

  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_warnIfReached(); // no-warning

  if (1) // no-crash
    exit(1);
}
