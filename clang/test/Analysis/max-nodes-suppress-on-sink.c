// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config max-nodes=12 -verify %s

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

void test_single_cfg_block_sink() {
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
