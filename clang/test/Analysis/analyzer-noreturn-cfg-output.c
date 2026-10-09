// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG %s 2>&1 | FileCheck %s

void analyzer_stop(void) __attribute__((analyzer_noreturn));
void real_stop(void) __attribute__((noreturn));

// CHECK-LABEL: void analyzer(
// CHECK: (ANALYZER NORETURN)]
// CHECK-NOT: (NORETURN)]
void analyzer(void) {
  analyzer_stop();
}

// CHECK-LABEL: void real(
// CHECK: (NORETURN)]
// CHECK-NOT: (ANALYZER NORETURN)]
void real(void) {
  real_stop();
}
