// RUN: not --crash %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection \
// RUN:   -x c   %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-C-ONLY
// RUN: not --crash %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection \
// RUN:   -x c++ %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-CXX-ONLY
// REQUIRES: crash-recovery

// Stack traces require back traces.
// REQUIRES: backtrace

void clang_analyzer_crash(void);

void inlined(int x, float y) {
  clang_analyzer_crash();
}

void test(void) {
  inlined(0, 0);
}

// CHECK:                0. Program arguments: {{.*}}clang
// CHECK-NEXT:           1. <eof> parser at end of file
// CHECK-NEXT:           2. While analyzing stack:
//
// CHECK-C-ONLY-NEXT:       #0 Calling inlined at line 17
// CHECK-C-ONLY-NEXT:       #1 Calling test
//
// CHECK-CXX-ONLY-NEXT:     #0 Calling inlined(int, float) at line 17
// CHECK-CXX-ONLY-NEXT:     #1 Calling test()
//
// CHECK-NEXT:           3. {{.*}}crash-trace.c:{{[0-9]+}}:3: Error evaluating statement
