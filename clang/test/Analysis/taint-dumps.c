// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.security.taint\
// RUN:                    -analyzer-checker=debug.ExprInspection %s\
// RUN:                                          2>&1 | FileCheck %s

void clang_analyzer_printState(void);
int getchar(void);

// CHECK: Tainted symbols:
// CHECK-NEXT: conj_$2{{.*}} : 0
int test_taint_dumps(void) {
  int x = getchar();
  clang_analyzer_printState();
  return x;
}
