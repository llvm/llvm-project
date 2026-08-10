// RUN: %clang_analyze_cc1 %s -o %t -verify
// expected-no-diagnostics

// Test handling of ObjC bool literals.

typedef signed char BOOL;

void rdar_10597458(void) {
  if (__objc_yes)
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}

void rdar_10597458_b(BOOL b) {
  if (b == __objc_no)
    return;
  
  if (b == __objc_no) {
    int *p = 0;
    *p = 0xDEADBEEF; // no-warning
  }
}
