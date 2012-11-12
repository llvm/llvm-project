// RUN: %clang --analyze %s -o %t -Xclang -verify
// expected-no-diagnostics

// Test handling of ObjC bool literals.

typedef signed char BOOL;

void rdar_10597458() {
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
