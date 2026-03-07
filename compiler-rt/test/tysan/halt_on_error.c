// RUN: %clang_tysan %s -o %t
// RUN: %run %t 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-CONTINUE %s
// RUN: %env_tysan_opts=halt_on_error=1 not %run %t 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-HALT %s

int main() {

  int i = 5;

  float *f = (float *)&i;

  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4
  // CHECK-HALT: ABORTING
  *f = 5.0f;

  // CHECK-CONTINUE: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK-CONTINUE: READ of size 4
  // CHECK-HALT-NOT: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK-HALT-NOT: READ of size 4
  i = *f;

  return 0;
}
