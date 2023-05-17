// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux-gnu -O2 -o - %s | FileCheck -check-prefix=NULL-POINTER-INVALID  %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux-gnu -O2 -o - %s -fno-delete-null-pointer-checks | FileCheck -check-prefix=NULL-POINTER-VALID  %s

// Test that clang does not remove the null pointer check with
// -fno-delete-null-pointer-checks.
int null_check(int *P) {
// NULL-POINTER-VALID: %[[TOBOOL:.*]] = icmp eq ptr %P, null
// NULL-POINTER-INVALID-NOT: icmp eq
// NULL-POINTER-VALID: %[[SEL:.*]] = select i1 %[[TOBOOL:.*]], ptr null, ptr
// NULL-POINTER-INVALID-NOT: select i1
// NULL-POINTER-VALID: load i32, ptr %[[SEL:.*]]
  int *Q = P;
  if (P) {
    Q = P + 2;
  }
  return *Q;
}

// NULL-POINTER-INVALID-NOT: attributes #0 = {{.*}} null_pointer_is_valid
// NULL-POINTER-VALID: attributes #0 = {{.*}} null_pointer_is_valid
