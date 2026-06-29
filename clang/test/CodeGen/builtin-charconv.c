// RUN: %clang_cc1 -triple x86_64-linux-gnu -O0 -emit-llvm %s -o - | FileCheck %s

// POC runtime codegen: an inline loop, no external charconv runtime symbol.

int gec;

// CHECK-LABEL: define{{.*}} ptr @do_to_chars
char *do_to_chars(char *first, char *last, int v, int base) {
  return __builtin_to_chars(first, last, v, base);
}
// CHECK: udiv
// CHECK: urem
// CHECK-NOT: call{{.*}}@{{.*}}chars

// CHECK-LABEL: define{{.*}} ptr @do_from_chars
const char *do_from_chars(const char *first, const char *last, int *out, int base) {
  return __builtin_from_chars(first, last, out, base, &gec);
}
// CHECK: @llvm.umul.with.overflow
// CHECK: @llvm.uadd.with.overflow
// CHECK-NOT: call{{.*}}@{{.*}}chars
