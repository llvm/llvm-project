// RUN: %clang -S -emit-llvm -o - -O0 %s | FileCheck %s --implicit-check-not="call void @llvm.lifetime" -check-prefixes=CHECK
// RUN: %clang -S -emit-llvm -o - -O1 %s | FileCheck %s --implicit-check-not="call void @llvm.lifetime" -check-prefixes=CHECK,LIFETIME
// RUN: %clang -S -emit-llvm -o - -O2 %s | FileCheck %s --implicit-check-not="call void @llvm.lifetime" -check-prefixes=CHECK,LIFETIME
// RUN: %clang -S -emit-llvm -o - -O3 %s | FileCheck %s --implicit-check-not="call void @llvm.lifetime" -check-prefixes=CHECK,LIFETIME

extern void use(char *a);

// CHECK-LABEL: @helper_no_markers
__attribute__((always_inline)) void helper_no_markers(void) {
  char a;
  // LIFETIME: call void @llvm.lifetime.start.p0(
  use(&a);
  // LIFETIME: call void @llvm.lifetime.end.p0(
}

// CHECK-LABEL: @lifetime_test
void lifetime_test(void) {
// LIFETIME: call void @llvm.lifetime.start.p0(
  helper_no_markers();
// LIFETIME: call void @llvm.lifetime.end.p0(
}
