// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple amdgcn %s | FileCheck %s -check-prefix=AMDGCN

void use(char *a);

__attribute__((always_inline)) void helper_no_markers() {
  char a;
  use(&a);
}

void lifetime_test() {
// CHECK: @llvm.lifetime.start.p0i
// AMDGCN: @llvm.lifetime.start.p5i
  helper_no_markers();
}
