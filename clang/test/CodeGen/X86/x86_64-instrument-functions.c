// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -fexperimental-new-pass-manager -triple x86_64-unknown-unknown -S -finstrument-functions -O0 -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -triple x86_64-unknown-unknown -S -finstrument-functions -O2 -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -triple x86_64-unknown-unknown -S -finstrument-functions-after-inlining -O2 -o - -emit-llvm %s | FileCheck -check-prefix=NOINLINE %s

__attribute__((always_inline)) int leaf(int x) {
  return x;
// CHECK-LABEL: define {{.*}} @leaf
// CHECK: call void @__cyg_profile_func_enter
// CHECK-NOT: cyg_profile
// CHECK: call void @__cyg_profile_func_exit
// CHECK-NOT: cyg_profile
// CHECK: ret
}

int root(int x) {
  return leaf(x);
// CHECK-LABEL: define {{.*}} @root
// CHECK: call void @__cyg_profile_func_enter
// CHECK-NOT: cyg_profile

// Inlined from leaf():
// CHECK: call void @__cyg_profile_func_enter
// CHECK-NOT: cyg_profile
// CHECK: call void @__cyg_profile_func_exit
// CHECK-NOT: cyg_profile

// CHECK: call void @__cyg_profile_func_exit
// CHECK: ret

// NOINLINE-LABEL: define {{.*}} @root
// NOINLINE: call void @__cyg_profile_func_enter
// NOINLINE-NOT: cyg_profile
// NOINLINE: call void @__cyg_profile_func_exit
// NOINLINE-NOT: cyg_profile
// NOINLINE: ret
}
