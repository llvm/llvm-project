// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -fno-experimental-new-pass-manager -triple x86_64-unknown-unknown -S -finstrument-functions -O2 -o - %s | FileCheck %s
// RUN: %clang_cc1 -fno-experimental-new-pass-manager -triple x86_64-unknown-unknown -S -finstrument-functions-after-inlining -O2 -o - %s | FileCheck -check-prefix=NOINLINE %s

// RUN: %clang_cc1 -fexperimental-new-pass-manager -triple x86_64-unknown-unknown -S -finstrument-functions -O2 -o - %s | FileCheck %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -triple x86_64-unknown-unknown -S -finstrument-functions-after-inlining -O2 -o - %s | FileCheck -check-prefix=NOINLINE %s

// It's not so nice having asm tests in Clang, but we need to check that we set
// up the pipeline correctly in order to have the instrumentation inserted.

int leaf(int x) {
  return x;
// CHECK-LABEL: leaf:
// CHECK: callq __cyg_profile_func_enter
// CHECK-NOT: cyg_profile
// CHECK: callq __cyg_profile_func_exit
// CHECK-NOT: cyg_profile
// CHECK: ret
}

int root(int x) {
  return leaf(x);
// CHECK-LABEL: root:
// CHECK: callq __cyg_profile_func_enter
// CHECK-NOT: cyg_profile

// Inlined from leaf():
// CHECK: callq __cyg_profile_func_enter
// CHECK-NOT: cyg_profile
// CHECK: callq __cyg_profile_func_exit

// CHECK-NOT: cyg_profile
// CHECK: callq __cyg_profile_func_exit
// CHECK: ret

// NOINLINE-LABEL: root:
// NOINLINE: callq __cyg_profile_func_enter
// NOINLINE-NOT: cyg_profile
// NOINLINE: callq __cyg_profile_func_exit
// NOINLINE: ret
}
