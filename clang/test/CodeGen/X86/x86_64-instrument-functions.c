// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -finstrument-functions -O0 -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -finstrument-functions -O2 -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -finstrument-functions-after-inlining -O2 -o - -emit-llvm %s | FileCheck -check-prefix=NOINLINE %s

__attribute__((always_inline)) int leaf(int x) {
  return x;
// CHECK-LABEL: define {{.*}} @leaf(i32 noundef %x) #0 {
}

int root(int x) {
  return leaf(x);
// CHECK-LABEL: define {{.*}} @root(i32 noundef %x) #1 {
// NOINLINE-LABEL: define {{.*}} @root(i32 noundef %x) #1 {
}

// CHECK: attributes #0 = { {{.*}}"instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit"
// CHECK: attributes #1 = { {{.*}}"instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit"
// NOINLINE: attributes #0 = { {{.*}}"instrument-function-entry-inlined"="__cyg_profile_func_enter" "instrument-function-exit-inlined"="__cyg_profile_func_exit"
// NOINLINE: attributes #1 = { {{.*}}"instrument-function-entry-inlined"="__cyg_profile_func_enter" "instrument-function-exit-inlined"="__cyg_profile_func_exit"
