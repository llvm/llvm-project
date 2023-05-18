// RUN: %clang_cc1 -triple x86_64-gnu-linux %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-gnu-linux %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-windows-msvc %s -emit-llvm -o - | FileCheck %s
void test_builtin_set_flt_rounds() {
  __builtin_set_flt_rounds(1);
  // CHECK: call void @llvm.set.rounding(i32 1)
}
