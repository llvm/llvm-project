// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern int si;
extern unsigned int ui;

int test_builtin_ppc_mulhw() {
  // CHECK-LABEL: @test_builtin_ppc_mulhw(
  // CHECK:       %2 = call i32 @llvm.ppc.mulhw(i32 %0, i32 %1)
  return __builtin_ppc_mulhw(si, si);
}

unsigned int test_builtin_ppc_mulhwu() {
  // CHECK-LABEL: @test_builtin_ppc_mulhwu(
  // CHECK:       %2 = call i32 @llvm.ppc.mulhwu(i32 %0, i32 %1)
  return __builtin_ppc_mulhwu(ui, ui);
}
