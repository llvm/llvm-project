// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s -check-prefix=CHECK-PWR8
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s -check-prefix=CHECK-PWR8
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s -check-prefix=CHECK-PWR8
// RUN: %clang_cc1 -triple powerpc-unknown-aix %s -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s -check-prefix=CHECK-PWR8
// RUN: not %clang_cc1 -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - 2>&1 | FileCheck %s -check-prefix=CHECK-NOPWR8
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - 2>&1 | FileCheck %s -check-prefix=CHECK-NOPWR8
// RUN: not %clang_cc1 -triple powerpc-unknown-aix %s -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - 2>&1 | FileCheck %s -check-prefix=CHECK-NOPWR8

extern void *a;

void test_icbt() {
// CHECK-LABEL: @test_icbt(

  __icbt(a);
// CHECK-PWR8: call void @llvm.ppc.icbt(i8* %0)
// CHECK-NOPWR8: error: this builtin is only valid on POWER8 or later CPUs
}

void test_builtin_ppc_icbt() {
// CHECK-LABEL: @test_builtin_ppc_icbt(

  __builtin_ppc_icbt(a);
// CHECK-PWR8: call void @llvm.ppc.icbt(i8* %0)
// CHECK-NOPWR8: error: this builtin is only valid on POWER8 or later CPUs
}
