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
extern volatile char *c_addr;
extern char c;

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

int test_builtin_ppc_stbcx() {
// CHECK-PWR8-LABEL: @test_builtin_ppc_stbcx(
// CHECK-PWR8:         [[TMP0:%.*]] = load i8*, i8** @c_addr, align {{[0-9]+}}
// CHECK-PWR8-NEXT:    [[TMP1:%.*]] = load i8, i8* @c, align 1
// CHECK-PWR8-NEXT:    [[TMP2:%.*]] = sext i8 [[TMP1]] to i32
// CHECK-PWR8-NEXT:    [[TMP3:%.*]] = call i32 @llvm.ppc.stbcx(i8* [[TMP0]], i32 [[TMP2]])
// CHECK-PWR8-NEXT:    ret i32 [[TMP3]]
// CHECK-NOPWR8: error: this builtin is only valid on POWER8 or later CPUs
  return __builtin_ppc_stbcx(c_addr, c);
}

vector unsigned char test_ldrmb(char *ptr) {
  // CHECK-NOPWR8: error: this builtin is only valid on POWER8 or later CPUs
  return __builtin_vsx_ldrmb(ptr, 14);
}

void test_strmbb(char *ptr, vector unsigned char data) {
  // CHECK-NOPWR8: error: this builtin is only valid on POWER8 or later CPUs
  __builtin_vsx_strmb(ptr, 14, data);
}
