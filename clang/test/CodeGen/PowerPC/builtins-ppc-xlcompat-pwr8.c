// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s -check-prefix=CHECK-PWR8
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s -check-prefix=CHECK-PWR8
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s -check-prefix=CHECK-PWR8
// RUN: %clang_cc1 -triple powerpc-unknown-aix %s -emit-llvm %s \
// RUN:   -target-cpu pwr8 -o - | FileCheck %s -check-prefix=CHECK-PWR8
// RUN: not %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - 2>&1 | FileCheck %s -check-prefix=CHECK-NOPWR8
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - 2>&1 | FileCheck %s -check-prefix=CHECK-NOPWR8
// RUN: not %clang_cc1 -triple powerpc-unknown-aix %s -emit-llvm %s \
// RUN:   -target-cpu pwr7 -o - 2>&1 | FileCheck %s -check-prefix=CHECK-NOPWR8

extern void *a;
extern volatile char *c_addr;
extern char *ptr;
extern char c;
extern int i;
extern vector unsigned char vuc;

void test_xlcompat() {
  // CHECK-PWR8-LABEL: @test_xlcompat(
  // CHECK-PWR8: call void @llvm.ppc.icbt(ptr %{{[0-9]+}})
  // CHECK-NOPWR8: error: '__builtin_ppc_icbt' needs target feature isa-v207-instructions
  __icbt(a);

  // CHECK-PWR8: call void @llvm.ppc.icbt(ptr %{{[0-9]+}})
  // CHECK-NOPWR8: error: '__builtin_ppc_icbt' needs target feature isa-v207-instructions
  __builtin_ppc_icbt(a);

  // CHECK-PWR8:         [[TMP0:%.*]] = load ptr, ptr @c_addr, align {{[0-9]+}}
  // CHECK-PWR8-NEXT:    [[TMP1:%.*]] = load i8, ptr @c, align 1
  // CHECK-PWR8-NEXT:    [[TMP2:%.*]] = sext i8 [[TMP1]] to i32
  // CHECK-PWR8-NEXT:    [[TMP3:%.*]] = call i32 @llvm.ppc.stbcx(ptr [[TMP0]], i32 [[TMP2]])
  // CHECK-NOPWR8: error: '__builtin_ppc_stbcx' needs target feature isa-v207-instructions
  i = __builtin_ppc_stbcx(c_addr, c);

  // CHECK-NOPWR8: error: '__builtin_vsx_ldrmb' needs target feature isa-v207-instructions
  vuc = __builtin_vsx_ldrmb(ptr, 14);

  // CHECK-NOPWR8: error: '__builtin_vsx_strmb' needs target feature isa-v207-instructions
  __builtin_vsx_strmb(ptr, 14, vuc);
}
