// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -O2 -target-cpu pwr8 -triple=powerpc-unknown-aix \
// RUN:  -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -O2 -target-cpu pwr8 -triple=powerpc64-unknown-aix \
// RUN:  -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -O2 -target-cpu pwr8 -triple=powerpc64le-unknown-linux-gnu \
// RUN:  -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -O2 -target-cpu pwr8 -triple=powerpc64-unknown-linux-gnu \
// RUN:  -emit-llvm %s -o - | FileCheck %s
// RAUN: not %clang_cc1 -no-opaque-pointers -O2 -target-cpu pwr7 -triple=powerpc-unknown-aix \
// RAUN:  -emit-llvm %s -o - 2>&1 | FileCheck %s \
// RAUN:  --check-prefix=CHECK-NON-PWR8-ERR

int test_lwarx(volatile int* a) {
  // CHECK-LABEL: @test_lwarx
  // CHECK: %0 = tail call i32 asm sideeffect "lwarx $0, ${1:y}", "=r,*Z,~{memory}"(i32* elementtype(i32) %a)
  return __lwarx(a);
}

short test_lharx(volatile short *a) {
  // CHECK-LABEL: @test_lharx
  // CHECK: %0 = tail call i16 asm sideeffect "lharx $0, ${1:y}", "=r,*Z,~{memory}"(i16* elementtype(i16) %a)
  // CHECK-NON-PWR8-ERR:  error: this builtin is only valid on POWER8 or later CPUs
  return __lharx(a);
}

char test_lbarx(volatile char *a) {
  // CHECK-LABEL: @test_lbarx
  // CHECK: %0 = tail call i8 asm sideeffect "lbarx $0, ${1:y}", "=r,*Z,~{memory}"(i8* elementtype(i8) %a)
  // CHECK-NON-PWR8-ERR:  error: this builtin is only valid on POWER8 or later CPUs
  return __lbarx(a);
}

int test_stwcx(volatile int* a, int val) {
  // CHECK-LABEL: @test_stwcx
  // CHECK: %0 = bitcast i32* %a to i8*
  // CHECK: %1 = tail call i32 @llvm.ppc.stwcx(i8* %0, i32 %val)
  return __stwcx(a, val);
}

int test_sthcx(volatile short *a, short val) {
  // CHECK-LABEL: @test_sthcx
  // CHECK: %0 = bitcast i16* %a to i8*
  // CHECK: %1 = sext i16 %val to i32
  // CHECK: %2 = tail call i32 @llvm.ppc.sthcx(i8* %0, i32 %1)
  // CHECK-NON-PWR8-ERR:  error: this builtin is only valid on POWER8 or later CPUs
  return __sthcx(a, val);
}

// Extra test cases that previously caused error during usage.
int test_lharx_intret(volatile short *a) {
  // CHECK-LABEL: @test_lharx_intret
  // CHECK: %0 = tail call i16 asm sideeffect "lharx $0, ${1:y}", "=r,*Z,~{memory}"(i16* elementtype(i16) %a)
  // CHECK-NON-PWR8-ERR:  error: this builtin is only valid on POWER8 or later CPUs
  return __lharx(a);
}

int test_lbarx_intret(volatile char *a) {
  // CHECK-LABEL: @test_lbarx_intret
  // CHECK: %0 = tail call i8 asm sideeffect "lbarx $0, ${1:y}", "=r,*Z,~{memory}"(i8* elementtype(i8) %a)
  // CHECK-NON-PWR8-ERR:  error: this builtin is only valid on POWER8 or later CPUs
  return __lbarx(a);
}
