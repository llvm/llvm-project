// REQUIRES: powerpc-registered-target
// RUN: not %clang_cc1 -triple=powerpc-unknown-aix -emit-llvm %s -o - 2>&1 |\
// RUN: FileCheck %s --check-prefix=CHECK32-ERROR
// RUN: %clang_cc1 -O2 -triple=powerpc64-unknown-aix -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=CHECK64
// RUN: %clang_cc1 -O2 -triple=powerpc64le-unknown-linux-gnu -emit-llvm %s \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK64
// RUN: %clang_cc1 -O2 -triple=powerpc64-unknown-linux-gnu -emit-llvm %s \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK64

long test_ldarx(volatile long* a) {
  // CHECK64-LABEL: @test_ldarx
  // CHECK64: %0 = tail call i64 asm sideeffect "ldarx $0, ${1:y}", "=r,*Z,~{memory}"(i64* elementtype(i64) %a)
  // CHECK32-ERROR: error: this builtin is only available on 64-bit targets
  return __ldarx(a);
}

int test_stdcx(volatile long* addr, long val) {
  // CHECK64-LABEL: @test_stdcx
  // CHECK64: %0 = bitcast i64* %addr to i8*
  // CHECK64: %1 = tail call i32 @llvm.ppc.stdcx(i8* %0, i64 %val)
  // CHECK32-ERROR: error: this builtin is only available on 64-bit targets
  return __stdcx(addr, val);
}
