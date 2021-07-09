// RUN: not %clang_cc1 -triple=powerpc-unknown-aix -emit-llvm %s -o - 2>&1 |\
// RUN: FileCheck %s --check-prefix=CHECK32-ERROR
// RUN: %clang_cc1 -triple=powerpc64-unknown-aix -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=CHECK64
// RUN: %clang_cc1 -triple=powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK64
// RUN: %clang_cc1 -triple=powerpc64-unknown-unknown -emit-llvm %s \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK64

long test_ldarx(volatile long* a) {
  // CHECK64-LABEL: @test_ldarx
  // CHECK64: %0 = load i64*, i64** %a.addr, align 8
  // CHECK64: %1 = bitcast i64* %0 to i8*
  // CHECK64: %2 = call i64 @llvm.ppc.ldarx(i8* %1)
  // CHECK32-ERROR: error: this builtin is only available on 64-bit targets
  return __ldarx(a);
}

int test_stdcx(volatile long* addr, long val) {
  // CHECK64-LABEL: @test_stdcx
  // CHECK64: %0 = load i64*, i64** %addr.addr, align 8
  // CHECK64: %1 = bitcast i64* %0 to i8*
  // CHECK64: %2 = load i64, i64* %val.addr, align 8
  // CHECK64: %3 = call i32 @llvm.ppc.stdcx(i8* %1, i64 %2)
  // CHECK32-ERROR: error: this builtin is only available on 64-bit targets
  return __stdcx(addr, val);
}
