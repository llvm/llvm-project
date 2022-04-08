// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-linux-gnu -target-cpu core2 %s -S -emit-llvm -o - | FileCheck %s

// All atomics up to 16 bytes should be emitted inline on x86_64. The
// backend can reform __sync_whatever calls if necessary (e.g. the CPU
// doesn't have cmpxchg16b).

__int128 test_sync_call(__int128 *addr, __int128 val) {
  // CHECK-LABEL: @test_sync_call
  // CHECK: atomicrmw add i128* {{.*}} seq_cst, align 16
  return __sync_fetch_and_add(addr, val);
}

__int128 test_c11_call(_Atomic __int128 *addr, __int128 val) {
  // CHECK-LABEL: @test_c11_call
  // CHECK: atomicrmw sub i128* {{.*}} monotonic, align 16
  return __c11_atomic_fetch_sub(addr, val, 0);
}

__int128 test_atomic_call(__int128 *addr, __int128 val) {
  // CHECK-LABEL: @test_atomic_call
  // CHECK: atomicrmw or i128* {{.*}} monotonic, align 16
  return __atomic_fetch_or(addr, val, 0);
}

__int128 test_expression(_Atomic __int128 *addr) {
  // CHECK-LABEL: @test_expression
  // CHECK: atomicrmw and i128* {{.*}} seq_cst, align 16
  *addr &= 1;
}
