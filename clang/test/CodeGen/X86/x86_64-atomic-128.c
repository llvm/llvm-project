// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-cpu core2 %s -emit-llvm -o - | FileCheck %s

// All atomics up to 16 bytes should be emitted inline on x86_64. The
// backend can reform __sync_whatever calls if necessary (e.g. the CPU
// doesn't have cmpxchg16b).

__int128 test_sync_call(__int128 *addr, __int128 val) {
  // CHECK-LABEL: @test_sync_call
  // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 16
  return __sync_fetch_and_add(addr, val);
}

__int128 test_c11_call(_Atomic __int128 *addr, __int128 val) {
  // CHECK-LABEL: @test_c11_call
  // CHECK: atomicrmw sub ptr {{.*}} monotonic, align 16
  return __c11_atomic_fetch_sub(addr, val, 0);
}

__int128 test_atomic_call(__int128 *addr, __int128 val) {
  // CHECK-LABEL: @test_atomic_call
  // CHECK: atomicrmw or ptr {{.*}} monotonic, align 16
  return __atomic_fetch_or(addr, val, 0);
}

__int128 test_expression(_Atomic __int128 *addr) {
  // CHECK-LABEL: @test_expression
  // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 16
  *addr &= 1;
  return 0;
}
