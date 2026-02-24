// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Verify that _Atomic __int256 operations generate the correct libcalls.
// __int256 is too large for inline atomics (256 bits > cmpxchg16b), so all
// operations must route through __atomic_* libcalls with size=32.

_Atomic __int256_t glob;

// CHECK-LABEL: define{{.*}} void @atomic_load(ptr{{.*}}sret(i256)
// CHECK: call void @__atomic_load(i64 noundef 32, ptr noundef @glob, ptr noundef %{{.*}}, i32 noundef 5)
__int256_t atomic_load(void) {
  return __c11_atomic_load(&glob, __ATOMIC_SEQ_CST);
}

// CHECK-LABEL: define{{.*}} void @atomic_store(ptr noundef byval(i256) align 16 %0)
// CHECK: call void @__atomic_store(i64 noundef 32, ptr noundef @glob, ptr noundef %{{.*}}, i32 noundef 3)
void atomic_store(__int256_t val) {
  __c11_atomic_store(&glob, val, __ATOMIC_RELEASE);
}

// CHECK-LABEL: define{{.*}} void @atomic_exchange(ptr{{.*}}sret(i256){{.*}}, ptr noundef byval(i256) align 16 %0)
// CHECK: call void @__atomic_exchange(i64 noundef 32, ptr noundef @glob, ptr noundef %{{.*}}, ptr noundef %{{.*}}, i32 noundef 5)
__int256_t atomic_exchange(__int256_t val) {
  return __c11_atomic_exchange(&glob, val, __ATOMIC_SEQ_CST);
}

// CHECK-LABEL: define{{.*}} i1 @atomic_cas(
// CHECK: call{{.*}} i1 @__atomic_compare_exchange(i64 noundef 32, ptr noundef @glob, ptr noundef %{{.*}}, ptr noundef %{{.*}}, i32 noundef 4, i32 noundef 2)
_Bool atomic_cas(__int256_t *expected, __int256_t desired) {
  return __c11_atomic_compare_exchange_strong(
      &glob, expected, desired, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
