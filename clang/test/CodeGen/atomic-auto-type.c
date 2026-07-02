// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c11 -emit-llvm -o - %s | FileCheck %s

// This is a regression test for handling of __auto_type inside _Atomic.
// Previously this could lead to an undeduced AutoType escaping into
// ASTContext::getTypeInfoImpl and causing an assertion failure.

// 24-byte struct exceeds the lock-free threshold on x86_64.
struct Large { double a, b, c; };

// _Atomic double is lock-free on x86_64: deduced type lowers to plain stores.
// CHECK-LABEL: @test_atomic_auto_type(
// CHECK:      alloca double
// CHECK-NEXT: %[[XA:.*]] = alloca double
// CHECK-NEXT: %[[AX:.*]] = alloca double
// CHECK:      store double {{.*}}, ptr %[[XA]]
// CHECK:      store double {{.*}}, ptr %[[AX]]
// CHECK:      ret void

void test_atomic_auto_type(double x) {
  __auto_type _Atomic xa = x;
  _Atomic __auto_type ax = x;
}

// _Atomic struct Large is not lock-free, so assignments (which must be atomic)
// call __atomic_store. Initialization of _Atomic is not required to be atomic.
// CHECK-LABEL: @test_atomic_auto_type_lock(
// CHECK:      %[[LA:.*]] = alloca %struct.Large
// CHECK-NEXT: %[[AL:.*]] = alloca %struct.Large
// CHECK:      call void @llvm.memcpy{{.*}}({{.*}} %[[LA]],
// CHECK:      call void @llvm.memcpy{{.*}}({{.*}} %[[AL]],
// CHECK:      call void @__atomic_store({{.*}} %[[LA]],
// CHECK:      call void @__atomic_store({{.*}} %[[AL]],
// CHECK:      ret void

void test_atomic_auto_type_lock(struct Large x) {
  __auto_type _Atomic la = x;
  _Atomic __auto_type al = x;
  la = x;
  al = x;
}
