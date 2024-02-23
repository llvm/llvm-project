// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fstrict-vtable-pointers -o - %s \
// RUN: | FileCheck --check-prefixes=CHECK,CHECK-STRICT %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefixes=CHECK,CHECK-NONSTRICT %s

struct TestVirtualFn {
  virtual void foo();
};
// CHECK-LABEL: define{{.*}} void @test_dynamic_class
extern "C" void test_dynamic_class(TestVirtualFn *p) {
  // CHECK: store ptr %p, ptr %p.addr
  // CHECK-NEXT: [[TMP0:%.*]] = load ptr, ptr %p.addr

  // CHECK-NONSTRICT-NEXT: store ptr [[TMP0]], ptr %d

  // CHECK-STRICT-NEXT: [[TMP2:%.*]] = call ptr @llvm.launder.invariant.group.p0(ptr [[TMP0]])
  // CHECK-STRICT-NEXT: store ptr [[TMP2]], ptr %d

  // CHECK-NEXT: ret void
  TestVirtualFn *d = __builtin_start_object_lifetime(p);
}

// CHECK-LABEL: define{{.*}} void @test_scalar_pointer
extern "C" void test_scalar_pointer(int *p) {
  // CHECK: entry
  // CHECK-NEXT: %p.addr = alloca ptr
  // CHECK-NEXT: %d = alloca ptr
  // CHECK-NEXT: store ptr %p, ptr %p.addr, align 8
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr %p.addr
  // CHECK-NEXT: store ptr [[TMP]], ptr %d
  // CHECK-NEXT: ret void
  int *d = __builtin_start_object_lifetime(p);
}

struct TestNoInvariant {
  int x;
};
// CHECK-LABEL: define{{.*}} void @test_non_dynamic_class
extern "C" void test_non_dynamic_class(TestNoInvariant *p) {
  // CHECK: entry
  // CHECK-NOT: llvm.launder.invariant.group
  // CHECK-NEXT: %p.addr = alloca ptr, align 8
  // CHECK-NEXT: %d = alloca ptr
  // CHECK-NEXT: store ptr %p, ptr %p.addr
  // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr %p.addr
  // CHECK-NEXT: store ptr [[TMP]], ptr %d
  // CHECK-NEXT: ret void
  TestNoInvariant *d = __builtin_start_object_lifetime(p);
}
