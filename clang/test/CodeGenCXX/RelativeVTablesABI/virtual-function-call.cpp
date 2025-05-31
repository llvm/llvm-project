// Check that we call llvm.load.relative() on a vtable function call.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O3 -o - -emit-llvm | FileCheck %s

// CHECK:      define{{.*}} void @_Z5A_fooP1A(ptr noundef %a) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   %vtable = load ptr, ptr %a
// CHECK-NEXT:   [[func_ptr:%[0-9]+]] = tail call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
// CHECK-NEXT:   tail call void [[func_ptr]](ptr {{[^,]*}} %a)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

class A {
public:
  virtual void foo();
};

void A_foo(A *a) {
  a->foo();
}
