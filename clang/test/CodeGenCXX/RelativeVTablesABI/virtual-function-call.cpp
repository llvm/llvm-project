// Check that we call llvm.load.relative() on a vtable function call.

// RUN: %clang_cc1 -no-opaque-pointers %s -triple=aarch64-unknown-fuchsia -O3 -S -o - -emit-llvm | FileCheck %s

// CHECK:      define{{.*}} void @_Z5A_fooP1A(%class.A* noundef %a) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[this:%[0-9]+]] = bitcast %class.A* %a to i8**
// CHECK-NEXT:   %vtable1 = load i8*, i8** [[this]]
// CHECK-NEXT:   [[func_ptr:%[0-9]+]] = tail call i8* @llvm.load.relative.i32(i8* %vtable1, i32 0)
// CHECK-NEXT:   [[func:%[0-9]+]] = bitcast i8* [[func_ptr]] to void (%class.A*)*
// CHECK-NEXT:   tail call void [[func]](%class.A* {{[^,]*}} %a)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

class A {
public:
  virtual void foo();
};

void A_foo(A *a) {
  a->foo();
}
