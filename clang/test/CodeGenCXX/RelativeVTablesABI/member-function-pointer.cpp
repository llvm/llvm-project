// Member pointer to virtual function.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O3 -S -o - -emit-llvm | FileCheck %s

// CHECK:      define{{.*}} void @_Z4funcP1AMS_FvvE(ptr noundef %a, [2 x i64] %fn.coerce) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[fn_ptr:%.+]] = extractvalue [2 x i64] %fn.coerce, 0
// CHECK-NEXT:   [[adjust:%.+]] = extractvalue [2 x i64] %fn.coerce, 1
// CHECK-NEXT:   [[this_adj:%.+]] = getelementptr inbounds i8, ptr %a, i64 [[adjust]]
// CHECK-NEXT:   [[virtbit:%.+]] = and i64 [[fn_ptr]], 1
// CHECK-NEXT:   [[isvirt:%.+]] = icmp eq i64 [[virtbit]], 0
// CHECK-NEXT:   br i1 [[isvirt]], label %[[nonvirt:.+]], label %[[virt:.+]]
// CHECK:      [[virt]]:

// The loading of the virtual function here should be replaced with a llvm.load.relative() call.
// CHECK-NEXT:   [[vtable:%.+]] = load ptr, ptr [[this_adj]], align 8
// CHECK-NEXT:   [[offset:%.+]] = add i64 [[fn_ptr]], -1
// CHECK-NEXT:   [[ptr:%.+]] = tail call ptr @llvm.load.relative.i64(ptr [[vtable]], i64 [[offset]])
// CHECK-NEXT:   br label %[[memptr_end:.+]]
// CHECK:      [[nonvirt]]:
// CHECK-NEXT:   [[method2:%.+]] = inttoptr i64 [[fn_ptr]] to ptr
// CHECK-NEXT:   br label %[[memptr_end]]
// CHECK:      [[memptr_end]]:
// CHECK-NEXT:   [[method3:%.+]] = phi ptr [ [[ptr]], %[[virt]] ], [ [[method2]], %[[nonvirt]] ]
// CHECK-NEXT:   tail call void [[method3]](ptr {{[^,]*}} [[this_adj]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

class A {
public:
  virtual void foo();
};

class B : public A {
public:
  void foo() override;
};

typedef void (A::*A_foo)();

void func(A *a, A_foo fn) {
  (a->*fn)();
}
