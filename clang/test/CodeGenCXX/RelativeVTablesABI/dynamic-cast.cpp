// dynamic_cast
// Ensure that dynamic casting works normally

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O3 -S -o - -emit-llvm | FileCheck %s

// CHECK:      define{{.*}} ptr @_Z6upcastP1B(ptr noundef readnone returned %b) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret ptr %b
// CHECK-NEXT: }

// CHECK:      define{{.*}} ptr @_Z8downcastP1A(ptr noundef readonly %a) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[isnull:%[0-9]+]] = icmp eq ptr %a, null
// CHECK-NEXT:   br i1 [[isnull]], label %[[dynamic_cast_end:[a-z0-9._]+]], label %[[dynamic_cast_notnull:[a-z0-9._]+]]
// CHECK:      [[dynamic_cast_notnull]]:
// CHECK-NEXT:   [[as_b:%[0-9]+]] = tail call ptr @__dynamic_cast(ptr nonnull %a, ptr nonnull @_ZTI1A, ptr nonnull @_ZTI1B, i64 0)
// CHECK-NEXT:   br label %[[dynamic_cast_end]]
// CHECK:      [[dynamic_cast_end]]:
// CHECK-NEXT:   [[res:%[0-9]+]] = phi ptr [ [[as_b]], %[[dynamic_cast_notnull]] ], [ null, %entry ]
// CHECK-NEXT:   ret ptr [[res]]
// CHECK-NEXT: }

// CHECK: declare ptr @__dynamic_cast(ptr, ptr, ptr, i64) local_unnamed_addr

// CHECK:      define{{.*}} ptr @_Z8selfcastP1B(ptr noundef readnone returned %b) local_unnamed_addr
// CHECK-NEXT: entry
// CHECK-NEXT:   ret ptr %b
// CHECK-NEXT: }

// CHECK: define{{.*}} ptr @_Z9void_castP1B(ptr noundef readonly %b) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[isnull:%[0-9]+]] = icmp eq ptr %b, null
// CHECK-NEXT:   br i1 [[isnull]], label %[[dynamic_cast_end:[a-z0-9._]+]], label %[[dynamic_cast_notnull:[a-z0-9._]+]]
// CHECK:      [[dynamic_cast_notnull]]:
// CHECK-DAG:    [[vtable:%[a-z0-9]+]] = load ptr, ptr %b, align 8
// CHECK-DAG:    [[offset_ptr:%.+]] = getelementptr inbounds i8, ptr [[vtable]], i64 -8
// CHECK-DAG:    [[offset_to_top:%.+]] = load i32, ptr [[offset_ptr]], align 4
// CHECK-DAG:    [[offset_to_top2:%.+]] = sext i32 [[offset_to_top]] to i64
// CHECK-DAG:    [[casted:%.+]] = getelementptr inbounds i8, ptr %b, i64 [[offset_to_top2]]
// CHECK-NEXT:   br label %[[dynamic_cast_end]]
// CHECK:      [[dynamic_cast_end]]:
// CHECK-NEXT:   [[res:%[0-9]+]] = phi ptr [ [[casted]], %[[dynamic_cast_notnull]] ], [ null, %entry ]
// CHECK-NEXT:   ret ptr [[res]]
// CHECK-NEXT: }

class A {
public:
  virtual void foo();
};

class B : public A {
public:
  void foo() override;
};

void A::foo() {}
void B::foo() {}

A *upcast(B *b) {
  return dynamic_cast<A *>(b);
}

B *downcast(A *a) {
  return dynamic_cast<B *>(a);
}

B *selfcast(B *b) {
  return dynamic_cast<B *>(b);
}

void *void_cast(B *b) {
  return dynamic_cast<void *>(b);
}
