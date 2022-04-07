// dynamic_cast
// Ensure that dynamic casting works normally

// RUN: %clang_cc1 -no-opaque-pointers %s -triple=aarch64-unknown-fuchsia -O3 -S -o - -emit-llvm | FileCheck %s

// CHECK:      define{{.*}} %class.A* @_Z6upcastP1B(%class.B* noundef readnone %b) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[a:%[0-9]+]] = getelementptr %class.B, %class.B* %b, i64 0, i32 0
// CHECK-NEXT:   ret %class.A* [[a]]
// CHECK-NEXT: }

// CHECK:      define{{.*}} %class.B* @_Z8downcastP1A(%class.A* noundef readonly %a) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[isnull:%[0-9]+]] = icmp eq %class.A* %a, null
// CHECK-NEXT:   br i1 [[isnull]], label %[[dynamic_cast_end:[a-z0-9._]+]], label %[[dynamic_cast_notnull:[a-z0-9._]+]]
// CHECK:      [[dynamic_cast_notnull]]:
// CHECK-NEXT:   [[a:%[0-9]+]] = bitcast %class.A* %a to i8*
// CHECK-NEXT:   [[as_b:%[0-9]+]] = tail call i8* @__dynamic_cast(i8* nonnull [[a]], i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTI1B to i8*), i64 0)
// CHECK-NEXT:   [[b:%[0-9]+]] = bitcast i8* [[as_b]] to %class.B*
// CHECK-NEXT:   br label %[[dynamic_cast_end]]
// CHECK:      [[dynamic_cast_end]]:
// CHECK-NEXT:   [[res:%[0-9]+]] = phi %class.B* [ [[b]], %[[dynamic_cast_notnull]] ], [ null, %entry ]
// CHECK-NEXT:   ret %class.B* [[res]]
// CHECK-NEXT: }

// CHECK: declare i8* @__dynamic_cast(i8*, i8*, i8*, i64) local_unnamed_addr

// CHECK:      define{{.*}} %class.B* @_Z8selfcastP1B(%class.B* noundef readnone returned %b) local_unnamed_addr
// CHECK-NEXT: entry
// CHECK-NEXT:   ret %class.B* %b
// CHECK-NEXT: }

// CHECK: define{{.*}} i8* @_Z9void_castP1B(%class.B* noundef readonly %b) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[isnull:%[0-9]+]] = icmp eq %class.B* %b, null
// CHECK-NEXT:   br i1 [[isnull]], label %[[dynamic_cast_end:[a-z0-9._]+]], label %[[dynamic_cast_notnull:[a-z0-9._]+]]
// CHECK:      [[dynamic_cast_notnull]]:
// CHECK-DAG:    [[b2:%[0-9]+]] = bitcast %class.B* %b to i32**
// CHECK-DAG:    [[vtable:%[a-z0-9]+]] = load i32*, i32** [[b2]], align 8
// CHECK-DAG:    [[offset_ptr:%.+]] = getelementptr inbounds i32, i32* [[vtable]], i64 -2
// CHECK-DAG:    [[offset_to_top:%.+]] = load i32, i32* [[offset_ptr]], align 4
// CHECK-DAG:    [[b:%[0-9]+]] = bitcast %class.B* %b to i8*
// CHECK-DAG:    [[offset_to_top2:%.+]] = sext i32 [[offset_to_top]] to i64
// CHECK-DAG:    [[casted:%.+]] = getelementptr inbounds i8, i8* [[b]], i64 [[offset_to_top2]]
// CHECK-NEXT:   br label %[[dynamic_cast_end]]
// CHECK:      [[dynamic_cast_end]]:
// CHECK-NEXT:   [[res:%[0-9]+]] = phi i8* [ [[casted]], %[[dynamic_cast_notnull]] ], [ null, %entry ]
// CHECK-NEXT:   ret i8* [[res]]
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
