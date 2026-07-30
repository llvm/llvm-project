// RUN: %clang_cc1 -I %S/Inputs -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -no-enable-noundef-analysis -o - %s | FileCheck %s

#include "literal-support.h"

// Check the various selector names we'll be using, in order.

// CHECK: c"numberWithInt:\00"
// CHECK: c"numberWithUnsignedInt:\00"
// CHECK: c"numberWithUnsignedLongLong:\00"
// CHECK: c"numberWithChar:\00"
// CHECK: c"arrayWithObjects:count:\00"
// CHECK: c"dictionaryWithObjects:forKeys:count:\00"
// CHECK: c"prop\00"

// CHECK-LABEL: define{{.*}} void @test_numeric()
void test_numeric(void) {
  // CHECK: {{call.*objc_msgSend.*i32 17.* [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]}}
  id ilit = @17;
  // CHECK: {{call.*objc_msgSend.*i32 25.* [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]}}
  id ulit = @25u;
  // CHECK: {{call.*objc_msgSend.*i64 42.* [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]}}
  id ulllit = @42ull;
  // CHECK: {{call.*objc_msgSend.*i8 signext 97.* [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]}}
  id charlit = @'a';
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test_array
void test_array(id a, id b) {
  // CHECK: [[A:%.*]] = alloca ptr,
  // CHECK: [[B:%.*]] = alloca ptr,

  // Retaining parameters
  // CHECK: call ptr @llvm.objc.retain(ptr
  // CHECK: call ptr @llvm.objc.retain(ptr

  // Constructing the array
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [2 x ptr], ptr [[OBJECTS:%[A-Za-z0-9]+]], i64 0, i64 0
  // CHECK-NEXT: [[V0:%.*]] = load ptr, ptr [[A]],
  // CHECK-NEXT: store ptr [[V0]], ptr [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [2 x ptr], ptr [[OBJECTS]], i64 0, i64 1
  // CHECK-NEXT: [[V1:%.*]] = load ptr, ptr [[B]],
  // CHECK-NEXT: store ptr [[V1]], ptr [[T0]]

  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr @"OBJC_CLASSLIST
  // CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES
  // CHECK-NEXT: [[T3:%.*]] = call ptr @objc_msgSend(ptr [[T0]], ptr [[SEL]], ptr [[OBJECTS]], i64 2) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK: call void (...) @llvm.objc.clang.arc.use(ptr [[V0]], ptr [[V1]])
  id arr = @[a, b];

  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test_dictionary
void test_dictionary(id k1, id o1, id k2, id o2) {
  // CHECK: [[K1:%.*]] = alloca ptr,
  // CHECK: [[O1:%.*]] = alloca ptr,
  // CHECK: [[K2:%.*]] = alloca ptr,
  // CHECK: [[O2:%.*]] = alloca ptr,

  // Retaining parameters
  // CHECK: call ptr @llvm.objc.retain(ptr
  // CHECK: call ptr @llvm.objc.retain(ptr
  // CHECK: call ptr @llvm.objc.retain(ptr
  // CHECK: call ptr @llvm.objc.retain(ptr

  // Constructing the arrays
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [2 x ptr], ptr [[KEYS:%[A-Za-z0-9]+]], i64 0, i64 0
  // CHECK-NEXT: [[V0:%.*]] = load ptr, ptr [[K1]],
  // CHECK-NEXT: store ptr [[V0]], ptr [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [2 x ptr], ptr [[OBJECTS:%[A-Za-z0-9]+]], i64 0, i64 0
  // CHECK-NEXT: [[V1:%.*]] = load ptr, ptr [[O1]],
  // CHECK-NEXT: store ptr [[V1]], ptr [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [2 x ptr], ptr [[KEYS]], i64 0, i64 1
  // CHECK-NEXT: [[V2:%.*]] = load ptr, ptr [[K2]],
  // CHECK-NEXT: store ptr [[V2]], ptr [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [2 x ptr], ptr [[OBJECTS]], i64 0, i64 1
  // CHECK-NEXT: [[V3:%.*]] = load ptr, ptr [[O2]],
  // CHECK-NEXT: store ptr [[V3]], ptr [[T0]]

  // Constructing the dictionary
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr @"OBJC_CLASSLIST
  // CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES
  // CHECK-NEXT: [[T4:%.*]] = call ptr @objc_msgSend(ptr [[T0]], ptr [[SEL]], ptr [[OBJECTS]], ptr [[KEYS]], i64 2) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T4]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[V0]], ptr [[V1]], ptr [[V2]], ptr [[V3]])

  id dict = @{ k1 : o1, k2 : o2 };

  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: ret void
}

@interface A
@end

@interface B
@property (retain) A* prop;
@end

// CHECK-LABEL: define{{.*}} void @test_property
void test_property(B *b) {
  // Retain parameter
  // CHECK: call ptr @llvm.objc.retain

  // CHECK:      [[T0:%.*]] = getelementptr inbounds [1 x ptr], ptr [[OBJECTS:%.*]], i64 0, i64 0

  // Invoke 'prop'
  // CHECK:      [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES
  // CHECK-NEXT: [[V0:%.*]] = call ptr @objc_msgSend(ptr {{.*}}, ptr [[SEL]]) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[V0]])

  // Store to array.
  // CHECK-NEXT: store ptr [[V0]], ptr [[T0]]

  // Invoke arrayWithObjects:count:
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr @"OBJC_CLASSLIST
  // CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES
  // CHECK-NEXT: [[T3:%.*]] = call ptr @objc_msgSend(ptr [[T0]], ptr [[SEL]], ptr [[OBJECTS]], i64 1) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T3]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[V0]])
  // CHECK-NEXT: store
  id arr = @[ b.prop ];

  // Release b.prop
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[V0]])

  // Destroy arr
  // CHECK: call void @llvm.objc.release

  // Destroy b
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: ret void
}
