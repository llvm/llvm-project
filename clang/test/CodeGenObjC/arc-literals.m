// RUN: %clang_cc1 -I %S/Inputs -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s

#include "literal-support.h"

// Check the various selector names we'll be using, in order.

// CHECK: c"numberWithInt:\00"
// CHECK: c"numberWithUnsignedInt:\00"
// CHECK: c"numberWithUnsignedLongLong:\00"
// CHECK: c"numberWithChar:\00"
// CHECK: c"arrayWithObjects:count:\00"
// CHECK: c"dictionaryWithObjects:forKeys:count:\00"
// CHECK: c"prop\00"

// CHECK-LABEL: define void @test_numeric()
void test_numeric() {
  // CHECK: {{call.*objc_msgSend.*i32 17}}
  // CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue
  id ilit = @17;
  // CHECK: {{call.*objc_msgSend.*i32 25}}
  // CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue
  id ulit = @25u;
  // CHECK: {{call.*objc_msgSend.*i64 42}}
  // CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue
  id ulllit = @42ull;
  // CHECK: {{call.*objc_msgSend.*i8 signext 97}}
  // CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue
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

// CHECK-LABEL: define void @test_array
void test_array(id a, id b) {
  // CHECK: [[A:%.*]] = alloca i8*,
  // CHECK: [[B:%.*]] = alloca i8*,

  // Retaining parameters
  // CHECK: call i8* @llvm.objc.retain(i8*
  // CHECK: call i8* @llvm.objc.retain(i8*

  // Constructing the array
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OBJECTS:%[A-Za-z0-9]+]], i64 0, i64 0
  // CHECK-NEXT: [[V0:%.*]] = load i8*, i8** [[A]],
  // CHECK-NEXT: store i8* [[V0]], i8** [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OBJECTS]], i64 0, i64 1
  // CHECK-NEXT: [[V1:%.*]] = load i8*, i8** [[B]],
  // CHECK-NEXT: store i8* [[V1]], i8** [[T0]]

  // CHECK-NEXT: [[T0:%.*]] = load [[CLASS_T:%.*]]*, [[CLASS_T:%.*]]** @"OBJC_CLASSLIST
  // CHECK-NEXT: [[SEL:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[CLASS_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = bitcast [2 x i8*]* [[OBJECTS]] to i8**
  // CHECK-NEXT: [[T3:%.*]] = call i8* bitcast ({{.*@objc_msgSend.*}})(i8* [[T1]], i8* [[SEL]], i8** [[T2]], i64 2)
  // CHECK-NEXT: [[T4:%.*]] = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T3]])
  // CHECK: call void (...) @llvm.objc.clang.arc.use(i8* [[V0]], i8* [[V1]])
  id arr = @[a, b];

  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define void @test_dictionary
void test_dictionary(id k1, id o1, id k2, id o2) {
  // CHECK: [[K1:%.*]] = alloca i8*,
  // CHECK: [[O1:%.*]] = alloca i8*,
  // CHECK: [[K2:%.*]] = alloca i8*,
  // CHECK: [[O2:%.*]] = alloca i8*,

  // Retaining parameters
  // CHECK: call i8* @llvm.objc.retain(i8*
  // CHECK: call i8* @llvm.objc.retain(i8*
  // CHECK: call i8* @llvm.objc.retain(i8*
  // CHECK: call i8* @llvm.objc.retain(i8*

  // Constructing the arrays
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[KEYS:%[A-Za-z0-9]+]], i64 0, i64 0
  // CHECK-NEXT: [[V0:%.*]] = load i8*, i8** [[K1]],
  // CHECK-NEXT: store i8* [[V0]], i8** [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OBJECTS:%[A-Za-z0-9]+]], i64 0, i64 0
  // CHECK-NEXT: [[V1:%.*]] = load i8*, i8** [[O1]],
  // CHECK-NEXT: store i8* [[V1]], i8** [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[KEYS]], i64 0, i64 1
  // CHECK-NEXT: [[V2:%.*]] = load i8*, i8** [[K2]],
  // CHECK-NEXT: store i8* [[V2]], i8** [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OBJECTS]], i64 0, i64 1
  // CHECK-NEXT: [[V3:%.*]] = load i8*, i8** [[O2]],
  // CHECK-NEXT: store i8* [[V3]], i8** [[T0]]

  // Constructing the dictionary
  // CHECK-NEXT: [[T0:%.*]] = load [[CLASS_T:%.*]]*, [[CLASS_T:%.*]]** @"OBJC_CLASSLIST
  // CHECK-NEXT: [[SEL:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[CLASS_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = bitcast [2 x i8*]* [[OBJECTS]] to i8**
  // CHECK-NEXT: [[T3:%.*]] = bitcast [2 x i8*]* [[KEYS]] to i8**
  // CHECK-NEXT: [[T4:%.*]] = call i8* bitcast ({{.*@objc_msgSend.*}})(i8* [[T1]], i8* [[SEL]], i8** [[T2]], i8** [[T3]], i64 2)
  // CHECK-NEXT: [[T5:%.*]] = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T4]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(i8* [[V0]], i8* [[V1]], i8* [[V2]], i8* [[V3]])

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

// CHECK-LABEL: define void @test_property
void test_property(B *b) {
  // Retain parameter
  // CHECK: call i8* @llvm.objc.retain

  // CHECK:      [[T0:%.*]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[OBJECTS:%.*]], i64 0, i64 0

  // Invoke 'prop'
  // CHECK:      [[SEL:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES
  // CHECK-NEXT: [[T1:%.*]] = bitcast
  // CHECK-NEXT: [[T2:%.*]] = call [[B:%.*]]* bitcast ({{.*}} @objc_msgSend to {{.*}})(i8* [[T1]], i8* [[SEL]])
  // CHECK-NEXT: [[T3:%.*]] = bitcast [[B]]* [[T2]] to i8*
  // CHECK-NEXT: [[T4:%.*]] = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T3]])
  // CHECK-NEXT: [[V0:%.*]] = bitcast i8* [[T4]] to [[B]]*
  // CHECK-NEXT: [[V1:%.*]] = bitcast [[B]]* [[V0]] to i8*

  // Store to array.
  // CHECK-NEXT: store i8* [[V1]], i8** [[T0]]

  // Invoke arrayWithObjects:count:
  // CHECK-NEXT: [[T0:%.*]] = load [[CLASS_T]]*, [[CLASS_T]]** @"OBJC_CLASSLIST
  // CHECK-NEXT: [[SEL:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[CLASS_T]]* [[T0]] to i8*
  // CHECK-NEXT: [[T2:%.*]] = bitcast [1 x i8*]* [[OBJECTS]] to i8**
  // CHECK-NEXT: [[T3:%.*]] = call i8* bitcast ({{.*}} @objc_msgSend to {{.*}}(i8* [[T1]], i8* [[SEL]], i8** [[T2]], i64 1)
  // CHECK-NEXT: call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T3]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(i8* [[V1]])
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: store
  id arr = @[ b.prop ];

  // Release b.prop
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[B]]* [[V0]] to i8*
  // CHECK-NEXT: call void @llvm.objc.release(i8* [[T0]])

  // Destroy arr
  // CHECK: call void @llvm.objc.release

  // Destroy b
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: ret void
}
