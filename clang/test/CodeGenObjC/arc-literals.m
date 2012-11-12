// RUN: %clang_cc1 -I %S/Inputs -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-optzns -o - %s | FileCheck %s

#include "literal-support.h"

// Check the various selector names we'll be using, in order.

// CHECK: c"numberWithInt:\00"
// CHECK: c"numberWithUnsignedInt:\00"
// CHECK: c"numberWithUnsignedLongLong:\00"
// CHECK: c"numberWithChar:\00"
// CHECK: c"arrayWithObjects:count:\00"
// CHECK: c"dictionaryWithObjects:forKeys:count:\00"
// CHECK: c"prop\00"

// CHECK: define void @test_numeric()
void test_numeric() {
  // CHECK: {{call.*objc_msgSend.*i32 17}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id ilit = @17;
  // CHECK: {{call.*objc_msgSend.*i32 25}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id ulit = @25u;
  // CHECK: {{call.*objc_msgSend.*i64 42}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id ulllit = @42ull;
  // CHECK: {{call.*objc_msgSend.*i8 signext 97}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id charlit = @'a';
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void
}

// CHECK: define void @test_array
void test_array(id a, id b) {
  // Retaining parameters
  // CHECK: call i8* @objc_retain(i8*
  // CHECK: call i8* @objc_retain(i8*

  // Constructing the array
  // CHECK: getelementptr inbounds [2 x i8*]* [[OBJECTS:%[A-Za-z0-9]+]], i32 0, i32 0
  // CHECK: store i8*
  // CHECK: getelementptr inbounds [2 x i8*]* [[OBJECTS]], i32 0, i32 1
  // CHECK: store i8*

  // CHECK: {{call i8*.*objc_msgSend.*i64 2}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id arr = @[a, b];

  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void
}

// CHECK: define void @test_dictionary
void test_dictionary(id k1, id o1, id k2, id o2) {
  // Retaining parameters
  // CHECK: call i8* @objc_retain(i8*
  // CHECK: call i8* @objc_retain(i8*
  // CHECK: call i8* @objc_retain(i8*
  // CHECK: call i8* @objc_retain(i8*

  // Constructing the arrays
  // CHECK: getelementptr inbounds [2 x i8*]* [[KEYS:%[A-Za-z0-9]+]], i32 0, i32 0
  // CHECK: store i8*
  // CHECK: getelementptr inbounds [2 x i8*]* [[OBJECTS:%[A-Za-z0-9]+]], i32 0, i32 0
  // CHECK: store i8*
  // CHECK: getelementptr inbounds [2 x i8*]* [[KEYS]], i32 0, i32 1
  // CHECK: store i8*
  // CHECK: getelementptr inbounds [2 x i8*]* [[OBJECTS]], i32 0, i32 1
  // CHECK: store i8*

  // Constructing the dictionary
  // CHECK: {{call i8.*@objc_msgSend}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id dict = @{ k1 : o1, k2 : o2 };

  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void
}

@interface A
@end

@interface B
@property (retain) A* prop;
@end

// CHECK: define void @test_property
void test_property(B *b) {
  // Retain parameter
  // CHECK: call i8* @objc_retain

  // Invoke 'prop'
  // CHECK: load i8** @"\01L_OBJC_SELECTOR_REFERENCES
  // CHECK: {{call.*@objc_msgSend}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue

  // Invoke arrayWithObjects:count:
  // CHECK: load i8** @"\01L_OBJC_SELECTOR_REFERENCES
  // CHECK: {{call.*objc_msgSend}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id arr = @[ b.prop ];

  // Release b.prop
  // CHECK: call void @objc_release

  // Destroy arr
  // CHECK: call void @objc_release

  // Destroy b
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void
}
