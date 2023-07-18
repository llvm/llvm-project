// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -x objective-c %s -o - | FileCheck %s
// rdar://13192366
typedef signed char BOOL;
@interface NSObject 
{
  id isa;
}
@end

@interface MyClass : NSObject

@property (readwrite) BOOL boolean1;
@property (readwrite, copy) id object1;
@property (readwrite) BOOL boolean2;
@property (readwrite, copy) id object2;
@property (readwrite) BOOL boolean3;
@property (readwrite, copy) id object3;
@property (readwrite) BOOL boolean4;
@property (readwrite, copy) id object4;
@property (readwrite) BOOL boolean5;
@property (readwrite, copy) id object5;
@property (readwrite) BOOL boolean6;
@property (readwrite, copy) id object6;
@property (readwrite) BOOL boolean7;
@property (readwrite) BOOL MyBool;
@property (readwrite, copy) id object7;
@property (readwrite) BOOL boolean8;
@property (readwrite, copy) id object8;
@property (readwrite) BOOL boolean9;
@property (readwrite, copy) id object9;
@end

@implementation MyClass
{
  id MyIvar;
  BOOL _MyBool;
  char * pc;
}
@end

// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean1
// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean2
// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean3
// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean4
// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean5
// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean6
// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean7
// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean8
// CHECK: @{{.*}} = private unnamed_addr constant [10 x i8] c"_boolean9
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object1
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object2
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object3
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object4
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object5
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object6
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object7
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object8
// CHECK: @{{.*}} = private unnamed_addr constant [9 x i8] c"_object9
