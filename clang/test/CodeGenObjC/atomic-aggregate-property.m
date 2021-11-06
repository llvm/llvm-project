// RUN: %clang_cc1 -disable-noundef-analysis -triple x86_64-apple-darwin10  -fobjc-gc -emit-llvm -o - %s | FileCheck -check-prefix CHECK-LP64 %s
// RUN: %clang_cc1 -disable-noundef-analysis -x objective-c++ -triple x86_64-apple-darwin10  -fobjc-gc -emit-llvm -o - %s | FileCheck -check-prefix CHECK-LP64 %s
// rdar: // 7849824
// <rdar://problem/12547611>

struct s {
  double a, b, c, d;  
};

struct s1 {
    int i;
    id j;
    id k;
};

struct s2 {};

@interface A 
@property (readwrite) double x;
@property (readwrite) struct s y;
@property (nonatomic, readwrite) struct s1 z;
@property (readwrite) struct s2 a;
@end

@implementation A
@synthesize x;
@synthesize y;
@synthesize z;
@synthesize a;
@end
// CHECK-LP64: define internal double @"\01-[A x]"(
// CHECK-LP64: load atomic i64, i64* {{%.*}} unordered, align 8

// CHECK-LP64: define internal void @"\01-[A setX:]"(
// CHECK-LP64: store atomic i64 {{%.*}}, i64* {{%.*}} unordered, align 8

// CHECK-LP64: define internal void @"\01-[A y]"(
// CHECK-LP64: call void @objc_copyStruct(i8* {{%.*}}, i8* {{%.*}}, i64 32, i1 zeroext true, i1 zeroext false)

// CHECK-LP64: define internal void @"\01-[A setY:]"(
// CHECK-LP64: call void @objc_copyStruct(i8* {{%.*}}, i8* {{%.*}}, i64 32, i1 zeroext true, i1 zeroext false)

// CHECK-LP64: define internal void @"\01-[A z]"(
// CHECK-LP64: call i8* @objc_memmove_collectable(

// CHECK-LP64: define internal void @"\01-[A setZ:]"(
// CHECK-LP64: call i8* @objc_memmove_collectable(

// CHECK-LP64: define internal void @"\01-[A a]"(
// (do nothing)

// CHECK-LP64: define internal void @"\01-[A setA:]"(
// (do nothing)
