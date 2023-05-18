// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s

// This structure's size is not a power of two, so the property does
// not get native atomics, even though x86-64 can do unaligned atomics
// with a lock prefix.
struct s3 { char c[3]; };

// This structure's size is, so it does, because it can.
// FIXME: But we don't at the moment; the backend doesn't know how to generate
// correct code.
struct s4 { char c[4]; };

@interface Test0
@property struct s3 s3;
@property struct s4 s4;
@end
@implementation Test0
@synthesize s3, s4;
@end

// CHECK: define internal i24 @"\01-[Test0 s3]"(
// CHECK: call void @objc_copyStruct

// CHECK: define internal void @"\01-[Test0 setS3:]"(
// CHECK: call void @objc_copyStruct

// CHECK: define internal i32 @"\01-[Test0 s4]"(
// CHECK: call void @objc_copyStruct

// CHECK: define internal void @"\01-[Test0 setS4:]"(
// CHECK: call void @objc_copyStruct
