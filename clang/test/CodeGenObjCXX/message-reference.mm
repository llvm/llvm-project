// RUN: %clang_cc1 -no-opaque-pointers -x objective-c++ -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s
// rdar://8604515

@interface I {}
-(unsigned int&)referenceCount;
@end

@interface MyClass
+(int)writeBlip:(I*)srcBlip;
@end

@implementation MyClass
+(int)writeBlip:(I*)srcBlip{
  return ([srcBlip referenceCount] == 0);
}
@end

// CHECK: [[T:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i32* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: [[U:%.*]] = load i32, i32* [[T]]
// CHECK: [[V:%.*]] = icmp eq i32 [[U]], 0
