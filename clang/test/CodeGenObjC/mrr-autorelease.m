// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s
// rdar://8881826
// rdar://9423507

@interface I
{
  id ivar;
}
- (id) Meth;
@end

@implementation I
- (id) Meth {
   @autoreleasepool {
   }
  return 0;
}
@end

// CHECK-NOT: call i8* @objc_getClass
// CHECK: call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
