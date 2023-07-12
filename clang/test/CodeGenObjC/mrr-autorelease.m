// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

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

// CHECK-NOT: call ptr @objc_getClass
// CHECK: call ptr @objc_msgSend
// CHECK: call ptr @objc_msgSend
// CHECK: call void @objc_msgSend
