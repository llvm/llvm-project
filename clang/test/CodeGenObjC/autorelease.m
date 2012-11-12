// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-runtime=macosx-10.7 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm -fobjc-runtime=macosx-10.7 -o - %s | FileCheck %s
// rdar://8881826
// rdar://9412038

@interface I
{
  id ivar;
}
- (id) Meth;
+ (id) MyAlloc;;
@end

@implementation I
- (id) Meth {
   @autoreleasepool {
      id p = [I MyAlloc];
      if (!p)
        return ivar;
   }
  return 0;
}
+ (id) MyAlloc {
    return 0;
}
@end

// CHECK: call i8* @objc_autoreleasePoolPush
// CHECK: [[T:%.*]] = load i8** [[A:%.*]]
// CHECK: call void @objc_autoreleasePoolPop
