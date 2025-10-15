// RUN: %clang_cc1 -triple arm-apple-ios -emit-llvm -debug-info-kind=limited -fblocks  %s -o - | FileCheck %s
// Objective-C code cargo-culted from debug-info-lifetime-crash.m.
@protocol NSObject
- (id)copy;
@end
@class W;
@interface View1
@end
@implementation Controller {
    void (^Block)(void);
}
- (void)View:(View1 *)View foo:(W *)W
{
  // The reference from inside the block implicitly creates another
  // local variable for the referenced member. That is what gets
  // suppressed by the attribute.  It still gets debug info as a
  // member, though.
  // CHECK-NOT: !DILocalVariable(name: "weakSelf"
  // CHECK:     !DIDerivedType({{.*}} name: "weakSelf"
  // CHECK-NOT: !DILocalVariable(name: "weakSelf"
  __attribute__((nodebug)) __typeof(self) weakSelf = self;
  Block = [^{
    __typeof(self) strongSelf = weakSelf;
    } copy];
}
@end
