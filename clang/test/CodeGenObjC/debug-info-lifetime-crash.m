// RUN: %clang_cc1 -triple arm-apple-ios -emit-llvm -debug-info-kind=limited -fblocks -fobjc-runtime=ios-7.0.0 -fobjc-arc %s -o - | FileCheck %s
// rdar://problem/14990656
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
  // The debug type for these two will be identical, because we do not
  // actually emit the ownership qualifier.
  // CHECK: !DILocalVariable(name: "weakSelf",
  // CHECK-SAME:             line: [[@LINE+2]]
  // CHECK-SAME:             type: ![[SELFTY:[0-9]+]]
  __attribute__((objc_ownership(weak))) __typeof(self) weakSelf = self;
  Block = [^{
  // CHECK: !DILocalVariable(name: "strongSelf",
  // CHECK-SAME:             line: [[@LINE+2]]
  // CHECK-SAME:             type: ![[SELFTY]]
      __attribute__((objc_ownership(strong))) __typeof(self) strongSelf = weakSelf;
    } copy];
}
@end
