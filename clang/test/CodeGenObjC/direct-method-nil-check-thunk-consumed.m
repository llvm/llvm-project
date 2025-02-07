// REQUIRES: system-darwin

// RUN: %clang -fobjc-export-direct-methods   \
// RUN: -target arm64-apple-darwin -fobjc-arc \
// RUN: -O0 -S -emit-llvm %s -o - | FileCheck %s

#import <Foundation/Foundation.h>

@interface Shape: NSObject
@property(direct) int x;
@property(direct) int y;
- (instancetype)initWithX:(int)x Y:(int)y __attribute__((objc_direct));
- (void) move: (Shape *) __attribute__((ns_consumed)) s __attribute__((objc_direct));
+ (Shape*) default __attribute__((objc_direct));
@end

@implementation Shape
- (instancetype)initWithX:(int)x Y:(int)y  {
  if (self = [super init]) {
    _x = x;
    _y = y;
  }
  return self;
}

// Inner function should
//  1. Call inner set (because we alreaday know self is not null)
//  2. Call thunk get (because we don't know if s is null)
//  3. Release s.
// CHECK-LABEL: define dso_local void @"-<Shape move:>_inner"
// CHECK: {{.*}} = call i32 @"-<Shape x>"
// CHECK: call void @"-<Shape setX:>_inner"
// CHECK: {{.*}} = call i32 @"-<Shape y>"
// CHECK: call void @"-<Shape setY:>_inner"
// CHECK: call void @llvm.objc.storeStrong
// CHECK-LABEL: }

// Outer function should not release anything.
// CHECK-LABEL: define dso_local void @"-<Shape move:>"
// CHECK-NOT: call void @llvm.objc.storeStrong
// CHECK-LABEL: }
- (void) move: (Shape *) s {
  self.x = s.x;
  self.y = s.y;
}

+ (Shape*) default {
  return [[Shape alloc] initWithX:1 Y:1];
}
@end

int main() {
  Shape *s = [Shape default];
  Shape *t = nil;
  [t move:s];
}
