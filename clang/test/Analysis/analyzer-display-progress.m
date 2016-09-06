// RUN: %clang_cc1 -fblocks -analyze -analyzer-display-progress %s 2>&1 | FileCheck %s

#include "Inputs/system-header-simulator-objc.h"

static void f() {}

@interface I: NSObject
-(void)instanceMethod:(int)arg1 with:(int)arg2;
+(void)classMethod;
@end

@implementation I
-(void)instanceMethod:(int)arg1 with:(int)arg2 {}
+(void)classMethod {}
@end

void g(I *i, int x, int y) {
  [I classMethod];
  [i instanceMethod: x with: y];

  void (^block)(void);
  block = ^{};
  block();
}

// CHECK: analyzer-display-progress.m f
// CHECK: analyzer-display-progress.m -[I instanceMethod:with:]
// CHECK: analyzer-display-progress.m +[I classMethod]
// CHECK: analyzer-display-progress.m g
// CHECK: analyzer-display-progress.m block (line: 22, col: 11)
