// RUN: not %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -Wno-objc-root-class %s 2>&1 | FileCheck --check-prefix=CHECK-ARC %s
// rdar://9829425

// RUN: not %clang_cc1 -fsyntax-only -fblocks -Wno-objc-root-class %s 2>&1 | FileCheck %s
// rdar://11761511

extern void doSomething(void);

@interface Test
{
@public
  void (^aBlock)(void);
}
@property (retain) void (^aBlock)(void);
@property (weak, retain) void (^aBlockW)(void);
@property (strong, retain) void (^aBlockS)(void); // OK
@property (readonly, retain) void (^aBlockR)(void); // OK
@property (copy, retain) void (^aBlockC)(void);
@property (assign, retain) void (^aBlockA)(void);
@end

@implementation Test
@synthesize aBlock;
@dynamic aBlockW, aBlockS, aBlockR, aBlockC, aBlockA;
@end

int main(void) {
  Test *t;
  t.aBlock = ^{ doSomething(); };
  t.aBlockW = ^{ doSomething(); };
  t.aBlockS = ^{ doSomething(); };
}

// CHECK-ARC: 14:1: warning: retain'ed block property does not copy the block - use copy attribute instead
// CHECK-ARC: @property (retain) void (^aBlock)(void);
// CHECK-ARC: ^
// CHECK-ARC: 15:1: error: property attributes 'retain' and 'weak' are mutually exclusive
// CHECK-ARC: @property (weak, retain) void (^aBlockW)(void);
// CHECK-ARC: ^
// CHECK-ARC: 18:1: error: property attributes 'copy' and 'retain' are mutually exclusive
// CHECK-ARC: @property (copy, retain) void (^aBlockC)(void);
// CHECK-ARC: ^
// CHECK-ARC: 19:1: error: property attributes 'assign' and 'retain' are mutually exclusive
// CHECK-ARC: @property (assign, retain) void (^aBlockA)(void);
// CHECK-ARC: ^
// CHECK-ARC: 30:13: warning: assigning block literal to a weak property; object will be released after assignment
// CHECK-ARC:   t.aBlockW = ^{ doSomething(); };
// CHECK-ARC:             ^ ~~~~~~~~~~~~~~~~~~~
// CHECK-ARC: 2 warnings and 3 errors generated.

// CHECK: 14:1: warning: retain'ed block property does not copy the block - use copy attribute instead
// CHECK: @property (retain) void (^aBlock)(void);
// CHECK: ^
// CHECK: 15:1: error: property attributes 'retain' and 'weak' are mutually exclusive
// CHECK: @property (weak, retain) void (^aBlockW)(void);
// CHECK: ^
// CHECK: 18:1: error: property attributes 'copy' and 'retain' are mutually exclusive
// CHECK: @property (copy, retain) void (^aBlockC)(void);
// CHECK: ^
// CHECK: 19:1: error: property attributes 'assign' and 'retain' are mutually exclusive
// CHECK: @property (assign, retain) void (^aBlockA)(void);
// CHECK: ^
// CHECK: 1 warning and 3 errors generated.
