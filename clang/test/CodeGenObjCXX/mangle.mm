// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: @"_ZZ11+[A shared]E1a" = internal global
// CHECK: @"_ZZ11-[A(Foo) f]E1a" = internal global
// CHECK: v56@0:8i16i20i24i28i32i36i40i44^i48

@interface A
@end

@implementation A

+ (A *)shared {
  static A* a;
  
  return a;
}

@end

@interface A(Foo)
@end

@implementation A(Foo)
- (int)f {
  // FIXME: Add a member function to s and make sure that it's mangled correctly.
  struct s {
  };
  
  static s a;

  return 0;
}
@end

// PR6468
@interface Test
- (void) process: (int)r3 :(int)r4 :(int)r5 :(int)r6 :(int)r7 :(int)r8 :(int)r9 :(int)r10 :(int &)i;
@end

@implementation Test
- (void) process: (int)r3 :(int)r4 :(int)r5 :(int)r6 :(int)r7 :(int)r8 :(int)r9 :(int)r10 :(int &)i {
}
@end

// rdar://9566314
@interface NX
- (void)Meth;
@end

@implementation NX
- (void)Meth {
  void uiIsVisible();
// CHECK: call void @_Z11uiIsVisiblev
  uiIsVisible();
}
@end
