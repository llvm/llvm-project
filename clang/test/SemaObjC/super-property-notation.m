// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface B
+(int) classGetter;
-(int) getter;
@end

@interface A : B
@end

@implementation A
+(int) classGetter {
  return 0;
}

+(int) classGetter2 {
  return super.classGetter;
}

-(void) method {
  int x = super.getter;
}
@end

void f0() {
  // FIXME: not implemented yet.
  //int l1 = A.classGetter;
  int l2 = [A classGetter2];
}

