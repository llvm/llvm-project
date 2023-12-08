// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
@interface Ivar
- (float*)method;
@end

@interface A {
  A *Ivar;
}
- (int*)method;
@end

@implementation A
- (int*)method {
  int *ip = [Ivar method]; // Okay; calls A's method on the instance variable Ivar.
                           // Note that Objective-C calls Ivar's method.
  return 0;
}
@end
