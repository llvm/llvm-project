// RUN: %clang_cc1 -triple i386-apple-darwin8 -analyze -analyzer-checker=core,alpha.core -analyzer-constraints=range -analyzer-store=region -verify -Wno-objc-root-class %s

// <rdar://problem/6888289> - This test case shows that a nil instance
// variable can possibly be initialized by a method.
@interface RDar6888289
{
  id *x;
}
- (void) test:(id) y;
- (void) test2:(id) y;
- (void) invalidate;
@end

id *getVal(void);

@implementation RDar6888289
- (void) test:(id)y {
  if (!x)
    [self invalidate];
  *x = y;
}
- (void) test2:(id)y {
  if (!x) {}
  *x = y; // expected-warning {{null}}
}

- (void) invalidate {
  x = getVal();
}

@end

