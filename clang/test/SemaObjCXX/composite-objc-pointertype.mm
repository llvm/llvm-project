// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface Foo
@end

@implementation Foo
- (id)test {
        id bar;
    Class cl;
    Foo *f;

    (void)((bar!= 0) ? bar : 0);
    (void)((cl != 0) ? cl : 0);
    (void)((f != 0) ? 0 : f);
    return (0 == 1) ? 0 : bar;
}
@end

@class A;

void multilevel() {
  A **p;
  const A **q;

  using T = decltype(true ? q : p);
  using T = const A * const *;
}

