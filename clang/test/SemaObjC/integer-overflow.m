// RUN: %clang_cc1 -Wno-objc-root-class -fsyntax-only -verify %s

@interface Foo
@end

@implementation Foo
- (int)add:(int)a with:(int)b {
  return a + b;
}

- (void)testIntegerOverflows {
// expected-warning@+1 {{overflow in expression; result is 536'870'912 with type 'int'}}
  (void)[self add:0 with:4608 * 1024 * 1024];

// expected-warning@+1 {{overflow in expression; result is 536'870'912 with type 'int'}}
  (void)[self add:0 with:[self add:4608 * 1024 * 1024 with:0]];
}
@end
