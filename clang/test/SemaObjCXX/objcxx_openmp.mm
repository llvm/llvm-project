// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify %s
// expected-no-diagnostics

class Foo {
  int a;
};

@interface NSObject
@end

@interface Bar : NSObject {
  Foo *foo;
}
- (void)setSystemAndWindowCocoa:(class Foo *)foo_1;

@end

@implementation Bar : NSObject
- (void)setSystemAndWindowCocoa:(Foo *)foo_1 {
  foo = foo_1;
}
@end
