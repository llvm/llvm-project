// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface MyParent {
  int X;
}
@end
@implementation MyParent
@end

@interface MyParent(AA) {
}
@end
@implementation MyParent (AA)
- (void) setX: (int)in {X = in - 2;}
- (int) X {return X;}
@end

@interface MyClass : MyParent
@end
@implementation MyClass
@end

int foo(MyClass *o) {
  o.X = 2;
  return o.X;
}