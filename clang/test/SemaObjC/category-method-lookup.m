// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface Foo
@end
@implementation Foo
@end

@implementation Foo(Whatever)
+(float)returnsFloat { return 7.0; }
@end

@interface Foo (MoreStuff)
+(int)returnsInt;
@end

@implementation Foo (MoreStuff)
+(int)returnsInt {
  return 0;
}

+(void)returnsNothing {
}
-(int)callsReturnsInt {
  float f = [Foo returnsFloat]; // GCC doesn't find this method (which is a bug IMHO).
  [Foo returnsNothing];
  return [Foo returnsInt];
}
@end

int main() {return 0;}

