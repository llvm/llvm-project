// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -analyzer-store=region -verify -Wno-objc-root-class %s
// expected-no-diagnostics

typedef struct Foo { int x; } Bar;

@interface MyClass {}
- (Bar)foo;
@end
@implementation MyClass
- (Bar)foo { 
  struct Foo f = { 0 };
  return f;
}
@end

void createFoo(void) {
  MyClass *obj = 0;  
  Bar f = [obj foo]; // no-warning
}

void createFoo2(void) {
  MyClass *obj = 0;  
  [obj foo]; // no-warning
  Bar f = [obj foo]; // no-warning
}

