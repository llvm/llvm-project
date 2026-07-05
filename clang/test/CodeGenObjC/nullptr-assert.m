// RUN: %clang_cc1 -Wno-objc-root-class -o /dev/null -triple x86_64-- -emit-llvm %s
// REQUIRES: asserts
// Verify there is no assertion.

@interface A
@end

extern A *a;

@interface X
@end

@implementation X

-(void)test {
  struct S {
    A *a;
    int b;
  };
  struct S s[] = {{a, 0}, {(void *)0, 0}};
}
@end
