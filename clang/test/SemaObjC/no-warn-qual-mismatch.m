// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// radar 7211563

@interface X

+ (void)prototypeWithScalar:(int)aParameter;
+ (void)prototypeWithPointer:(void *)aParameter;

@end

@implementation X

+ (void)prototypeWithScalar:(const int)aParameter {}
+ (void)prototypeWithPointer:(void * const)aParameter {}

@end
