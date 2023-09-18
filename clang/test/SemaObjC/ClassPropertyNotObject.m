// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@protocol P @end

@interface I
@property Class<P> MyClass;
@property Class MyClass1;
@property void * VOIDSTAR;
@end

@implementation I
@synthesize MyClass, MyClass1, VOIDSTAR;
@end
