// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s 
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s 
// expected-no-diagnostics

@interface Singleton {
}
+ (Singleton*) instance;
@end

@implementation Singleton

- (void) someSelector { }

+ (Singleton*) instance { return 0; }

+ (void) compileError
{
     [Singleton.instance  someSelector]; // clang issues error here
}

@end

