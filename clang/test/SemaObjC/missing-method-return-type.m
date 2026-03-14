// RUN: %clang_cc1 -Wmissing-method-return-type -fsyntax-only -verify -Wno-objc-root-class %s

@interface I
-  initWithFoo:(id)foo; // expected-warning {{method has no return type specified; defaults to 'id'}}
@end

@implementation I
- initWithFoo:(id)foo { return 0; } // expected-warning {{method has no return type specified; defaults to 'id'}}
@end

