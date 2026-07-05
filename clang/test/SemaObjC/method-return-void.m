// RUN: %clang_cc1 -Wmethod-signatures -fsyntax-only -verify -Wno-objc-root-class %s

@interface Test
- (int)foo;
@end

@implementation Test
- (int)foo { return; } // expected-error {{non-void method 'foo' should return a value}}
@end
