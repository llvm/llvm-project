// RUN: %clang_cc1 -triple i386-apple-macosx10.10 -fobjc-arc -fsyntax-only -Wno-objc-root-class %s -verify -disable-free

@class Dictionary;

@interface Test
@end
@implementation Test
// rdar://problem/47403222
- (void)rdar47403222:(Dictionary *)opts {
  [self undeclaredMethod:undeclaredArg];
  // expected-error@-1{{no visible @interface for 'Test' declares the selector 'undeclaredMethod:'}}
  opts[(__bridge id)undeclaredKey] = 0;
  // expected-error@-1{{use of undeclared identifier 'undeclaredKey'}}
}
@end
