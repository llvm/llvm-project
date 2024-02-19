// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// Allow injection of ivars into implementation's implicit class.

@implementation INTFSTANDALONE // expected-warning {{cannot find interface declaration for 'INTFSTANDALONE'}}
{
  id IVAR1;
  id IVAR2;
}
- (id) Meth { return IVAR1; }
@end

