// RUN: %clang_cc1 -fsyntax-only -verify %s

// <rdar://problem/7605289>
@implementation Unknown (Blarg) // expected-error{{cannot find interface declaration for 'Unknown'}}
- (int)method { return ivar; } // expected-error{{use of undeclared identifier 'ivar'}}
@end
