// RUN: %clang_cc1 -fsyntax-only -verify %s

@implementation Unknown (Blarg) // expected-error{{cannot find interface declaration for 'Unknown'}}
- (int)method { return ivar; } // expected-error{{use of undeclared identifier 'ivar'}}
@end
