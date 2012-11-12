// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify -Wno-objc-root-class %s 

@interface I0 
@end

@implementation I0 // expected-note {{implementation started here}}
- meth { return 0; }

@interface I1 : I0 // expected-error {{missing '@end'}}
@end

@implementation I1 // expected-note {{implementation started here}}
-(void) im0 { self = [super init]; } // expected-warning {{not found}}

@interface I2 : I0 // expected-error {{missing '@end'}}
- I2meth;
@end

@implementation I2 // expected-note {{implementation started here}}
- I2meth { return 0; }

@implementation  I2(CAT) // expected-error 2 {{missing '@end'}} expected-note {{implementation started here}}
