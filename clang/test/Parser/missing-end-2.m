// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: //7824372

@interface A // expected-note {{class started here}}
-(void) im0;

@implementation A // expected-error {{missing '@end'}}
@end

@interface B { // expected-note {{class started here}}
}

@implementation B // expected-error {{missing '@end'}}
@end

@interface C // expected-note 2 {{class started here}}
@property int P;

@implementation C // expected-error 2 {{missing '@end'}}
