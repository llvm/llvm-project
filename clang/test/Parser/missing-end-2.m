// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify %s

@interface A // expected-note {{class started here}}
-(void) im0; // expected-note {{method 'im0' declared here}}

// expected-warning@+1 {{method definition for 'im0' not found}}
@implementation A // expected-error {{missing '@end'}}
@end

@interface B { // expected-note {{class started here}}
}

@implementation B // expected-error {{missing '@end'}}
@end

@interface C // expected-note 1 {{class started here}}
@property int P;

// expected-note@+1 {{implementation started here}}
@implementation C // expected-error 2 {{missing '@end'}}
