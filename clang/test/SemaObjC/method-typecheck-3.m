// RUN: %clang_cc1 -Wmethod-signatures -fsyntax-only -verify %s

@class B;
@interface A
- (B*)obj;
- (B*)a; // expected-note {{previous definition is here}}
- (void)takesA: (A*)a; // expected-note {{previous definition is here}}
- (void)takesId: (id)a; // expected-note {{previous definition is here}}
@end


@interface B : A
@end

@implementation B
- (id)obj {return self;} // 'id' overrides are permitted?
- (A*)a { return self;}  // expected-warning {{conflicting return type in implementation of 'a'}}
- (void)takesA: (B*)a  // expected-warning {{conflicting parameter types in implementation of 'takesA:'}}
{}
- (void)takesId: (B*)a // expected-warning {{conflicting parameter types in implementation of 'takesId:'}}
{}
@end
