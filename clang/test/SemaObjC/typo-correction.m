// RUN: %clang_cc1 %s -verify -fsyntax-only -fobjc-runtime=ios

@protocol P
-(id)description;
@end

@interface B<P>
@property int x;
@end

@interface S : B {
  id _someivar; // expected-note {{here}}
}
@end

// Spell-checking 'undefined' is ok.
undefined var; // expected-error {{unknown type name}}

typedef int super1;
@implementation S
-(void)foo:(id)p1 other:(id)p2 {
  // Spell-checking 'super' is not ok.
  super.x = 0;
  self.x = 0;
}

-(void)test {
  [self foo:[super description] other:someivar]; // expected-error {{use of undeclared identifier 'someivar'; did you mean '_someivar'?}}
}
@end

__attribute__ (( __objc_root_class__ ))
@interface I {
  id _interface; // expected-note {{'_interface' declared here}}
}
-(void)method;
@end

@interface I () {
  id _extension; // expected-note {{'_extension' declared here}}
}
@end

@implementation I {
  id _implementation; // expected-note {{'_implementation' declared here}}
}
-(void)method {
  (void)self->implementation; // expected-error {{'I' does not have a member named 'implementation'; did you mean '_implementation'?}}
  (void)self->interface; // expected-error {{'I' does not have a member named 'interface'; did you mean '_interface'?}}
  (void)self->extension; // expected-error {{'I' does not have a member named 'extension'; did you mean '_extension'?}}
}
@end

// rdar://problem/33102722
// Typo correction for a property when it has as correction candidates
// synthesized ivar and a class name, both at the same edit distance.
@class TypoCandidate;

__attribute__ (( __objc_root_class__ ))
@interface PropertyType
@property int x;
@end

__attribute__ (( __objc_root_class__ ))
@interface InterfaceC
@property(assign) PropertyType *typoCandidate; // expected-note {{'_typoCandidate' declared here}}
@end

@implementation InterfaceC
-(void)method {
  typoCandidate.x = 0; // expected-error {{use of undeclared identifier 'typoCandidate'; did you mean '_typoCandidate'?}}
}
@end
