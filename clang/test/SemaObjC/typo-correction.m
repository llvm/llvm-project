// RUN: %clang_cc1 %s -verify -fsyntax-only

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
