// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface Super @end

@interface INTFSTANDALONE : Super
{
  id IVAR;	// expected-note {{previous definition is here}}
}

@end

@implementation INTFSTANDALONE : Super // expected-warning {{class implementation may not have super class}}
{
  id PRIV_IVAR;
@protected
  id PRTCTD;	
@private
  id IVAR3;
  int IVAR;	// expected-error {{instance variable is already declared}}
@public
  id IVAR4;
}
@end

@interface Base @end

@implementation Base { 
    int ivar1; 
@public
    int ivar2; 
} 
@end

id fn1(INTFSTANDALONE *b) { return b->PRIV_IVAR; } // expected-error {{instance variable 'PRIV_IVAR' is private}}

id fn2(INTFSTANDALONE *b) { return b->PRTCTD; }  // expected-error {{instance variable 'PRTCTD' is protected}}

id fn4(INTFSTANDALONE *b) { return b->IVAR4; }

