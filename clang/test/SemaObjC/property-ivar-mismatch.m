// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// Test that arithmatic types on property and its ivar have exact match.

@interface Test4 
{
   char ivar; // expected-note{{instance variable is declared here}}
}
@property int prop;
@end

@implementation Test4
@synthesize prop = ivar;  // expected-error {{type of property 'prop' ('int') does not match type of instance variable 'ivar' ('char')}}
@end


@interface Test5
{
  void * _P; // expected-note {{instance variable is declared here}}
}
@property int P;
@end

@implementation Test5
@synthesize P=_P; // expected-error {{ype of property 'P' ('int') does not match type of instance variable '_P' ('void *')}}
@end

