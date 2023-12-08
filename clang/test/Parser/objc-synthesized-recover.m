// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface I1 
{
  int value;
  int value2;
}
@property int value;
@property int value2;
@end

@implementation I1
@synthesize value, - value2; // expected-error{{expected a property name}}
@synthesize value2;
@end
