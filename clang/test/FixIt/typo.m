// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin10 -DNON_FIXITS -verify -Wno-objc-root-class %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x objective-c -fsyntax-only -triple x86_64-apple-darwin10 -fixit -Wno-objc-root-class %t
// RUN: %clang_cc1 -x objective-c -fsyntax-only -triple x86_64-apple-darwin10 -pedantic -Werror -Wno-objc-root-class %t
// RUN: grep "@implementation Sub3" %t

@interface NSString // expected-note 2{{'NSString' declared here}}
+ (int)method:(int)x;
@end

void test() {
  NSstring *str = @"A string"; // expected-error{{unknown type name 'NSstring'; did you mean 'NSString'?}}
}

@protocol P1
@optional
@property int *sprop; // expected-note{{'sprop' declared here}}
@end

@interface A
{
  int his_ivar; // expected-note 2{{'his_ivar' declared here}}
  float wibble;
}
- (void)methodA;
+ (void)methodA;
@property int his_prop; // expected-note{{'his_prop' declared here}}
@end

@interface B : A <P1>
{
  int her_ivar; // expected-note 2{{'her_ivar' declared here}}
}

@property int her_prop; // expected-note{{'her_prop' declared here}}
- (void)inst_method1:(int)a;
+ (void)class_method1;
@end

@implementation A
@synthesize his_prop = his_ivar;
- (void)methodA { }
+ (void)methodA { }
@end

@implementation B
@synthesize her_prop = her_ivar;

-(void)inst_method1:(int)a {
  herivar = a; // expected-error{{use of undeclared identifier 'herivar'; did you mean 'her_ivar'?}}
  hisivar = a; // expected-error{{use of undeclared identifier 'hisivar'; did you mean 'his_ivar'?}}
  self->herivar = a; // expected-error{{'B' does not have a member named 'herivar'; did you mean 'her_ivar'?}}
  self->hisivar = a; // expected-error{{'B' does not have a member named 'hisivar'; did you mean 'his_ivar'?}}
  self.hisprop = 0; // expected-error{{property 'hisprop' not found on object of type 'B *'; did you mean 'his_prop'?}}
  self.herprop = 0; // expected-error{{property 'herprop' not found on object of type 'B *'; did you mean 'her_prop'?}}
  self.s_prop = 0; // expected-error{{property 's_prop' not found on object of type 'B *'; did you mean 'sprop'?}}
}

+(void)class_method1 {
}
@end

void test_message_send(B* b) {
  [NSstring method:17]; // expected-error{{unknown receiver 'NSstring'; did you mean 'NSString'?}}
}

@interface Collide // expected-note{{'Collide' declared here}}
{
@public
  int value; // expected-note{{'value' declared here}}
}

@property int value; // expected-note{{'value' declared here}}
@end

@implementation Collide
@synthesize value = value;
@end

void test2(Collide *a) {
  a.valu = 17; // expected-error{{property 'valu' not found on object of type 'Collide *'; did you mean 'value'?}}
  a->vale = 17; // expected-error{{'Collide' does not have a member named 'vale'; did you mean 'value'?}}
}

#ifdef NON_FIXITS
@interface Derived : Collid // expected-error{{cannot find interface declaration for 'Collid', superclass of 'Derived'; did you mean 'Collide'?}}
@end
#endif

#ifdef NON_FIXITS
@protocol NetworkSocket // expected-note{{'NetworkSocket' declared here}}
- (int)send:(void*)buffer bytes:(int)bytes;
@end

@interface IPv6 <Network_Socket> // expected-error{{cannot find protocol declaration for 'Network_Socket'; did you mean 'NetworkSocket'?}}
@end
#endif

@interface Super
- (int)method; // expected-note{{using}}
- (int)method2;
- (int)method3:(id)x;
@end

@interface Sub : Super
- (int)method;
@end

@implementation Sub
- (int)method {
  return [supper method]; // expected-error{{unknown receiver 'supper'; did you mean 'super'?}}
}
  
@end

double *isupper(int);

@interface Sub2 : Super
- (int)method2;
@end

@implementation Sub2
- (int)method2 {
  return [supper method2]; // expected-error{{unknown receiver 'supper'; did you mean 'super'?}}
}
@end

@interface Ivar
@end

@protocol Proto
@property (retain) id ivar;
@end

#ifdef NON_FIXITS
@interface User <Proto>
- (void)method; // expected-note{{also found}}
@end

@implementation User
@synthesize ivar;

- (void)method {
  // Test that we don't correct 'ivar' to 'Ivar'  e
  [ivar method]; // expected-warning{{multiple methods named 'method' found}}
}
@end
#endif

void f(A *a) {
  f(a) // expected-error{{expected ';' after expression}}
  [a methodA] // expected-error{{expected ';' after expression}}
  [A methodA] // expected-error{{expected ';' after expression}}
}

#ifdef NON_FIXITS
@interface Sub3 : Super
- (int)method3;
@end

@implementation Sub3
- (int)method3 {
  int x = super; // expected-error{{use of undeclared identifier 'super'}}
  return 0;
}
@end
#endif
