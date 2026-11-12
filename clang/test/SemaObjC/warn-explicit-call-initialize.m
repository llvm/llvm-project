// RUN: %clang_cc1  -fsyntax-only  -triple x86_64-apple-darwin10 -verify %s

@interface NSObject
+ (void)initialize; // expected-note 2 {{method 'initialize' declared here}}
@end

@interface I : NSObject 
+ (void)initialize; // expected-note {{method 'initialize' declared here}}
+ (void)SomeRandomMethod;
@end

@implementation I
- (void) Meth { 
  [I initialize];     // expected-warning {{explicit call to +initialize results in duplicate call to +initialize}} 
  [NSObject initialize]; // expected-warning {{explicit call to +initialize results in duplicate call to +initialize}}
}
+ (void)initialize {
  [super initialize];
}
+ (void)SomeRandomMethod { // expected-note {{method 'SomeRandomMethod' declared here}}
  [super initialize]; // expected-warning {{explicit call to [super initialize] should only be in implementation of +initialize}}
}
@end

