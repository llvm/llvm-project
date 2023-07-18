// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://10177744

@interface Foo
@property (nonatomic, retain) NSString* what; // expected-error {{unknown type name 'NSString'}} \
                                              // expected-error {{property with}} \
                                              // expected-note {{previous definition is here}}
@end
 
@implementation Foo
- (void) setWhat: (NSString*) value { // expected-error {{expected a type}} \
                                      // expected-warning {{conflicting parameter types in implementation of}}
  __what; // expected-error {{use of undeclared identifier}} \
          // expected-warning {{expression result unused}}
}
@synthesize what; // expected-note {{'what' declared here}}
@end

@implementation Bar // expected-warning {{cannot find interface declaration for}}
- (NSString*) what { // expected-error {{expected a type}}
  return __what; // expected-error {{use of undeclared identifier}}
}
@end
