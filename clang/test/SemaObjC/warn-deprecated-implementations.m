// RUN: %clang_cc1 -triple=x86_64-apple-macos10.10 -fsyntax-only -Wdeprecated-implementations -verify -Wno-objc-root-class %s

@protocol P
- (void) D __attribute__((deprecated)); // expected-note {{method 'D' declared here}}

- (void) unavailable __attribute__((__unavailable__)); // expected-note {{method 'unavailable' declared here}}
@end

@interface A <P>
+ (void)F __attribute__((deprecated));
@end

@interface A()
- (void) E __attribute__((deprecated));
@end

@implementation A
+ (void)F { }	// No warning, implementing its own deprecated method
- (void) D {} //  expected-warning {{implementing deprecated method}}
- (void) E {} // No warning, implementing deprecated method in its class extension.

- (void) unavailable { } // expected-warning {{implementing unavailable metho}}
@end

@interface A(CAT)
- (void) G __attribute__((deprecated)); 
@end

@implementation A(CAT)
- (void) G {} 	// No warning, implementing its own deprecated method
@end

__attribute__((deprecated)) // expected-note {{'CL' has been explicitly marked deprecated here}}
@interface CL // expected-note 2 {{class declared here}} 
@end

@implementation CL // expected-warning {{implementing deprecated class}}
@end

@implementation CL (SomeCategory) // expected-warning {{implementing deprecated category}}
@end

@interface CL_SUB : CL // expected-warning {{'CL' is deprecated}}
@end

@interface BASE
- (void) B __attribute__((deprecated)); // expected-note {{method 'B' declared here}}

+ (void) unavailable __attribute__((availability(macos, unavailable))); // expected-note {{method 'unavailable' declared here}}
@end

@interface SUB : BASE
@end

@implementation SUB
- (void) B {} // expected-warning {{implementing deprecated method}}
+ (void) unavailable { } // expected-warning {{implementing unavailable method}}
@end

@interface Test
@end

@interface Test()
- (id)initSpecialInPrivateHeader __attribute__((deprecated));
@end

@implementation Test
- (id)initSpecialInPrivateHeader {
  return (void *)0;
}
@end

__attribute__((deprecated))
@interface Test(DeprecatedCategory) // expected-note {{category declared here}}
@end

@implementation Test(DeprecatedCategory) // expected-warning {{implementing deprecated category}}
@end
