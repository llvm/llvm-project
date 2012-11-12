// RUN: %clang_cc1 -fsyntax-only -Wdeprecated-implementations -verify -Wno-objc-root-class %s
// rdar://8973810

@protocol P
- (void) D __attribute__((deprecated)); // expected-note {{method 'D' declared here}}
@end

@interface A <P>
+ (void)F __attribute__((deprecated)); // expected-note {{method 'F' declared here}}
@end

@interface A()
- (void) E __attribute__((deprecated)); // expected-note {{method 'E' declared here}}
@end

@implementation A
+ (void)F { } //  expected-warning {{Implementing deprecated method}}
- (void) D {} //  expected-warning {{Implementing deprecated method}}
- (void) E {} //  expected-warning {{Implementing deprecated method}}
@end

__attribute__((deprecated))
@interface CL // expected-note 2 {{class declared here}} // expected-note 2 {{declared here}}
@end

@implementation CL // expected-warning {{Implementing deprecated class}}
@end

@implementation CL ( SomeCategory ) // expected-warning {{'CL' is deprecated}} \
                                    // expected-warning {{Implementing deprecated category}}
@end

@interface CL_SUB : CL // expected-warning {{'CL' is deprecated}}
@end

@interface BASE
- (void) B __attribute__((deprecated)); // expected-note {{method 'B' declared here}}
@end

@interface SUB : BASE
@end

@implementation SUB
- (void) B {} // expected-warning {{Implementing deprecated method}}
@end

