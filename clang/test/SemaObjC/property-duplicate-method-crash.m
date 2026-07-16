// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class -Wno-objc-property-no-attribute %s

@interface I
@property (atomic) id atomic_prop; // expected-note {{property declared here}}
@end

@implementation I // expected-note {{implementation started here}}
@synthesize atomic_prop, atomic_prop1; // expected-error {{property implementation must have its declaration in interface 'I' or one of its extensions}}

- (id) atomic_prop { return 0; } // expected-note {{previous declaration is here}} \
                                 // expected-warning {{writable atomic property 'atomic_prop' cannot pair a synthesized setter with a user defined getter}} \
                                 // expected-note {{setter and getter must both be synthesized, or both be user defined, or the property must be nonatomic}}

- (id) atomic_prop; // expected-error {{duplicate declaration of method 'atomic_prop'}}

@end // expected-error {{expected method body}} \
     // expected-error@17 {{missing '@end'}}
