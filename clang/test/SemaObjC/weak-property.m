// RUN: %clang_cc1  -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -verify -Wno-objc-root-class %s
// rdar://8899430

@interface WeakPropertyTest {
    Class isa;
    __weak id value;
    id x;
}
@property (weak) id value1;
@property __weak id value;
@property () __weak id value2;

@property (weak, assign) id v1;  // expected-error {{property attributes 'assign' and 'weak' are mutually exclusive}}
@property (weak, copy) id v2; // expected-error {{property attributes 'copy' and 'weak' are mutually exclusive}}
@property (weak, retain) id v3; // expected-error {{property attributes 'retain' and 'weak' are mutually exclusive}}
@property (weak, assign) id v4;  // expected-error {{property attributes 'assign' and 'weak' are mutually exclusive}}

@property () __weak id x; // expected-note {{property declared here}}
@end

@implementation WeakPropertyTest
@synthesize x;	// expected-error {{existing instance variable 'x' for __weak property 'x' must be __weak}}
@dynamic value1, value, value2, v1,v2,v3,v4;
@end
