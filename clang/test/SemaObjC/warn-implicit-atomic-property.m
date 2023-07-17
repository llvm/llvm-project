// RUN: %clang_cc1 -fsyntax-only -Wimplicit-atomic-properties -verify -Wno-objc-root-class %s
// rdar://8774580

@interface Super
@property (nonatomic, readwrite) int P; // OK
@property (atomic, readwrite) int P1; // OK
@property (readwrite) int P2; // expected-note {{property declared here}}
@property int P3; // expected-note {{property declared here}}
@end

@implementation Super // expected-warning {{property is assumed atomic when auto-synthesizing the property}}
@synthesize P,P1,P2; // expected-warning {{property is assumed atomic by default}}
@end
