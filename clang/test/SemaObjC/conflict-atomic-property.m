// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://10260017

@interface Foo 
@property (nonatomic, assign, atomic) float dummy; // expected-error {{property attributes 'atomic' and 'nonatomic' are mutually exclusive}}
@property (nonatomic, assign) float d1;
@property (atomic, assign) float d2;
@property (assign) float d3;
@property (atomic, nonatomic, assign) float d4; // expected-error {{property attributes 'atomic' and 'nonatomic' are mutually exclusive}}
@end
