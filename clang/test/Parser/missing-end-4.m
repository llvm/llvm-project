// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface X1
@end
@implementation X1 // expected-note {{implementation started here}}
@interface Y1 // expected-error {{missing '@end'}}
@end
@end // expected-error {{'@end' must appear in an Objective-C context}}

@interface X2
@end
@implementation X2 // expected-note {{implementation started here}}
@protocol Y2 // expected-error {{missing '@end'}}
@end
@end // expected-error {{'@end' must appear in an Objective-C context}}

@interface X6 // expected-note {{class started here}}
@interface X7 // expected-error {{missing '@end'}}
@end
@end // expected-error {{'@end' must appear in an Objective-C context}}

@protocol P1 // expected-note {{protocol started here}}
@interface P2 // expected-error {{missing '@end'}}
@end
@end // expected-error {{'@end' must appear in an Objective-C context}}

@interface X4 // expected-note {{class started here}}
@implementation X4 // expected-error {{missing '@end'}}
@end
@end // expected-error {{'@end' must appear in an Objective-C context}}

@interface I
@end
@implementation I
@protocol P; // forward declarations of protocols in @implementations is allowed
@class C; // forward declarations of classes in @implementations is allowed
- (C<P>*) MyMeth {}
@end

@interface I2 {}
@protocol P2; // expected-error {{illegal interface qualifier}}
@class C2; // expected-error {{illegal interface qualifier}}
@end

@interface I3
@end
@implementation I3
- Meth {}
+ Cls {}
@protocol P3;
@end
