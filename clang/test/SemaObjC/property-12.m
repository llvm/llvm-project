// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify %s

@protocol P0
@property(readonly,assign) id X;
@end

@protocol P1
@property(readonly,retain) id X;
@end

@protocol P2
@property(readonly,copy) id X;
@end

@protocol P3
@property(readonly,readwrite) id X; // expected-error {{property attributes 'readonly' and 'readwrite' are mutually exclusive}}
@end

@protocol P4
@property(assign,copy) id X; // expected-error {{property attributes 'assign' and 'copy' are mutually exclusive}}
@end

@protocol P5
@property(assign,retain) id X; // expected-error {{property attributes 'assign' and 'retain' are mutually exclusive}}
@end

@protocol P6
@property(copy,retain) id X; // expected-error {{property attributes 'copy' and 'retain' are mutually exclusive}}
@end

@interface I0 <P0> @end
@implementation I0 
@synthesize X;
@end

@interface I1 <P1> @end
@implementation I1 
@synthesize X;
@end

@interface I2 <P2> @end
@implementation I2 
@synthesize X;
@end

@interface I3 <P3> @end
@implementation I3 
@synthesize X;
@end

@interface I4 <P4> @end
@implementation I4 
@synthesize X;
@end

@interface I5 <P5> @end
@implementation I5 
@synthesize X;
@end

@interface I6 <P6> @end
@implementation I6 
@synthesize X;
@end

