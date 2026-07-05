// RUN: %clang_cc1 -fsyntax-only -verify %s
// Test that a property can be synthesize in a category
// implementation with no error.

@protocol MyProtocol
@property float  myFloat;
@property float  anotherFloat; // expected-note 2 {{property declared}}
@end

@interface MyObject { float anotherFloat; }
@end

@interface MyObject (CAT) <MyProtocol>
@end

@implementation MyObject (CAT)	// expected-warning  {{property 'anotherFloat' requires method}} \
                                // expected-warning {{property 'anotherFloat' requires method 'setAnotherFloat:'}}
@dynamic myFloat;	// OK
@synthesize anotherFloat; // expected-error {{@synthesize not allowed in a category's implementation}}
@end
