// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface Foo // expected-note {{previous definition is here}}
@end

float Foo;	// expected-error {{redefinition of 'Foo' as different kind of symbol}}

@class Bar;  // expected-note {{previous definition is here}}

typedef int Bar;  // expected-error {{redefinition of 'Bar' as different kind of symbol}}

@implementation FooBar // expected-warning {{cannot find interface declaration for 'FooBar'}} 
@end


typedef int OBJECT; // expected-note {{previous definition is here}}

@class OBJECT ;	// expected-error {{redefinition of 'OBJECT' as different kind of symbol}}


typedef int Gorf;  // expected-note {{previous definition is here}}

@interface Gorf @end // expected-error {{redefinition of 'Gorf' as different kind of symbol}} expected-note {{previous definition is here}}

void Gorf(void) // expected-error {{redefinition of 'Gorf' as different kind of symbol}}
{
  int Bar, Foo, FooBar;
}

@protocol P -im1; @end
@protocol Q -im2; @end
@interface A<P> @end  // expected-note {{previous definition is here}}
@interface A<Q> @end  // expected-error {{duplicate interface definition for class 'A'}}

@protocol PP<P> @end  // expected-note {{previous definition is here}}
@protocol PP<Q> @end  // expected-warning {{duplicate protocol definition of 'PP'}}

@protocol DP<P> @end
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wduplicate-protocol"
@protocol DP<Q> @end
#pragma clang diagnostic pop

@interface A(Cat)<P> @end // expected-note {{previous definition is here}}
@interface A(Cat)<Q> @end // expected-warning {{duplicate definition of category 'Cat' on interface 'A'}}

// rdar 7626768
@class NSString;
NSString * TestBaz;  // expected-note {{previous definition is here}}
NSString * const TestBaz;  // expected-error {{redefinition of 'TestBaz' with a different type}}
