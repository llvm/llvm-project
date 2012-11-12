// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

/* This test is for categories which don't implement the accessors but some accessors are
   implemented in their base class implementation. In this case,no warning must be issued.
*/

@interface MyClass 
{
    int        _foo;
}
@property(readonly)    int        foo;
@end

@implementation MyClass
- (int) foo        { return _foo; }
@end

@interface MyClass (private)
@property(readwrite)    int        foo;
@end

@implementation MyClass (private)
- (void) setFoo:(int)foo    { _foo = foo; }
@end

@interface MyClass (public)
@property(readwrite)    int        foo;	// expected-note {{property declared here}}
@end

@implementation MyClass (public)// expected-warning {{property 'foo' requires method 'setFoo:' to be defined }}
@end 
