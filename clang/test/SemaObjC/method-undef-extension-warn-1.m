// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface MyClass
@end

@protocol P
- (void)Pmeth;
- (void)Pmeth1; // expected-note {{method 'Pmeth1' declared here}}
@end

// Class extension
@interface MyClass () <P>
- (void)meth2; // expected-note {{method 'meth2' declared here}}
@end

// Add a category to test that clang does not emit warning for this method.
@interface MyClass (Category) 
- (void)categoryMethod;
@end

@implementation MyClass // expected-warning {{method 'Pmeth1' in protocol 'P' not implemented}} \
                        // expected-warning {{method definition for 'meth2' not found}}
- (void)Pmeth {}
@end
