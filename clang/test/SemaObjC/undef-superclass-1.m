// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@class SUPER, Y; // expected-note 2 {{forward declaration of class here}}

@interface INTF :SUPER  // expected-error {{attempting to use the forward class 'SUPER' as superclass of 'INTF'}}
@end

@interface SUPER @end

@interface INTF1 : SUPER  // expected-note {{previous definition is here}}
@end

@interface INTF2 : INTF1
@end

@interface INTF3 : Y // expected-error {{attempting to use the forward class 'Y' as superclass of 'INTF3'}} \
                     // expected-note{{'INTF3' declared here}}
@end

@interface INTF1  // expected-error {{duplicate interface definition for class 'INTF1'}}
@end

@implementation SUPER
- (void)dealloc {
    [super dealloc]; // expected-error {{'SUPER' cannot use 'super' because it is a root class}}
}
@end

@interface RecursiveClass : RecursiveClass // expected-error {{trying to recursively use 'RecursiveClass' as superclass of 'RecursiveClass'}}
@end

@implementation RecursiveClass
@end

@implementation iNTF3 // expected-warning{{cannot find interface declaration for 'iNTF3'; did you mean 'INTF3'?}}
@end
