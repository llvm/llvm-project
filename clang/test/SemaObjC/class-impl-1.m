// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef int INTF3; // expected-note {{previous definition is here}}

@interface SUPER @end // expected-note {{previous definition is here}}

@interface OBJECT @end

@interface INTF  : OBJECT
@end

@implementation INTF @end // expected-note {{previous definition is here}}

@implementation INTF //  expected-error {{reimplementation of class 'INTF'}}
@end


@interface INTF1 : OBJECT
@end

@implementation INTF1 : SUPER // expected-error {{conflicting super class name 'SUPER'}}
@end

@interface INTF2 
@end

@implementation INTF2 : SUPR //  expected-error {{cannot find interface declaration for 'SUPR', superclass of 'INTF2'}}
@end

@implementation INTF3 @end // expected-error {{redefinition of 'INTF3' as different kind of symbol}}

@implementation INTF4 @end // expected-warning {{cannot find interface declaration for 'INTF4'}}

@class INTF5; // expected-note{{forward declaration of class here}}

@implementation INTF5 {  // expected-warning {{cannot find interface declaration for 'INTF5'}}
  int x;
}
@end

