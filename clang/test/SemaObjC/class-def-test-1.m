// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@protocol SUPER; // expected-note {{protocol 'SUPER' has no definition}}

@interface SUPER <SUPER> @end // expected-warning {{cannot find protocol definition for 'SUPER'}}

typedef int INTF; //  expected-note {{previous definition is here}}

@interface INTF @end // expected-error {{redefinition of 'INTF' as different kind of symbol}}

@interface OBJECT @end	// expected-note {{previous definition is here}}

@interface INTF1 : OBJECT @end // expected-note {{previous definition is here}}

@interface INTF1 : OBJECT @end // expected-error {{duplicate interface definition for class 'INTF1'}}

typedef int OBJECT; // expected-error {{redefinition of 'OBJECT' as different kind of symbol}}

typedef int OBJECT2; // expected-note 2 {{previous definition is here}}
@interface INTF2 : OBJECT2 @end // expected-error {{redefinition of 'OBJECT2' as different kind of symbol}}

@implementation INTF2 : OBJECT2 @end // expected-error {{redefinition of 'OBJECT2' as different kind of symbol}}

@protocol PROTO;

@interface INTF3 : PROTO @end // expected-error {{cannot find interface declaration for 'PROTO', superclass of 'INTF3'}}

// Make sure we allow the following (for GCC compatibility).
@interface NSObject @end
typedef NSObject TD_NSObject;
@interface XCElementUnit : TD_NSObject {}
@end

// Make sure we don't typo-correct to ourselves.
@interface SomeClassSub : SomeClassSup // expected-error{{cannot find interface declaration for 'SomeClassSup', superclass of 'SomeClassSub'}}
@end
