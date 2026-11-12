// RUN: %clang_cc1 -fsyntax-only -verify -Wselector-type-mismatch %s

__attribute__((objc_root_class))
@interface Foo
@property() int dynamic_property;
@property(direct) int direct_property; // expected-note {{previous declaration is here}}
@end

@implementation Foo
@dynamic dynamic_property;
@dynamic direct_property; // expected-error {{direct property cannot be @dynamic}}
@end

@interface Foo (Bar)
@property() int dynamic_category_property;
@property(direct) int direct_category_property; // expected-note {{previous declaration is here}}
@end

@implementation Foo (Bar)
@dynamic dynamic_category_property;
@dynamic direct_category_property; // expected-error {{direct property cannot be @dynamic}}
@end
