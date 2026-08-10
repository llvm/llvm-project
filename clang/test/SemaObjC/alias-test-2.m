// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

// Note: GCC doesn't produce any of the following errors.
@interface Super @end // expected-note {{previous definition is here}}

@interface MyWpModule @end  // expected-note {{previous definition is here}}

@compatibility_alias  MyAlias MyWpModule;

@compatibility_alias  AliasForSuper Super;

@implementation MyAlias : AliasForSuper // expected-error {{conflicting super class name 'Super'}}
@end

@interface MyAlias : AliasForSuper // expected-error {{duplicate interface definition for class 'MyWpModule'}}
@end

