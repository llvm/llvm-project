// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s


@interface Foo
-(void)paramNamedID:(int)id usesIDType:(id)notShadowed;
-(void)paramNamedID:(int)id, id notShadowed; // expected-warning{{use of C-style parameters in Objective-C method declarations is deprecated}}
@end
