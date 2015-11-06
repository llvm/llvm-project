// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://16560476

__attribute__((objc_subclassing_restricted))
@interface Leaf // okay
@end

__attribute__((objc_subclassing_restricted))
@interface SubClassOfLeaf : Leaf // expected-note {{class is declared here}}
@end


@interface SubClass : SubClassOfLeaf // expected-error {{cannot subclass a class with objc_subclassing_restricted attribute}}
@end

__attribute__((objc_root_class))
@interface PlainRoot
@end

__attribute__((objc_subclassing_restricted))
@interface Sub2Class : PlainRoot // okay
@end
