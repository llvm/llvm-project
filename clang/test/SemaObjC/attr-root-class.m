// RUN: %clang_cc1 -fsyntax-only -verify -Wattributes -Wobjc-root-class %s
@interface RootClass {} // expected-warning {{class 'RootClass' defined without specifying a base class}} \
                        // expected-note {{add a super class to fix this problem}}
@end
@implementation RootClass
@end

__attribute__((objc_root_class))
@interface NonRootClass : RootClass   // expected-error {{objc_root_class attribute may only be specified on a root class declaration}}
@end
@implementation NonRootClass
@end

__attribute__((objc_root_class)) static void nonClassDeclaration(void)  // expected-error {{'objc_root_class' attribute only applies to Objective-C interfaces}}
{
}

__attribute__((objc_root_class(1))) // expected-error {{'objc_root_class' attribute takes no arguments}}
@interface I1
@end
