// RUN: %clang_cc1 -verify -Wno-objc-root-class  -fsyntax-only  %s

__attribute__((objc_nonlazy_class))
@interface A
@end
@implementation A
@end

__attribute__((objc_nonlazy_class)) int X; // expected-error {{'objc_nonlazy_class' attribute only applies to Objective-C interfaces}}

__attribute__((objc_nonlazy_class()))
@interface B
@end
@implementation B
@end

__attribute__((objc_nonlazy_class("foo"))) // expected-error{{'objc_nonlazy_class' attribute takes no arguments}}
@interface C
@end
@implementation C
@end

__attribute__((objc_nonlazy_class)) // expected-error {{'objc_nonlazy_class' attribute only applies to Objective-C interfaces}}
@protocol B
@end

__attribute__((objc_nonlazy_class)) // expected-error {{'objc_nonlazy_class' attribute only applies to Objective-C interfaces}}
void foo(void);

@interface E
@end

__attribute__((objc_nonlazy_class))
@implementation E
@end

__attribute__((objc_nonlazy_class))
@implementation E (MyCat)
@end
