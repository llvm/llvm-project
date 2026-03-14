// RUN: %clang_cc1 -verify -fsyntax-only  %s

__attribute__((objc_runtime_visible))
@interface A
@end

@interface A(X)
@end

@implementation A(X) // expected-error{{cannot implement a category for class 'A' that is only visible via the Objective-C runtime}}
@end

@interface B : A
@end

@implementation B // expected-error{{cannot implement subclass 'B' of a superclass 'A' that is only visible via the Objective-C runtime}}
@end


