// RUN: %clang -target i386-apple-darwin -fsyntax-only -Xclang -verify %s
// RUN: %clang -target i386-apple-darwin -x objective-c++ -fsyntax-only -Xclang -verify %s

@interface NSObject
@end

__attribute__((objc_class_stub)) // expected-warning {{'objc_class_stub' attribute ignored}}
__attribute__((objc_subclassing_restricted))
@interface StubClass : NSObject
@end
