// RUN: %clang -target x86_64-apple-darwin -fsyntax-only -Xclang -verify %s
// RUN: %clang -target x86_64-apple-darwin -x objective-c++ -fsyntax-only -Xclang -verify %s

@interface NSObject
@end

__attribute__((objc_class_stub))
@interface MissingSubclassingRestrictedAttribute : NSObject // expected-error {{'objc_class_stub' attribute cannot be specified on a class that does not have the 'objc_subclassing_restricted' attribute}}
@end

__attribute__((objc_class_stub))
__attribute__((objc_subclassing_restricted))
@interface ValidClassStubAttribute : NSObject
@end

@implementation ValidClassStubAttribute // expected-error {{cannot declare implementation of a class declared with the 'objc_class_stub' attribute}}
@end

@implementation ValidClassStubAttribute (MyCategory)
@end

__attribute__((objc_class_stub(123))) // expected-error {{'objc_class_stub' attribute takes no arguments}}
@interface InvalidClassStubAttribute : NSObject
@end

__attribute__((objc_class_stub)) // expected-error {{'objc_class_stub' attribute only applies to Objective-C interfaces}}
int cannotHaveObjCClassStubAttribute() {}
