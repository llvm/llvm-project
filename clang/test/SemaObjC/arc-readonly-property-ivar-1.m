// RUN: %clang_cc1  -fobjc-default-synthesize-properties  -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fobjc-arc -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1  -x objective-c++ -fobjc-default-synthesize-properties  -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fobjc-arc -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar:// 10558871

@interface PP
@property (readonly) id ReadOnlyPropertyNoBackingIvar;
@property (readonly) id ReadOnlyProperty;
@property (readonly) id ReadOnlyPropertyX;
@end

@implementation PP {
__weak id _ReadOnlyProperty;
}
@synthesize ReadOnlyPropertyNoBackingIvar;
@synthesize ReadOnlyProperty = _ReadOnlyProperty;
@synthesize ReadOnlyPropertyX = _ReadOnlyPropertyX;
@end

@interface DD
@property (readonly) id ReadOnlyProperty;
@property (readonly) id ReadOnlyPropertyStrong;
@property (readonly) id ReadOnlyPropertyNoBackingIvar;
@end

@implementation DD {
__weak id _ReadOnlyProperty;
__strong id _ReadOnlyPropertyStrong;
}
@end
