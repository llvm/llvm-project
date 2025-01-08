// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.10.0 -verify %s
// expected-no-diagnostics

@interface NSObject
@end

[[clang::objc_exception]]
@interface Foo {
  [[clang::iboutlet]] NSObject *h;
}
@property (readonly) [[clang::objc_returns_inner_pointer]] void *i, *j;
@property (readonly) [[clang::iboutlet]] NSObject *k;
@end

[[clang::objc_runtime_name("name")]] @protocol Bar;

[[clang::objc_protocol_requires_explicit_implementation]]
@protocol Baz
@end

@interface Quux
-(void)g1 [[clang::ns_consumes_self]];
-(void)g2 __attribute__((ns_consumes_self));
-(void)h1: (int)x [[clang::ns_consumes_self]];
-(void)h2: (int)x __attribute__((ns_consumes_self));
-(void) [[clang::ns_consumes_self]] i1;
-(void) __attribute__((ns_consumes_self)) i2;
@end
