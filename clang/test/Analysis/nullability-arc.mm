// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,nullability\
// RUN:                       -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,nullability\
// RUN:                       -analyzer-output=text -verify %s -fobjc-arc

#if !__has_feature(objc_arc)
// expected-no-diagnostics
#endif


#define nil ((id)0)

@interface Param
@end

@interface Base
- (void)foo:(Param *_Nonnull)param;
@end

@interface Derived : Base
@end

@implementation Derived
- (void)foo:(Param *)param {
  // FIXME: Why do we not emit the warning under ARC?
  [super foo:param];
#if __has_feature(objc_arc)
  // expected-warning@-2{{nil passed to a callee that requires a non-null 1st parameter}}
  // expected-note@-3   {{nil passed to a callee that requires a non-null 1st parameter}}
#endif

  [self foo:nil];
#if __has_feature(objc_arc)
  // expected-note@-2{{Calling 'foo:'}}
  // expected-note@-3{{Passing nil object reference via 1st parameter 'param'}}
#endif
}
@end

