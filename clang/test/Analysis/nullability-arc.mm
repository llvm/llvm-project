// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,nullability\
// RUN:                       -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,nullability\
// RUN:                       -analyzer-output=text -verify %s -fobjc-arc



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

  [self foo:nil];
  // expected-warning@-1{{nil passed to a callee that requires a non-null 1st parameter}}
  // expected-note@-2   {{nil passed to a callee that requires a non-null 1st parameter}}
}
@end

