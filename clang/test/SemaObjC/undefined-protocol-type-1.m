// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol p1, p4;
@protocol p2 @end

@interface T
- (T<p2, p3, p1, p4>*) meth;  // expected-error {{cannot find protocol declaration for 'p3'}}
- (T<p2, p3, p1, p4>*) meth1;  // expected-error {{cannot find protocol declaration for 'p3'}}
@end
