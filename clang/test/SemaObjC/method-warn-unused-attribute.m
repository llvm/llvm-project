// RUN: %clang_cc1  -fsyntax-only -Wunused-value -verify %s

@interface INTF
- (id) foo __attribute__((warn_unused_result)); 
- (void) garf __attribute__((warn_unused_result)); // expected-warning {{attribute 'warn_unused_result' cannot be applied to Objective-C method without return value}}
- (int) fee __attribute__((warn_unused_result));
+ (int) c __attribute__((warn_unused_result));
@end

void foo(INTF *a) {
  [a garf];
  [a fee]; // expected-warning {{ignoring return value of function declared with warn_unused_result attribute}}
  [INTF c]; // expected-warning {{ignoring return value of function declared with warn_unused_result attribute}}
}


