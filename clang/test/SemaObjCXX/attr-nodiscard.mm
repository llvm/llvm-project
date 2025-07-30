// RUN: %clang_cc1  -fsyntax-only -verify %s

template<class T>
struct [[nodiscard]] expected {};

using E = expected<int>;

@interface INTF
- (int) a [[nodiscard]];
+ (int) b [[nodiscard]];
- (expected<int>) c;
+ (expected<int>) d;
- (E) e;
+ (E) f;
- (void) g [[nodiscard]]; // expected-warning {{attribute 'nodiscard' cannot be applied to Objective-C method without return value}}
@end

void foo(INTF *a) {
  [a a]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  [INTF b]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  [a c]; // expected-warning {{ignoring return value of type 'expected<int>' declared with 'nodiscard' attribute}}
  [INTF d]; // expected-warning {{ignoring return value of type 'expected<int>' declared with 'nodiscard' attribute}}
  [a e]; // expected-warning {{ignoring return value of type 'expected<int>' declared with 'nodiscard' attribute}}
  [INTF f]; // expected-warning {{ignoring return value of type 'expected<int>' declared with 'nodiscard' attribute}}
  [a g];
}
