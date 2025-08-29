// RUN: %clang_cc1  -fsyntax-only -verify %s

template<class T>
struct [[nodiscard]] expected {};

using E = expected<int>;

using NI [[nodiscard]] = int; // expected-warning {{'[[nodiscard]]' attribute ignored when applied to a typedef}}
using WURI [[clang::warn_unused_result]] = int;

using EIgnored [[clang::candiscard]] = E;
using NIIgnored [[clang::candiscard]] = NI;
using WURIgnored [[clang::candiscard]] = WURI;

@interface INTF
- (int) a [[nodiscard]];
+ (int) b [[nodiscard]];
- (expected<int>) c;
+ (expected<int>) d;
- (E) e;
- (EIgnored) e_ignored;
- (E) e_ignored2 [[clang::candiscard]];
+ (E) f;
- (void) g [[nodiscard]]; // expected-warning {{attribute 'nodiscard' cannot be applied to Objective-C method without return value}}
- (NI) h;
- (NIIgnored) h_ignored;
- (NI) h_ignored2 [[clang::candiscard]];
- (WURI) i;
- (WURIgnored) i_ignored;
- (WURI) i_ignored2 [[clang::candiscard]];
@end

void foo(INTF *a) {
  [a a]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  [INTF b]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  [a c]; // expected-warning {{ignoring return value of type 'expected<int>' declared with 'nodiscard' attribute}}
  [INTF d]; // expected-warning {{ignoring return value of type 'expected<int>' declared with 'nodiscard' attribute}}
  [a e]; // expected-warning {{ignoring return value of type 'expected<int>' declared with 'nodiscard' attribute}}
  [a e_ignored];  // no warning
  [a e_ignored2]; // no warning
  [INTF f]; // expected-warning {{ignoring return value of type 'expected<int>' declared with 'nodiscard' attribute}}
  [a g]; // no warning because g returns void
  [a h]; // no warning because attribute is ignored
  [a h_ignored];  // no warning
  [a h_ignored2]; // no warning
  [a i]; // expected-warning {{ignoring return value of type 'WURI' declared with 'clang::warn_unused_result' attribute}}
  [a i_ignored];  // no warning
  [a i_ignored2]; // no warning
}
