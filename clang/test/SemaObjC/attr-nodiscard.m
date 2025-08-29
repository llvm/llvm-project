// RUN: %clang_cc1  -fsyntax-only -verify %s

struct [[nodiscard]] expected {};

typedef struct expected E;

[[nodiscard]] typedef int NI; // expected-warning {{'[[nodiscard]]' attribute ignored when applied to a typedef}}
typedef __attribute__((warn_unused_result)) int WUR;
typedef __attribute__((candiscard)) struct expected EIgnored;
typedef __attribute__((candiscard)) WUR WURIgnored;

@interface INTF
- (int) a [[nodiscard]];
+ (int) b [[nodiscard]];
- (struct expected) c;
+ (struct expected) d;
- (E) e;
- (EIgnored) e_ignored;
- (E) e_ignored2 __attribute__((candiscard));
+ (E) f;
- (void) g [[nodiscard]]; // expected-warning {{attribute 'nodiscard' cannot be applied to Objective-C method without return value}}
- (NI) h;
- (NI) h_ignored __attribute__((candiscard));
- (WUR) i;
- (WURIgnored) i_ignored;
- (WUR) i_ignored2 __attribute__((candiscard));
@end

void foo(INTF *a) {
  [a a]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  [INTF b]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  [a c]; // expected-warning {{ignoring return value of type 'expected' declared with 'nodiscard' attribute}}
  [INTF d]; // expected-warning {{ignoring return value of type 'expected' declared with 'nodiscard' attribute}}
  [a e]; // expected-warning {{ignoring return value of type 'expected' declared with 'nodiscard' attribute}}
  [INTF f]; // expected-warning {{ignoring return value of type 'expected' declared with 'nodiscard' attribute}}
  [a g]; // no warning because g returns void
  [a h]; // no warning because attribute is ignored when applied to a typedef
  [a h_ignored];  // no warning
  [a i]; // expected-warning {{ignoring return value of type 'WUR' declared with 'warn_unused_result' attribute}}
  [a i_ignored];  // no warning
  [a i_ignored2]; // no warning
}
