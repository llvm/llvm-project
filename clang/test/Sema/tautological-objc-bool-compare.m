// RUN: %clang_cc1 %s -verify

typedef signed char BOOL;
#define YES __objc_yes
#define NO __objc_no

BOOL B;

void test() {
  int r;
  r = B > 0;
  r = B > 1; // expected-warning {{result of comparison of constant 1 with expression of type 'BOOL' is always false, as the only well defined values for 'BOOL' are YES and NO}}
  r = B < 1;
  r = B < 0; // expected-warning {{result of comparison of constant 0 with expression of type 'BOOL' is always false, as the only well defined values for 'BOOL' are YES and NO}}
  r = B >= 0; // expected-warning {{result of comparison of constant 0 with expression of type 'BOOL' is always true, as the only well defined values for 'BOOL' are YES and NO}}
  r = B <= 0;

  r = B > YES; // expected-warning {{result of comparison of constant YES with expression of type 'BOOL' is always false, as the only well defined values for 'BOOL' are YES and NO}}
  r = B > NO;
  r = B < NO; // expected-warning {{result of comparison of constant NO with expression of type 'BOOL' is always false, as the only well defined values for 'BOOL' are YES and NO}}
  r = B < YES;
  r = B >= NO; // expected-warning {{result of comparison of constant NO with expression of type 'BOOL' is always true, as the only well defined values for 'BOOL' are YES and NO}}
  r = B <= NO;
}
