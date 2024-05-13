// RUN: %clang_cc1 %s -verify -fsyntax-only -Wno-strict-prototypes

void f1(void) __attribute__((ownership_takes("foo"))); // expected-error {{'ownership_takes' attribute requires parameter 1 to be an identifier}}
void *f2(void) __attribute__((ownership_returns(foo, 1, 2)));  // expected-error {{'ownership_returns' attribute takes no more than 1 argument}}
void f3(void) __attribute__((ownership_holds(foo, 1))); // expected-error {{'ownership_holds' attribute parameter 1 is out of bounds}}
void *f4(void) __attribute__((ownership_returns(foo)));
void f5(void) __attribute__((ownership_holds(foo)));  // expected-error {{'ownership_holds' attribute takes at least 2 arguments}}
void f6(void) __attribute__((ownership_holds(foo, 1, 2, 3)));  // expected-error {{'ownership_holds' attribute parameter 1 is out of bounds}}
void f7(void) __attribute__((ownership_takes(foo)));  // expected-error {{'ownership_takes' attribute takes at least 2 arguments}}
void f8(int *i, int *j, int k) __attribute__((ownership_holds(foo, 1, 2, 4)));  // expected-error {{'ownership_holds' attribute parameter 3 is out of bounds}}

int f9 __attribute__((ownership_takes(foo, 1)));  // expected-warning {{'ownership_takes' attribute only applies to non-K&R-style functions}}

void f10(int i) __attribute__((ownership_holds(foo, 1)));  // expected-error {{'ownership_holds' attribute only applies to pointer arguments}}
void *f11(float i) __attribute__((ownership_returns(foo, 1)));  // expected-error {{'ownership_returns' attribute only applies to integer arguments}}
void *f12(float i, int k, int f, int *j) __attribute__((ownership_returns(foo, 4)));  // expected-error {{'ownership_returns' attribute only applies to integer arguments}}

void f13(int *i, int *j) __attribute__((ownership_holds(foo, 1))) __attribute__((ownership_takes(foo, 2)));
void f14(int i, int j, int *k) __attribute__((ownership_holds(foo, 3))) __attribute__((ownership_takes(foo, 3)));  // expected-error {{'ownership_takes' and 'ownership_holds' attributes are not compatible}}

void f15(int, int)
  __attribute__((ownership_returns(foo, 1)))  // expected-error {{'ownership_returns' attribute index does not match; here it is 1}}
  __attribute__((ownership_returns(foo, 2))); // expected-note {{declared with index 2 here}}
void f16(int *i, int *j) __attribute__((ownership_holds(foo, 1))) __attribute__((ownership_holds(foo, 1))); // OK, same index
void f17(void*) __attribute__((ownership_takes(__, 1)));
void f18() __attribute__((ownership_takes(foo, 1)));  // expected-warning {{'ownership_takes' attribute only applies to non-K&R-style functions}}
