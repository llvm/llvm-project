// RUN: %clang_cc1 %s -verify -fsyntax-only -triple x86_64-pc-linux-gnu
typedef int __attribute__((wraps)) wrapping_int;
typedef unsigned __attribute__((wraps)) wrapping_u32;

int implicit_truncation(void) {
  const wrapping_int A = 1;
  return 2147483647 + A; // no warning
}

struct R {
  wrapping_int a: 2; // test bitfield sign change -- no warning
  wrapping_u32 b: 1; // test bitfield overflow/truncation -- no warning
  int baseline: 2; // baseline, should warn
};

void bitfields_truncation(void) {
  struct R r;
  r.a = 2; // this value changes from 2 to -2
  ++r.a;

  r.b = 2; // changes from 2 to 0
  ++r.b;

  // expected-warning@+1 {{to bit-field changes value from}}
  r.baseline = 2;
}

extern void implicitly_discards_wraps_attribute(int discards);

int discard_test(void) {
  wrapping_int A = 1;
  int __attribute__((no_wraps)) B = 1;
  // expected-warning@+1 {{'wraps' attribute may be implicitly discarded}}
  implicitly_discards_wraps_attribute(A);

  // expected-warning@+1 {{'no_wraps' attribute may be implicitly discarded}}
  implicitly_discards_wraps_attribute(B);

  int C = A; // assignments don't warn right now -- probably too noisy
  return A; // neither do non-wrapping return types
}

void useless_wraps_attribute(void) {
  // expected-error@+1 {{cannot use attribute 'wraps' with non-integer type}}
  float __attribute__((wraps)) A = 3.14;
  // expected-error@+1 {{cannot use attribute 'no_wraps' with non-integer type}}
  float __attribute__((no_wraps)) B = 3.14;
}

void wraps_used_with_no_wraps(void) {
  // expected-error@+1 {{attribute 'wraps' cannot be used alongside 'no_wraps'}}
  unsigned char __attribute__((wraps)) __attribute__((no_wraps)) A;
  // expected-error@+1 {{attribute 'wraps' cannot be used alongside 'no_wraps'}}
  long long __attribute__((no_wraps)) __attribute__((wraps)) B;
}

void pointers_and_pointees(void) {
  // expected-error@+1 {{cannot use attribute 'wraps' with non-integer type}}
  char *A __attribute__((wraps));
  // expected-error@+1 {{cannot use attribute 'no_wraps' with non-integer type}}
  char *B __attribute__((no_wraps));

  // allowed, see other tests in CodeGen
  char C __attribute__((wraps)) = 127;
  char *D = &C;
}
