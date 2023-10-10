// RUN: %clang_cc1 -verify -fsyntax-only -std=c2x -triple=x86_64-unknown-linux %s

// Regression test for bug where we used to hit an assertion due to shift amount
// being larger than 64 bits. We want to see a warning about too large shift
// amount.
void test1(int *a) {
  (void)(*a >> 123456789012345678901uwb <= 0); // expected-warning {{shift count >= width of type}}
}

// Similar to test1 above, but using __uint128_t instead of __BitInt.
// We want to see a warning about too large shift amount.
void test2(__uint128_t *a) {
  (void)(*a >> ((__uint128_t)__UINT64_MAX__ + 1) <= 0); // expected-warning {{shift count >= width of type}}
}
