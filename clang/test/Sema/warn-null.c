// RUN: %clang_cc1 %s -verify

// PR10837: Warn if a non-pointer-typed expression is folded to a null pointer
int *p = 0;
int *q = '\0'; // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
int *r = (1 - 1); // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
void f() {
  p = 0;
  q = '\0'; // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
  r = 1 - 1; // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
}
