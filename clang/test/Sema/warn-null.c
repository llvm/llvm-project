// RUN: %clang_cc1 %s -verify

#define SOME_ADDR (unsigned long long)0

// PR10837: Warn if a non-pointer-typed expression is folded to a null pointer
int *p = 0;
int *q = '\0'; // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
int *r = (1 - 1); // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
void f(void) {
  p = 0;
  q = '\0'; // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
  r = 1 - 1; // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
  p = SOME_ADDR; // expected-warning{{expression which evaluates to zero treated as a null pointer constant}}
}
