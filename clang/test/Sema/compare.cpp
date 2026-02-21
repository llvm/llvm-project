// RUN: %clang_cc1 -fsyntax-only
// Expect no assertion failures in this file (#173614).
typedef unsigned long __attribute__((__vector_size__(8))) W;

int i;
W g;

void negation(void) {
  W w = i == (-g);
}

void bitwiseNot(void) {
  W w = i == (~g);
}
