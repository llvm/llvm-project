// RUN: %clang_cc1 -verify -Wunsequenced -Wno-unused-value %s

/* WG14 N1282: Yes
 * Clarification of Expressions
 */

int g;

int f(int i) {
  g = i;
  return 0;
}

int main(void) {
  int x;
  x = (10, g = 1, 20) + (30, g = 2, 40); /* Line A */ // expected-warning {{multiple unsequenced modifications to 'g'}}
  x = (10, f(1), 20) + (30, f(2), 40); /* Line B */
  x = (g = 1) + (g = 2); /* Line C */                 // expected-warning {{multiple unsequenced modifications to 'g'}}
  return 0;
}
