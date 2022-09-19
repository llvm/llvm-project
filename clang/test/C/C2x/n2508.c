// RUN: %clang_cc1 -verify -std=c2x %s

/* WG14 N2508: partial
 * Free positioning of labels inside compound statements
 */
void test() {
  {
  inner:
  }

  switch (1) {
  // FIXME: this should be accepted per C2x 6.8.2p2.
  case 1: // expected-error {{label at end of switch compound statement: expected statement}}
  }

  {
  multiple: labels: on: a: line:
  }

final:
}

