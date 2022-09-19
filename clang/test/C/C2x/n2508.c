// RUN: %clang_cc1 -verify -std=c2x %s

// expected-no-diagnostics

/* WG14 N2508: yes
 * Free positioning of labels inside compound statements
 */
void test() {
  {
  inner:
  }

  switch (1) {
  case 1:
  }

  {
  multiple: labels: on: a: line:
  }

final:
}

