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

void test_labels() {
label:
  int i = 0;

  switch (i) {
  case 1:
    _Static_assert(true);
  default:
    _Static_assert(true);
  }
}
