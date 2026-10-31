// RUN: %clang_cc1 -std=c++20 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-static-sized-array \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -verify %s

void unsafe_pointer_arithmetic(int idx) {
  int buffer[10]; // expected-warning {{'buffer' is an unsafe buffer that does not perform bounds checks}}

  int *u1 = buffer + 10;  // expected-note {{used in pointer arithmetic here}}
  int *u2 = buffer + 15;  // expected-note {{used in pointer arithmetic here}}

  int *u3 = buffer - 1;   // expected-note {{used in pointer arithmetic here}}

  int *u4 = buffer + idx; // expected-note {{used in pointer arithmetic here}}
}
