// RUN: %clang_cc1 -std=c++20 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -verify %s

// CHECK-NOT: [-Wunsafe-buffer-usage]


void foo(unsigned idx) {
  char c = '0';
  c = "abc"[0];
  c = "abc"[1];
  c = "abc"[2];
  c = "abc"[3];
  c = "abc"[4]; // expected-warning{{unsafe buffer access}}
  c = "abc"[idx]; // expected-warning{{unsafe buffer access}}
  c = ""[0];
  c = ""[1]; // expected-warning{{unsafe buffer access}}
}
