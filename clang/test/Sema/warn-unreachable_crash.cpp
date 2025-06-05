// RUN: %clang_cc1 -verify -Wunreachable-code %s

static void test(__fp16& x) {
  if (x != 0 || x != 1.0) { // expected-note{{}}
      x = 0.9;
    } else
      x = 0.8; // expected-warning{{code will never be executed}}
}
