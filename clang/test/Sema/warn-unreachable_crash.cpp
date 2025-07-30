// RUN: %clang_cc1 -verify -Wunreachable-code %s

// Previously this test will crash
static void test(__fp16& x) {
  if (x != 0 || x != 1.0) { // expected-note{{}} no-crash
      x = 0.9;
    } else
      x = 0.8; // expected-warning{{code will never be executed}}
}

static void test2(__fp16& x) {
  if (x != 1 && x == 1.0) { // expected-note{{}} no-crash
      x = 0.9; // expected-warning{{code will never be executed}}
    } else
      x = 0.8;
}
