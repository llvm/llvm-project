// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -verify -Wunreachable-code %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +fullfp16 -verify -Wunreachable-code %s
// REQUIRES: aarch64-registered-target

// =======  __fp16 version  =======
static void test_fp16(__fp16 &x) {
  if (x != 0 || x != 1.0) {           // expected-note {{}} no-crash
    x = 0.9;
  } else
    x = 0.8;                          // expected-warning{{code will never be executed}}
}

static void test_fp16_b(__fp16 &x) {
  if (x != 1 && x == 1.0) {           // expected-note {{}} no-crash
    x = 0.9;                          // expected-warning{{code will never be executed}}
  } else
    x = 0.8;
}

// =======  _Float16 version  =======
static void test_f16(_Float16 &x) {
  if (x != 0 || x != 1.0) {           // expected-note {{}} no-crash
    x = 0.9;
  } else
    x = 0.8;                          // expected-warning{{code will never be executed}}
}

static void test_f16_b(_Float16 &x) {
  if (x != 1 && x == 1.0) {           // expected-note {{}} no-crash
    x = 0.9;                          // expected-warning{{code will never be executed}}
  } else
    x = 0.8;
}
