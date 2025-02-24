// RUN: %clang_cc1 -std=c++17 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -verify %s

// Previously we had a bug where we would diagnose *no* unsafe buffer warnings
// if the warning was disabled by pragma and left disabled until end-of-TU.
// This is a reasonable way to disable unsafe buffer warnings on an entire
// .c/cpp file, and it shouldn't disable the warnings in headers or previous
// source locations, so we test that this works.

// FIXME: This RUNX line should pass, but it does not, because we check if the
// warning is enabled *at startup*. If we ever implement a way to query if the
// warning is enabled anywhere in the TU, we can enable this RUN line.
// RUNX: %clang_cc1 -std=c++17 -Wno-all -verify %s

#pragma clang diagnostic warning "-Wunsafe-buffer-usage"
int *w1(int *p) {
  return p + 1; // expected-warning{{unsafe pointer arithmetic}}
}
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
int *w2(int *p) {
  return p + 1;
}
