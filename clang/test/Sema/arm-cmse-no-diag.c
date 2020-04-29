// RUN: %clang_cc1 -triple thumbv8m.base-none-eabi -mcmse -verify -Wno-cmse-union-leak %s
// expected-no-diagnostics

union U { unsigned n; char b[4]; } u;

void (*fn2)(int, union U) __attribute__((cmse_nonsecure_call));

union U xyzzy() __attribute__((cmse_nonsecure_entry)) {
  fn2(0, u);
  return u;
}
