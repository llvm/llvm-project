// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -Wno-error=incompatible-pointer-types -emit-llvm -o - -verify %s

int test_cpu_supports(void) {
  return __builtin_cpu_supports(L"wide"); // expected-warning {{incompatible pointer types}}
}

int test_cpu_is(void) {
  return __builtin_cpu_is(L"wide"); // expected-warning {{incompatible pointer types}}
}
