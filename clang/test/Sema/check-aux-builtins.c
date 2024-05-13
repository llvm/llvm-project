// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple aarch64 -aux-triple x86_64-linux-pc -fsyntax-only -verify %s

void func(void) {
  (void)__builtin_cpu_is("atom");
  __builtin_cpu_is("INVALID"); // expected-error{{invalid cpu name for builtin}}
}
