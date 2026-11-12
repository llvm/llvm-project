// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-pc -target-feature +sse2 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-pc %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spir-unknown-unknown %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple armv7a-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple riscv32 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple riscv64 %s

void a(const char *a, ...) __attribute__((format(printf, 1, 2)));    // no-error

void b(char *a, _Float16 b) __attribute__((format(printf, 1, 2))); // expected-warning {{GCC requires a function with the 'format' attribute to be variadic}}

void call_no_default_promotion(void) {
  a("%f", (_Float16)1.0); // expected-warning{{format specifies type 'double' but the argument has type '_Float16'}}
  b("%f", (_Float16)1.0); // expected-warning{{format specifies type 'double' but the argument has type '_Float16'}}
}
