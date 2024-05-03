// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-pc -target-feature +sse2 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-pc %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spir-unknown-unknown %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple armv7a-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple riscv32 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple riscv64 %s

template <typename... Args>
__attribute__((format(printf, 1, 2)))
void format(const char *fmt, Args &&...args); // expected-warning{{GCC requires a function with the 'format' attribute to be variadic}}

template<typename... Args>
__attribute__((format(scanf, 1, 2)))
int scan(const char *fmt, Args &&...args); // expected-warning{{GCC requires a function with the 'format' attribute to be variadic}}

void do_format() {
  format("%f", (_Float16)123.f); // expected-warning{{format specifies type 'double' but the argument has type '_Float16'}}

  _Float16 Float16;
  scan("%f", &Float16); // expected-warning{{format specifies type 'float *' but the argument has type '_Float16 *'}}
  scan("%lf", &Float16); // expected-warning{{format specifies type 'double *' but the argument has type '_Float16 *'}}
  scan("%Lf", &Float16); // expected-warning{{format specifies type 'long double *' but the argument has type '_Float16 *'}}
}
