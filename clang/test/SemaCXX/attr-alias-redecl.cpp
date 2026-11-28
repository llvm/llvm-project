// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// expected-no-diagnostics

// Regression test for https://github.com/llvm/llvm-project/issues/173464

typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));

extern const __m128 kAlias;
const __m128 kAlias __attribute__((alias("kOne")));
