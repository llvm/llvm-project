// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fsyntax-only -fcuda-is-device -verify -DEXPECT_ERR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

// The alias attribute is not allowed in CUDA device code.
void bar();
__attribute__((alias("bar"))) void foo();
#ifdef EXPECT_ERR
// expected-error@-2 {{CUDA does not support aliases}}
#else
// expected-no-diagnostics
#endif
