// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fsyntax-only -fcuda-is-device -target-sdk-version=9.0 -verify %s

// The alias attribute is not allowed in CUDA device code before 10.0.
extern "C" void bar();
__attribute__((alias("bar"))) void foo();
// expected-error@-1 {{CUDA older than 10.0 does not support .alias}}
