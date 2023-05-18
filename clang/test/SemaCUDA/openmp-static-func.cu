// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:            -verify -fopenmp %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:            -verify -fopenmp -x hip %s
// expected-no-diagnostics

// Tests there is no assertion in Sema::markKnownEmitted when fopenmp is used
// with CUDA/HIP host compilation.

static void f() {}

static void g() { f(); }

static void h() { g(); }
