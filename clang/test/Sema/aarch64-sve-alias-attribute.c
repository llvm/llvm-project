// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -aux-triple aarch64 -target-feature +sve -fopenmp-is-target-device -fopenmp -verify -fsyntax-only %s

static __inline__ __attribute__((__clang_arm_builtin_alias(__builtin_sve_svabd_n_f64_m))) // expected-no-diagnostics
void
nop(void);
