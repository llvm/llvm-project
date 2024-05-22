// RUN: %clang_cc1 -triple riscv64 -target-feature +v %s
// REQUIRES: riscv-registered-target

// expected-no-diagnostics

#include <sifive_vector.h>
