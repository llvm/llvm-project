// RUN: %clang_cc1 -triple riscv64 -target-feature +v -emit-llvm -o - -verify %s

#pragma clang riscv intrinsic vector
// expected-no-diagnostics
