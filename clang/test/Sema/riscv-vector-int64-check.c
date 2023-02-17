// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +zve32x -target-feature +zfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target
#include <riscv_vector.h>

vint64m1_t foo() { /* expected-error {{RISC-V type 'vint64m1_t' (aka '__rvv_int64m1_t') requires the 'zve64x' extension}} */
} /* expected-warning {{non-void function does not return a value}}*/
