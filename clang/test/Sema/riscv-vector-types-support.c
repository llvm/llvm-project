// RUN: %clang_cc1 %s -triple riscv64 -fsyntax-only -verify

typedef __attribute__((riscv_rvv_vector_bits(256))) void norvvflag; // expected-error{{'riscv_rvv_vector_bits' attribute is not supported on targets missing 'zve32x'; specify an appropriate -march= or -mcpu=}}
