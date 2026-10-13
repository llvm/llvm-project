// Tests for riscv_rvv_vector_bits when no fixed VLEN is supplied (i.e. no
// -mrvv-vector-bits=). The bound for riscv_rvv_vector_bits(N) on
// a data type then comes from the ISA's Zvl*b extension, scaled by LMUL.
//
// Bound: N <= MinElts * EltSize * zvl / RVVBitsPerBlock = zvl * LMUL (bits)
//
// vintXm1_t  -> bound = zvl
// vintXm2_t  -> bound = zvl * 2
// vintXmf2_t -> bound = zvl / 2
//
// Mask types still need -mrvv-vector-bits=.

// zvl128b: ISA minimum VLEN = 128.
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl128b \
// RUN:   -ffreestanding -fsyntax-only -verify \
// RUN:   %s

// zvl256b: ISA minimum VLEN = 256.
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl256b \
// RUN:   -ffreestanding -fsyntax-only -verify \
// RUN:   -DZVL256 %s

typedef __rvv_int32m1_t  vint32m1_t;
typedef __rvv_int32m2_t  vint32m2_t;
typedef __rvv_int32mf2_t vint32mf2_t;
typedef __rvv_bool32_t   vbool32_t;

#if defined(ZVL256)

// zvl256b: m1 bound is 256.
typedef vint32m1_t  m1_eq  __attribute__((riscv_rvv_vector_bits(256)));
typedef vint32m1_t  m1_lt  __attribute__((riscv_rvv_vector_bits(128)));
typedef vint32m1_t  m1_too __attribute__((riscv_rvv_vector_bits(512)));
// expected-error@-1 {{invalid RVV vector size '512', must not exceed '256' based on LMUL of type and the minimum VLEN}}

// zvl256b: m2 bound is 512.
typedef vint32m2_t  m2_eq  __attribute__((riscv_rvv_vector_bits(512)));
typedef vint32m2_t  m2_lt  __attribute__((riscv_rvv_vector_bits(256)));
typedef vint32m2_t  m2_too __attribute__((riscv_rvv_vector_bits(1024)));
// expected-error@-1 {{invalid RVV vector size '1024', must not exceed '512' based on LMUL of type and the minimum VLEN}}

// zvl256b: mf2 bound is 128.
typedef vint32mf2_t mf2_eq  __attribute__((riscv_rvv_vector_bits(128)));
typedef vint32mf2_t mf2_lt  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32mf2_t mf2_too __attribute__((riscv_rvv_vector_bits(256)));
// expected-error@-1 {{invalid RVV vector size '256', must not exceed '128' based on LMUL of type and the minimum VLEN}}

#else // zvl128b

// zvl128b: m1 bound is 128.
typedef vint32m1_t  m1_eq  __attribute__((riscv_rvv_vector_bits(128)));
typedef vint32m1_t  m1_lt  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m1_t  m1_min __attribute__((riscv_rvv_vector_bits(32)));
typedef vint32m1_t  m1_too __attribute__((riscv_rvv_vector_bits(256)));
// expected-error@-1 {{invalid RVV vector size '256', must not exceed '128' based on LMUL of type and the minimum VLEN}}

// zvl128b: m2 bound is 256.
typedef vint32m2_t  m2_eq  __attribute__((riscv_rvv_vector_bits(256)));
typedef vint32m2_t  m2_lt  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m2_t  m2_too __attribute__((riscv_rvv_vector_bits(512)));
// expected-error@-1 {{invalid RVV vector size '512', must not exceed '256' based on LMUL of type and the minimum VLEN}}

// zvl128b: mf2 bound is 64.
typedef vint32mf2_t mf2_eq  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32mf2_t mf2_lt  __attribute__((riscv_rvv_vector_bits(32)));
typedef vint32mf2_t mf2_too __attribute__((riscv_rvv_vector_bits(128)));
// expected-error@-1 {{invalid RVV vector size '128', must not exceed '64' based on LMUL of type and the minimum VLEN}}

// Lane count must be a power of two.
typedef vint32m1_t m1_npow2 __attribute__((riscv_rvv_vector_bits(96)));
// expected-error@-1 {{invalid RVV vector size '96', the resulting element count '3' must be a power of 2}}

typedef vbool32_t mask __attribute__((riscv_rvv_vector_bits(4)));
// expected-error@-1 {{'riscv_rvv_vector_bits' is only supported when '-mrvv-vector-bits=<bits>' is specified with a value of "zvl" or a power 2 in the range [64,65536]}}

#endif
