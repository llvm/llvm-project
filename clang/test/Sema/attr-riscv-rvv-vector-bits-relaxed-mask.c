// Mask types still need a fixed VLEN.
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -mvscale-min=2 -mvscale-max=2 \
// RUN:   -ffreestanding -fsyntax-only -verify %s

typedef __rvv_bool32_t vbool32_t;

// Masks must still match exactly, counted in lanes: vbool32_t has 4 lanes.
typedef vbool32_t mask_full __attribute__((riscv_rvv_vector_bits(4)));
typedef vbool32_t mask_shorter __attribute__((riscv_rvv_vector_bits(2)));
// expected-error@-1 {{invalid RVV vector size '2', expected size is '4' based on LMUL of type and '-mrvv-vector-bits'}}
