// VLEN=128: vbool32_t has 4 lanes at full size.
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl128b \
// RUN:   -ffreestanding -fsyntax-only -verify %s

typedef __rvv_bool32_t vbool32_t;

// Masks follow the same rule as data types, counted in lanes.
typedef vbool32_t mask_full __attribute__((riscv_rvv_vector_bits(4)));
typedef vbool32_t mask_shorter __attribute__((riscv_rvv_vector_bits(2)));
typedef vbool32_t mask_one __attribute__((riscv_rvv_vector_bits(1)));
typedef vbool32_t mask_too_big __attribute__((riscv_rvv_vector_bits(8)));
// expected-error@-1 {{invalid RVV vector size '8', must not exceed '4' based on LMUL of type and the minimum VLEN}}
typedef vbool32_t mask_three __attribute__((riscv_rvv_vector_bits(3)));
// expected-error@-1 {{invalid RVV vector size '3', the resulting element count '3' must be a power of 2}}
typedef vbool32_t mask_zero __attribute__((riscv_rvv_vector_bits(0)));
// expected-error@-1 {{invalid RVV vector size '0', the resulting element count '0' must be a power of 2}}
