// VLEN=512: vint32m1_t has 16 elems at full size.
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl512b \
// RUN:   -ffreestanding -fsyntax-only -verify %s

typedef __rvv_int32m1_t vint32m1_t;

typedef vint32m1_t v_full   __attribute__((riscv_rvv_vector_bits(512)));
typedef vint32m1_t v_i32x8  __attribute__((riscv_rvv_vector_bits(256)));
typedef vint32m1_t v_i32x4  __attribute__((riscv_rvv_vector_bits(128)));
typedef vint32m1_t v_i32x2  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m1_t v_i32x1  __attribute__((riscv_rvv_vector_bits(32)));

// Still reject N > vlen*lmul.
typedef vint32m1_t too_big __attribute__((riscv_rvv_vector_bits(1024)));
// expected-error@-1 {{invalid RVV vector size '1024', must not exceed '512' based on LMUL of type and the minimum VLEN}}

// Reject non-pow2 element count (5 * 32 = 160).
typedef vint32m1_t five_elems __attribute__((riscv_rvv_vector_bits(160)));
// expected-error@-1 {{invalid RVV vector size '160', the resulting element count '5' must be a power of 2}}
