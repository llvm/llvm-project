// VLEN=128: vint32m1_t has 4 elems at full size, vint8m1_t has 16 elems.
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl128b \
// RUN:   -ffreestanding -fsyntax-only -verify \
// RUN:   %s

// VLEN=512: vint32m1_t has 16 elems at full size.
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -target-feature +zvl512b \
// RUN:   -ffreestanding -fsyntax-only -verify \
// RUN:   -DVLEN512 %s

typedef __rvv_int32m1_t vint32m1_t;
typedef __rvv_int8m1_t vint8m1_t;
typedef __rvv_bool32_t vbool32_t;

#if !defined(VLEN512) // VLEN=128

// Equal-size case still works (compat).
typedef vint32m1_t v_eq_i32 __attribute__((riscv_rvv_vector_bits(128)));

// Shorter sizes (power of 2) are accepted.
typedef vint32m1_t v_i32x2 __attribute__((riscv_rvv_vector_bits(64)));   // 2 elems
typedef vint32m1_t v_i32x1 __attribute__((riscv_rvv_vector_bits(32)));   // 1 elem
typedef vint8m1_t  v_i8x8  __attribute__((riscv_rvv_vector_bits(64)));   // 8 elems
typedef vint8m1_t  v_i8x4  __attribute__((riscv_rvv_vector_bits(32)));   // 4 elems

// Reject N > vlen*lmul.
typedef vint32m1_t too_big __attribute__((riscv_rvv_vector_bits(256)));
// expected-error@-1 {{invalid RVV vector size '256', must not exceed '128' based on LMUL of type and the minimum VLEN}}

// Reject element count that is not a power of 2.
typedef vint32m1_t three_elems __attribute__((riscv_rvv_vector_bits(96)));
// expected-error@-1 {{invalid RVV vector size '96', the resulting element count '3' must be a power of 2}}

// Reject sizes that aren't a multiple of the element size.
typedef vint32m1_t odd_bits __attribute__((riscv_rvv_vector_bits(40)));
// expected-error@-1 {{invalid RVV vector size '40', must be a positive multiple of the element size '32'}}

// Reject zero.
typedef vint32m1_t zero_bits __attribute__((riscv_rvv_vector_bits(0)));
// expected-error@-1 {{invalid RVV vector size '0', must be a positive multiple of the element size '32'}}

// Masks must still match exactly, counted in lanes: vbool32_t has 4 lanes.
typedef vbool32_t mask_full __attribute__((riscv_rvv_vector_bits(4)));
typedef vbool32_t mask_shorter __attribute__((riscv_rvv_vector_bits(2)));
// expected-error@-1 {{invalid RVV vector size '2', expected size is '4' based on LMUL of type and '-mrvv-vector-bits'}}

#else // VLEN=512

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

#endif
