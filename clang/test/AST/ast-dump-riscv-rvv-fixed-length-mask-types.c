// Test printing of RVV fixed-length mask types (VectorKind::RVVFixedLengthMask*).

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve32x \
// RUN:   -ffreestanding -mvscale-min=1 -mvscale-max=1 -ast-dump %s \
// RUN:   | FileCheck %s

typedef __rvv_bool64_t vbool64_t;
typedef __rvv_bool32_t vbool32_t;
typedef __rvv_bool16_t vbool16_t;
typedef __rvv_bool8_t  vbool8_t;
typedef __rvv_bool4_t  vbool4_t;
typedef __rvv_bool2_t  vbool2_t;
typedef __rvv_bool1_t  vbool1_t;

// With vscale==1, __riscv_v_fixed_vlen==64.

// RVVFixedLengthMask_1 (1-bit mask, getNumElements()==1)
typedef vbool64_t fixed_bool64_t __attribute__((riscv_rvv_vector_bits(1)));
// RVVFixedLengthMask_2 (2-bit mask, getNumElements()==1)
typedef vbool32_t fixed_bool32_t __attribute__((riscv_rvv_vector_bits(2)));
// RVVFixedLengthMask_4 (4-bit mask, getNumElements()==1)
typedef vbool16_t fixed_bool16_t __attribute__((riscv_rvv_vector_bits(4)));
// RVVFixedLengthMask (8-bit mask, getNumElements()==1)
typedef vbool8_t fixed_bool8_t __attribute__((riscv_rvv_vector_bits(8)));
// RVVFixedLengthMask (16-bit mask, getNumElements()==2)
typedef vbool4_t fixed_bool4_t __attribute__((riscv_rvv_vector_bits(16)));
// RVVFixedLengthMask (32-bit mask, getNumElements()==4)
typedef vbool2_t fixed_bool2_t __attribute__((riscv_rvv_vector_bits(32)));
// RVVFixedLengthMask (64-bit mask, getNumElements()==8)
typedef vbool1_t fixed_bool1_t __attribute__((riscv_rvv_vector_bits(64)));

fixed_bool64_t b64;
// CHECK: b64 'fixed_bool64_t':'__attribute__((__riscv_rvv_vector_bits__(1))) unsigned char'

fixed_bool32_t b32;
// CHECK: b32 'fixed_bool32_t':'__attribute__((__riscv_rvv_vector_bits__(2))) unsigned char'

fixed_bool16_t b16;
// CHECK: b16 'fixed_bool16_t':'__attribute__((__riscv_rvv_vector_bits__(4))) unsigned char'

fixed_bool8_t b8;
// CHECK: b8 'fixed_bool8_t':'__attribute__((__riscv_rvv_vector_bits__(1 * sizeof(unsigned char) * 8))) unsigned char'

fixed_bool4_t b4;
// CHECK: b4 'fixed_bool4_t':'__attribute__((__riscv_rvv_vector_bits__(2 * sizeof(unsigned char) * 8))) unsigned char'

fixed_bool2_t b2;
// CHECK: b2 'fixed_bool2_t':'__attribute__((__riscv_rvv_vector_bits__(4 * sizeof(unsigned char) * 8))) unsigned char'

fixed_bool1_t b1;
// CHECK: b1 'fixed_bool1_t':'__attribute__((__riscv_rvv_vector_bits__(8 * sizeof(unsigned char) * 8))) unsigned char'
