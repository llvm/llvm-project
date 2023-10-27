// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=1 -mvscale-max=1 -flax-vector-conversions=none -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=2 -mvscale-max=2 -flax-vector-conversions=none -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=4 -mvscale-max=4 -flax-vector-conversions=none -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=8 -mvscale-max=8 -flax-vector-conversions=none -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=16 -mvscale-max=16 -flax-vector-conversions=none -ffreestanding -fsyntax-only -verify %s

// expected-no-diagnostics

// REQUIRES: riscv-registered-target

#define FIXED_ATTR __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)))

typedef __rvv_int8m1_t vint8m1_t;
typedef __rvv_uint8m1_t vuint8m1_t;
typedef __rvv_int16m1_t vint16m1_t;
typedef __rvv_uint16m1_t vuint16m1_t;
typedef __rvv_int32m1_t vint32m1_t;
typedef __rvv_uint32m1_t vuint32m1_t;
typedef __rvv_int64m1_t vint64m1_t;
typedef __rvv_uint64m1_t vuint64m1_t;
typedef __rvv_float32m1_t vfloat32m1_t;
typedef __rvv_float64m1_t vfloat64m1_t;

typedef vfloat32m1_t fixed_float32m1_t FIXED_ATTR;
typedef vfloat64m1_t fixed_float64m1_t FIXED_ATTR;
typedef vint32m1_t fixed_int32m1_t FIXED_ATTR;
typedef vint64m1_t fixed_int64m1_t FIXED_ATTR;

// RVV VLS types can be cast to RVV VLA types, regardless of lane size.
// NOTE: the list below is NOT exhaustive for all RVV types.

#define CAST(from, to) \
    void from##_to_##to(from a, to b) { \
        b = (to) a; \
    }

#define TESTCASE(ty1, ty2) \
    CAST(ty1, ty2) \
    CAST(ty2, ty1)

TESTCASE(fixed_float32m1_t, vfloat32m1_t)
TESTCASE(fixed_float32m1_t, vfloat64m1_t)
TESTCASE(fixed_float32m1_t, vint32m1_t)
TESTCASE(fixed_float32m1_t, vint64m1_t)

TESTCASE(fixed_float64m1_t, vfloat32m1_t)
TESTCASE(fixed_float64m1_t, vfloat64m1_t)
TESTCASE(fixed_float64m1_t, vint32m1_t)
TESTCASE(fixed_float64m1_t, vint64m1_t)

TESTCASE(fixed_int32m1_t, vfloat32m1_t)
TESTCASE(fixed_int32m1_t, vfloat64m1_t)
TESTCASE(fixed_int32m1_t, vint32m1_t)
TESTCASE(fixed_int32m1_t, vint64m1_t)

TESTCASE(fixed_int64m1_t, vfloat32m1_t)
TESTCASE(fixed_int64m1_t, vfloat64m1_t)
TESTCASE(fixed_int64m1_t, vint32m1_t)
TESTCASE(fixed_int64m1_t, vint64m1_t)
