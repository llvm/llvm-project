// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=128 -flax-vector-conversions=none -fallow-half-arguments-and-returns -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=256 -flax-vector-conversions=none -fallow-half-arguments-and-returns -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=512 -flax-vector-conversions=none -fallow-half-arguments-and-returns -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=1024 -flax-vector-conversions=none -fallow-half-arguments-and-returns -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=2048 -flax-vector-conversions=none -fallow-half-arguments-and-returns -ffreestanding -fsyntax-only -verify %s

// expected-no-diagnostics

#include <arm_sve.h>

#define N __ARM_FEATURE_SVE_BITS
#define FIXED_ATTR __attribute__((arm_sve_vector_bits(N)))

typedef svfloat32_t fixed_float32_t FIXED_ATTR;
typedef svfloat64_t fixed_float64_t FIXED_ATTR;
typedef svint32_t fixed_int32_t FIXED_ATTR;
typedef svint64_t fixed_int64_t FIXED_ATTR;
typedef svbool_t fixed_bool_t FIXED_ATTR;

// SVE VLSTs can be cast to SVE VLATs, regardless of lane size.
// NOTE: the list below is NOT exhaustive for all SVE types.

#define CAST(from, to) \
    void from##_to_##to(from a, to b) { \
        b = (to) a; \
    }

#define TESTCASE(ty1, ty2) \
    CAST(ty1, ty2) \
    CAST(ty2, ty1)

TESTCASE(fixed_float32_t, svfloat32_t)
TESTCASE(fixed_float32_t, svfloat64_t)
TESTCASE(fixed_float32_t, svint32_t)
TESTCASE(fixed_float32_t, svint64_t)

TESTCASE(fixed_float64_t, svfloat32_t)
TESTCASE(fixed_float64_t, svfloat64_t)
TESTCASE(fixed_float64_t, svint32_t)
TESTCASE(fixed_float64_t, svint64_t)

TESTCASE(fixed_int32_t, svfloat32_t)
TESTCASE(fixed_int32_t, svfloat64_t)
TESTCASE(fixed_int32_t, svint32_t)
TESTCASE(fixed_int32_t, svint64_t)

TESTCASE(fixed_int64_t, svfloat32_t)
TESTCASE(fixed_int64_t, svfloat64_t)
TESTCASE(fixed_int64_t, svint32_t)
TESTCASE(fixed_int64_t, svint64_t)

TESTCASE(fixed_bool_t, svbool_t)
