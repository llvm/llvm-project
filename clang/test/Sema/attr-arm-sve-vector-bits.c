// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=128 -fallow-half-arguments-and-returns %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=256 -fallow-half-arguments-and-returns %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=512 -fallow-half-arguments-and-returns %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=1024 -fallow-half-arguments-and-returns %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=2048 -fallow-half-arguments-and-returns %s

#define N __ARM_FEATURE_SVE_BITS_EXPERIMENTAL

typedef __SVInt8_t svint8_t;
typedef __SVInt16_t svint16_t;
typedef __SVInt32_t svint32_t;
typedef __SVInt64_t svint64_t;
typedef __SVUint8_t svuint8_t;
typedef __SVUint16_t svuint16_t;
typedef __SVUint32_t svuint32_t;
typedef __SVUint64_t svuint64_t;
typedef __SVFloat16_t svfloat16_t;
typedef __SVFloat32_t svfloat32_t;
typedef __SVFloat64_t svfloat64_t;

#if defined(__ARM_FEATURE_SVE_BF16)
typedef __SVBFloat16_t svbfloat16_t;
#endif

typedef __SVBool_t svbool_t;

// Define valid fixed-width SVE types
typedef svint8_t fixed_int8_t __attribute__((arm_sve_vector_bits(N)));
typedef svint16_t fixed_int16_t __attribute__((arm_sve_vector_bits(N)));
typedef svint32_t fixed_int32_t __attribute__((arm_sve_vector_bits(N)));
typedef svint64_t fixed_int64_t __attribute__((arm_sve_vector_bits(N)));

typedef svuint8_t fixed_uint8_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint16_t fixed_uint16_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint32_t fixed_uint32_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint64_t fixed_uint64_t __attribute__((arm_sve_vector_bits(N)));

typedef svfloat16_t fixed_float16_t __attribute__((arm_sve_vector_bits(N)));
typedef svfloat32_t fixed_float32_t __attribute__((arm_sve_vector_bits(N)));
typedef svfloat64_t fixed_float64_t __attribute__((arm_sve_vector_bits(N)));

typedef svbfloat16_t fixed_bfloat16_t __attribute__((arm_sve_vector_bits(N)));

typedef svbool_t fixed_bool_t __attribute__((arm_sve_vector_bits(N)));

// Attribute must have a single argument
typedef svint8_t no_argument __attribute__((arm_sve_vector_bits));         // expected-error {{'arm_sve_vector_bits' attribute takes one argument}}
typedef svint8_t two_arguments __attribute__((arm_sve_vector_bits(2, 4))); // expected-error {{'arm_sve_vector_bits' attribute takes one argument}}

// The number of SVE vector bits must be an integer constant expression
typedef svint8_t non_int_size1 __attribute__((arm_sve_vector_bits(2.0)));   // expected-error {{'arm_sve_vector_bits' attribute requires an integer constant}}
typedef svint8_t non_int_size2 __attribute__((arm_sve_vector_bits("256"))); // expected-error {{'arm_sve_vector_bits' attribute requires an integer constant}}

typedef __clang_svint8x2_t svint8x2_t;
typedef __clang_svfloat32x3_t svfloat32x3_t;

// Attribute must be attached to a single SVE vector or predicate type.
typedef void *badtype1 __attribute__((arm_sve_vector_bits(N)));         // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'void *'}}
typedef int badtype2 __attribute__((arm_sve_vector_bits(N)));           // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'int'}}
typedef float badtype3 __attribute__((arm_sve_vector_bits(N)));         // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'float'}}
typedef svint8x2_t badtype4 __attribute__((arm_sve_vector_bits(N)));    // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'svint8x2_t' (aka '__clang_svint8x2_t')}}
typedef svfloat32x3_t badtype5 __attribute__((arm_sve_vector_bits(N))); // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'svfloat32x3_t' (aka '__clang_svfloat32x3_t')}}
