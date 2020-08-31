// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -msve-vector-bits=128 \
// RUN:  | FileCheck %s --check-prefix=CHECK-128
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -msve-vector-bits=256 \
// RUN:  | FileCheck %s --check-prefix=CHECK-256
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -msve-vector-bits=512 \
// RUN:  | FileCheck %s --check-prefix=CHECK-512
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -msve-vector-bits=1024 \
// RUN:  | FileCheck %s --check-prefix=CHECK-1024
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -msve-vector-bits=2048 \
// RUN:  | FileCheck %s --check-prefix=CHECK-2048

#define N __ARM_FEATURE_SVE_BITS_EXPERIMENTAL

typedef __SVInt8_t fixed_int8_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVInt16_t fixed_int16_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVInt32_t fixed_int32_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVInt64_t fixed_int64_t __attribute__((arm_sve_vector_bits(N)));

typedef __SVUint8_t fixed_uint8_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVUint16_t fixed_uint16_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVUint32_t fixed_uint32_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVUint64_t fixed_uint64_t __attribute__((arm_sve_vector_bits(N)));

typedef __SVFloat16_t fixed_float16_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVFloat32_t fixed_float32_t __attribute__((arm_sve_vector_bits(N)));
typedef __SVFloat64_t fixed_float64_t __attribute__((arm_sve_vector_bits(N)));

typedef __SVBFloat16_t fixed_bfloat16_t __attribute__((arm_sve_vector_bits(N)));

typedef __SVBool_t fixed_bool_t __attribute__((arm_sve_vector_bits(N)));

template <typename T> struct S {};

// CHECK-128: _Z2f11SI9__SVE_VLSIu10__SVInt8_tLj128EEE
// CHECK-256: _Z2f11SI9__SVE_VLSIu10__SVInt8_tLj256EEE
// CHECK-512: _Z2f11SI9__SVE_VLSIu10__SVInt8_tLj512EEE
// CHECK-1024: _Z2f11SI9__SVE_VLSIu10__SVInt8_tLj1024EEE
// CHECK-2048: _Z2f11SI9__SVE_VLSIu10__SVInt8_tLj2048EEE
void f1(S<fixed_int8_t>) {}

// CHECK-128: _Z2f21SI9__SVE_VLSIu11__SVInt16_tLj128EEE
// CHECK-256: _Z2f21SI9__SVE_VLSIu11__SVInt16_tLj256EEE
// CHECK-512: _Z2f21SI9__SVE_VLSIu11__SVInt16_tLj512EEE
// CHECK-1024: _Z2f21SI9__SVE_VLSIu11__SVInt16_tLj1024EEE
// CHECK-2048: _Z2f21SI9__SVE_VLSIu11__SVInt16_tLj2048EEE
void f2(S<fixed_int16_t>) {}

// CHECK-128: _Z2f31SI9__SVE_VLSIu11__SVInt32_tLj128EEE
// CHECK-256: _Z2f31SI9__SVE_VLSIu11__SVInt32_tLj256EEE
// CHECK-512: _Z2f31SI9__SVE_VLSIu11__SVInt32_tLj512EEE
// CHECK-1024: _Z2f31SI9__SVE_VLSIu11__SVInt32_tLj1024EEE
// CHECK-2048: _Z2f31SI9__SVE_VLSIu11__SVInt32_tLj2048EEE
void f3(S<fixed_int32_t>) {}

// CHECK-128: _Z2f41SI9__SVE_VLSIu11__SVInt64_tLj128EEE
// CHECK-256: _Z2f41SI9__SVE_VLSIu11__SVInt64_tLj256EEE
// CHECK-512: _Z2f41SI9__SVE_VLSIu11__SVInt64_tLj512EEE
// CHECK-1024: _Z2f41SI9__SVE_VLSIu11__SVInt64_tLj1024EEE
// CHECK-2048: _Z2f41SI9__SVE_VLSIu11__SVInt64_tLj2048EEE
void f4(S<fixed_int64_t>) {}

// CHECK-128: _Z2f51SI9__SVE_VLSIu11__SVUint8_tLj128EEE
// CHECK-256: _Z2f51SI9__SVE_VLSIu11__SVUint8_tLj256EEE
// CHECK-512: _Z2f51SI9__SVE_VLSIu11__SVUint8_tLj512EEE
// CHECK-1024: _Z2f51SI9__SVE_VLSIu11__SVUint8_tLj1024EEE
// CHECK-2048: _Z2f51SI9__SVE_VLSIu11__SVUint8_tLj2048EEE
void f5(S<fixed_uint8_t>) {}

// CHECK-128: _Z2f61SI9__SVE_VLSIu12__SVUint16_tLj128EEE
// CHECK-256: _Z2f61SI9__SVE_VLSIu12__SVUint16_tLj256EEE
// CHECK-512: _Z2f61SI9__SVE_VLSIu12__SVUint16_tLj512EEE
// CHECK-1024: _Z2f61SI9__SVE_VLSIu12__SVUint16_tLj1024EEE
// CHECK-2048: _Z2f61SI9__SVE_VLSIu12__SVUint16_tLj2048EEE
void f6(S<fixed_uint16_t>) {}

// CHECK-128: _Z2f71SI9__SVE_VLSIu12__SVUint32_tLj128EEE
// CHECK-256: _Z2f71SI9__SVE_VLSIu12__SVUint32_tLj256EEE
// CHECK-512: _Z2f71SI9__SVE_VLSIu12__SVUint32_tLj512EEE
// CHECK-1024: _Z2f71SI9__SVE_VLSIu12__SVUint32_tLj1024EEE
// CHECK-2048: _Z2f71SI9__SVE_VLSIu12__SVUint32_tLj2048EEE
void f7(S<fixed_uint32_t>) {}

// CHECK-128: _Z2f81SI9__SVE_VLSIu12__SVUint64_tLj128EEE
// CHECK-256: _Z2f81SI9__SVE_VLSIu12__SVUint64_tLj256EEE
// CHECK-512: _Z2f81SI9__SVE_VLSIu12__SVUint64_tLj512EEE
// CHECK-1024: _Z2f81SI9__SVE_VLSIu12__SVUint64_tLj1024EEE
// CHECK-2048: _Z2f81SI9__SVE_VLSIu12__SVUint64_tLj2048EEE
void f8(S<fixed_uint64_t>) {}

// CHECK-128: _Z2f91SI9__SVE_VLSIu13__SVFloat16_tLj128EEE
// CHECK-256: _Z2f91SI9__SVE_VLSIu13__SVFloat16_tLj256EEE
// CHECK-512: _Z2f91SI9__SVE_VLSIu13__SVFloat16_tLj512EEE
// CHECK-1024: _Z2f91SI9__SVE_VLSIu13__SVFloat16_tLj1024EEE
// CHECK-2048: _Z2f91SI9__SVE_VLSIu13__SVFloat16_tLj2048EEE
void f9(S<fixed_float16_t>) {}

// CHECK-128: _Z3f101SI9__SVE_VLSIu13__SVFloat32_tLj128EEE
// CHECK-256: _Z3f101SI9__SVE_VLSIu13__SVFloat32_tLj256EEE
// CHECK-512: _Z3f101SI9__SVE_VLSIu13__SVFloat32_tLj512EEE
// CHECK-1024: _Z3f101SI9__SVE_VLSIu13__SVFloat32_tLj1024EEE
// CHECK-2048: _Z3f101SI9__SVE_VLSIu13__SVFloat32_tLj2048EEE
void f10(S<fixed_float32_t>) {}

// CHECK-128: _Z3f111SI9__SVE_VLSIu13__SVFloat64_tLj128EEE
// CHECK-256: _Z3f111SI9__SVE_VLSIu13__SVFloat64_tLj256EEE
// CHECK-512: _Z3f111SI9__SVE_VLSIu13__SVFloat64_tLj512EEE
// CHECK-1024: _Z3f111SI9__SVE_VLSIu13__SVFloat64_tLj1024EEE
// CHECK-2048: _Z3f111SI9__SVE_VLSIu13__SVFloat64_tLj2048EEE
void f11(S<fixed_float64_t>) {}

// CHECK-128: _Z3f121SI9__SVE_VLSIu14__SVBfloat16_tLj128EEE
// CHECK-256: _Z3f121SI9__SVE_VLSIu14__SVBfloat16_tLj256EEE
// CHECK-512: _Z3f121SI9__SVE_VLSIu14__SVBfloat16_tLj512EEE
// CHECK-1024: _Z3f121SI9__SVE_VLSIu14__SVBfloat16_tLj1024EEE
// CHECK-2048: _Z3f121SI9__SVE_VLSIu14__SVBfloat16_tLj2048EEE
void f12(S<fixed_bfloat16_t>) {}

// CHECK-128: _Z3f131SI9__SVE_VLSIu10__SVBool_tLj128EEE
// CHECK-256: _Z3f131SI9__SVE_VLSIu10__SVBool_tLj256EEE
// CHECK-512: _Z3f131SI9__SVE_VLSIu10__SVBool_tLj512EEE
// CHECK-1024: _Z3f131SI9__SVE_VLSIu10__SVBool_tLj1024EEE
// CHECK-2048: _Z3f131SI9__SVE_VLSIu10__SVBool_tLj2048EEE
void f13(S<fixed_bool_t>) {}
