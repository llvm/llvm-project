// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -mvscale-min=1 -mvscale-max=1 \
// RUN:  | FileCheck %s --check-prefix=CHECK-128
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -mvscale-min=2 -mvscale-max=2 \
// RUN:  | FileCheck %s --check-prefix=CHECK-256
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -mvscale-min=4 -mvscale-max=4 \
// RUN:  | FileCheck %s --check-prefix=CHECK-512
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -mvscale-min=8 -mvscale-max=8 \
// RUN:  | FileCheck %s --check-prefix=CHECK-1024
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +sve -target-feature +bf16 -mvscale-min=16 -mvscale-max=16 \
// RUN:  | FileCheck %s --check-prefix=CHECK-2048

#define N __ARM_FEATURE_SVE_BITS

namespace std {
class type_info;
};

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

typedef __SVBfloat16_t fixed_bfloat16_t __attribute__((arm_sve_vector_bits(N)));

typedef __SVBool_t fixed_bool_t __attribute__((arm_sve_vector_bits(N)));

auto &fs8 = typeid(fixed_int8_t);
auto &fs16 = typeid(fixed_int16_t);
auto &fs32 = typeid(fixed_int32_t);
auto &fs64 = typeid(fixed_int64_t);

auto &fu8 = typeid(fixed_uint8_t);
auto &fu16 = typeid(fixed_uint16_t);
auto &fu32 = typeid(fixed_uint32_t);
auto &fu64 = typeid(fixed_uint64_t);

auto &ff16 = typeid(fixed_float16_t);
auto &ff32 = typeid(fixed_float32_t);
auto &ff64 = typeid(fixed_float64_t);

auto &fbf16 = typeid(fixed_bfloat16_t);

auto &fb8 = typeid(fixed_bool_t);

// CHECK-128: @_ZTI9__SVE_VLSIu10__SVInt8_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVInt8_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu10__SVInt8_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVInt8_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu10__SVInt8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVInt8_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu10__SVInt8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVInt8_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu10__SVInt8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVInt8_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu11__SVInt16_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt16_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu11__SVInt16_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt16_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu11__SVInt16_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt16_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu11__SVInt16_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt16_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu11__SVInt16_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt16_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu11__SVInt32_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt32_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu11__SVInt32_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt32_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu11__SVInt32_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt32_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu11__SVInt32_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt32_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu11__SVInt32_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt32_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu11__SVInt64_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt64_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu11__SVInt64_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt64_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu11__SVInt64_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt64_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu11__SVInt64_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt64_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu11__SVInt64_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVInt64_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu11__SVUint8_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVUint8_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu11__SVUint8_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVUint8_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu11__SVUint8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVUint8_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu11__SVUint8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVUint8_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu11__SVUint8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu11__SVUint8_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu12__SVUint16_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint16_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu12__SVUint16_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint16_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu12__SVUint16_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint16_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu12__SVUint16_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint16_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu12__SVUint16_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint16_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu12__SVUint32_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint32_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu12__SVUint32_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint32_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu12__SVUint32_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint32_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu12__SVUint32_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint32_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu12__SVUint32_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint32_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu12__SVUint64_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint64_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu12__SVUint64_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint64_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu12__SVUint64_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint64_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu12__SVUint64_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint64_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu12__SVUint64_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu12__SVUint64_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu13__SVFloat16_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat16_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu13__SVFloat16_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat16_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu13__SVFloat16_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat16_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu13__SVFloat16_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat16_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu13__SVFloat16_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat16_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu13__SVFloat32_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat32_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu13__SVFloat32_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat32_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu13__SVFloat32_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat32_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu13__SVFloat32_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat32_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu13__SVFloat32_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat32_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu13__SVFloat64_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat64_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu13__SVFloat64_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat64_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu13__SVFloat64_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat64_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu13__SVFloat64_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat64_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu13__SVFloat64_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu13__SVFloat64_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu14__SVBfloat16_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu14__SVBfloat16_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu14__SVBfloat16_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu14__SVBfloat16_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu14__SVBfloat16_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu14__SVBfloat16_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu14__SVBfloat16_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu14__SVBfloat16_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu14__SVBfloat16_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu14__SVBfloat16_tLj2048EE

// CHECK-128: @_ZTI9__SVE_VLSIu10__SVBool_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVBool_tLj128EE
// CHECK-256: @_ZTI9__SVE_VLSIu10__SVBool_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVBool_tLj256EE
// CHECK-512: @_ZTI9__SVE_VLSIu10__SVBool_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVBool_tLj512EE
// CHECK-1024: @_ZTI9__SVE_VLSIu10__SVBool_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVBool_tLj1024EE
// CHECK-2048: @_ZTI9__SVE_VLSIu10__SVBool_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__SVE_VLSIu10__SVBool_tLj2048EE
