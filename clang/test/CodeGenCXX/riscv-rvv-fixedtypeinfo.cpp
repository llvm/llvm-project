// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +f -target-feature +d \
// RUN:  -target-feature +zve64d -mvscale-min=1 -mvscale-max=1 \
// RUN:  | FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +f -target-feature +d \
// RUN:  -target-feature +zve64d -mvscale-min=2 -mvscale-max=2 \
// RUN:  | FileCheck %s --check-prefix=CHECK-128
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +f -target-feature +d \
// RUN:  -target-feature +zve64d -mvscale-min=4 -mvscale-max=4 \
// RUN:  | FileCheck %s --check-prefix=CHECK-256
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +f -target-feature +d \
// RUN:  -target-feature +zve64d -mvscale-min=8 -mvscale-max=8 \
// RUN:  | FileCheck %s --check-prefix=CHECK-512
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +f -target-feature +d \
// RUN:  -target-feature +zve64d -mvscale-min=16 -mvscale-max=16 \
// RUN:  | FileCheck %s --check-prefix=CHECK-1024

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

typedef vint8m1_t fixed_int8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint16m1_t fixed_int16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint32m1_t fixed_int32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint64m1_t fixed_int64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

typedef vuint8m1_t fixed_uint8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint16m1_t fixed_uint16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint32m1_t fixed_uint32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint64m1_t fixed_uint64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

typedef vfloat32m1_t fixed_float32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vfloat64m1_t fixed_float64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

namespace std {
class type_info;
};

auto &fs8 = typeid(fixed_int8m1_t);
auto &fs16 = typeid(fixed_int16m1_t);
auto &fs32 = typeid(fixed_int32m1_t);
auto &fs64 = typeid(fixed_int64m1_t);

auto &fu8 = typeid(fixed_uint8m1_t);
auto &fu16 = typeid(fixed_uint16m1_t);
auto &fu32 = typeid(fixed_uint32m1_t);
auto &fu64 = typeid(fixed_uint64m1_t);

auto &ff32 = typeid(fixed_float32m1_t);
auto &ff64 = typeid(fixed_float64m1_t);

// CHECK-64: @_ZTI9__RVV_VLSIu14__rvv_int8m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu14__rvv_int8m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu14__rvv_int8m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu14__rvv_int8m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu14__rvv_int8m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int16m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int16m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int16m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int16m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int16m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int32m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int32m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int32m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int32m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int32m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int64m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int64m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int64m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int64m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int64m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_uint8m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_uint8m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_uint8m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_uint8m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_uint8m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint16m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint16m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint16m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint16m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint16m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint32m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint32m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint32m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint32m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint32m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint64m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint64m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint64m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint64m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint64m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_float32m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_float32m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_float32m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_float32m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_float32m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m1_tLj1024EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_float64m1_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m1_tLj64EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_float64m1_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m1_tLj128EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_float64m1_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m1_tLj256EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_float64m1_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m1_tLj512EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_float64m1_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m1_tLj1024EE
