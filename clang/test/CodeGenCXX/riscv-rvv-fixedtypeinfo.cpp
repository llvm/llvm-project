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

typedef __rvv_int8mf8_t vint8mf8_t;
typedef __rvv_uint8mf8_t vuint8mf8_t;

typedef __rvv_int8mf4_t vint8mf4_t;
typedef __rvv_uint8mf4_t vuint8mf4_t;
typedef __rvv_int16mf4_t vint16mf4_t;
typedef __rvv_uint16mf4_t vuint16mf4_t;

typedef __rvv_int8mf2_t vint8mf2_t;
typedef __rvv_uint8mf2_t vuint8mf2_t;
typedef __rvv_int16mf2_t vint16mf2_t;
typedef __rvv_uint16mf2_t vuint16mf2_t;
typedef __rvv_int32mf2_t vint32mf2_t;
typedef __rvv_uint32mf2_t vuint32mf2_t;
typedef __rvv_float32mf2_t vfloat32mf2_t;

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

typedef __rvv_int8m2_t vint8m2_t;
typedef __rvv_uint8m2_t vuint8m2_t;
typedef __rvv_int16m2_t vint16m2_t;
typedef __rvv_uint16m2_t vuint16m2_t;
typedef __rvv_int32m2_t vint32m2_t;
typedef __rvv_uint32m2_t vuint32m2_t;
typedef __rvv_int64m2_t vint64m2_t;
typedef __rvv_uint64m2_t vuint64m2_t;
typedef __rvv_float32m2_t vfloat32m2_t;
typedef __rvv_float64m2_t vfloat64m2_t;

typedef __rvv_int8m4_t vint8m4_t;
typedef __rvv_uint8m4_t vuint8m4_t;
typedef __rvv_int16m4_t vint16m4_t;
typedef __rvv_uint16m4_t vuint16m4_t;
typedef __rvv_int32m4_t vint32m4_t;
typedef __rvv_uint32m4_t vuint32m4_t;
typedef __rvv_int64m4_t vint64m4_t;
typedef __rvv_uint64m4_t vuint64m4_t;
typedef __rvv_float32m4_t vfloat32m4_t;
typedef __rvv_float64m4_t vfloat64m4_t;

typedef __rvv_int8m8_t vint8m8_t;
typedef __rvv_uint8m8_t vuint8m8_t;
typedef __rvv_int16m8_t vint16m8_t;
typedef __rvv_uint16m8_t vuint16m8_t;
typedef __rvv_int32m8_t vint32m8_t;
typedef __rvv_uint32m8_t vuint32m8_t;
typedef __rvv_int64m8_t vint64m8_t;
typedef __rvv_uint64m8_t vuint64m8_t;
typedef __rvv_float32m8_t vfloat32m8_t;
typedef __rvv_float64m8_t vfloat64m8_t;

typedef vint8mf8_t fixed_int8mf8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/8)));

typedef vuint8mf8_t fixed_uint8mf8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/8)));

typedef vint8mf4_t fixed_int8mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/4)));
typedef vint16mf4_t fixed_int16mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/4)));

typedef vuint8mf4_t fixed_uint8mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/4)));
typedef vuint16mf4_t fixed_uint16mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/4)));

typedef vint8mf2_t fixed_int8mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/2)));
typedef vint16mf2_t fixed_int16mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/2)));
typedef vint32mf2_t fixed_int32mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/2)));

typedef vuint8mf2_t fixed_uint8mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/2)));
typedef vuint16mf2_t fixed_uint16mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/2)));
typedef vuint32mf2_t fixed_uint32mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/2)));

typedef vfloat32mf2_t fixed_float32mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen/2)));

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

typedef vint8m2_t fixed_int8m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));
typedef vint16m2_t fixed_int16m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));
typedef vint32m2_t fixed_int32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));
typedef vint64m2_t fixed_int64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));

typedef vuint8m2_t fixed_uint8m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));
typedef vuint16m2_t fixed_uint16m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));
typedef vuint32m2_t fixed_uint32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));
typedef vuint64m2_t fixed_uint64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));

typedef vfloat32m2_t fixed_float32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));
typedef vfloat64m2_t fixed_float64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*2)));

typedef vint8m4_t fixed_int8m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));
typedef vint16m4_t fixed_int16m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));
typedef vint32m4_t fixed_int32m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));
typedef vint64m4_t fixed_int64m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));

typedef vuint8m4_t fixed_uint8m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));
typedef vuint16m4_t fixed_uint16m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));
typedef vuint32m4_t fixed_uint32m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));
typedef vuint64m4_t fixed_uint64m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));

typedef vfloat32m4_t fixed_float32m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));
typedef vfloat64m4_t fixed_float64m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*4)));

typedef vint8m8_t fixed_int8m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));
typedef vint16m8_t fixed_int16m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));
typedef vint32m8_t fixed_int32m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));
typedef vint64m8_t fixed_int64m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));

typedef vuint8m8_t fixed_uint8m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));
typedef vuint16m8_t fixed_uint16m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));
typedef vuint32m8_t fixed_uint32m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));
typedef vuint64m8_t fixed_uint64m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));

typedef vfloat32m8_t fixed_float32m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));
typedef vfloat64m8_t fixed_float64m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen*8)));

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

auto &fs8m2 = typeid(fixed_int8m2_t);
auto &fs16m2 = typeid(fixed_int16m2_t);
auto &fs32m2 = typeid(fixed_int32m2_t);
auto &fs64m2 = typeid(fixed_int64m2_t);

auto &fu8m2 = typeid(fixed_uint8m2_t);
auto &fu16m2 = typeid(fixed_uint16m2_t);
auto &fu32m2 = typeid(fixed_uint32m2_t);
auto &fu64m2 = typeid(fixed_uint64m2_t);

auto &ff32m2 = typeid(fixed_float32m2_t);
auto &ff64m2 = typeid(fixed_float64m2_t);

auto &fs8m4 = typeid(fixed_int8m4_t);
auto &fs16m4 = typeid(fixed_int16m4_t);
auto &fs32m4 = typeid(fixed_int32m4_t);
auto &fs64m4 = typeid(fixed_int64m4_t);

auto &fu8m4 = typeid(fixed_uint8m4_t);
auto &fu16m4 = typeid(fixed_uint16m4_t);
auto &fu32m4 = typeid(fixed_uint32m4_t);
auto &fu64m4 = typeid(fixed_uint64m4_t);

auto &ff32m4 = typeid(fixed_float32m4_t);
auto &ff64m4 = typeid(fixed_float64m4_t);

auto &fs8m8 = typeid(fixed_int8m8_t);
auto &fs16m8 = typeid(fixed_int16m8_t);
auto &fs32m8 = typeid(fixed_int32m8_t);
auto &fs64m8 = typeid(fixed_int64m8_t);

auto &fu8m8 = typeid(fixed_uint8m8_t);
auto &fu16m8 = typeid(fixed_uint16m8_t);
auto &fu32m8 = typeid(fixed_uint32m8_t);
auto &fu64m8 = typeid(fixed_uint64m8_t);

auto &ff32m8 = typeid(fixed_float32m8_t);
auto &ff64m8 = typeid(fixed_float64m8_t);

auto &fs8mf2 = typeid(fixed_int8mf2_t);
auto &fs16mf2 = typeid(fixed_int16mf2_t);
auto &fs32mf2 = typeid(fixed_int32mf2_t);

auto &fu8mf2 = typeid(fixed_uint8mf2_t);
auto &fu16mf2 = typeid(fixed_uint16mf2_t);
auto &fu32mf2 = typeid(fixed_uint32mf2_t);

auto &ff32mf2 = typeid(fixed_float32mf2_t);

auto &fs8mf4 = typeid(fixed_int8mf4_t);
auto &fs16mf4 = typeid(fixed_int16mf4_t);

auto &fu8mf4 = typeid(fixed_uint8mf4_t);
auto &fu16mf4 = typeid(fixed_uint16mf4_t);

auto &fs8mf8 = typeid(fixed_int8mf8_t);

auto &fu8mf8 = typeid(fixed_uint8mf8_t);

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

// CHECK-64: @_ZTI9__RVV_VLSIu14__rvv_int8m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu14__rvv_int8m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu14__rvv_int8m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu14__rvv_int8m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu14__rvv_int8m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int16m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int16m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int16m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int16m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int16m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int32m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int32m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int32m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int32m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int32m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int64m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int64m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int64m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int64m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int64m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_uint8m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_uint8m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_uint8m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_uint8m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_uint8m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint16m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint16m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint16m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint16m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint16m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint32m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint32m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint32m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint32m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint32m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint64m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint64m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint64m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint64m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint64m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_float32m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_float32m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_float32m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_float32m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_float32m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_float64m2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m2_tLj128EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_float64m2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m2_tLj256EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_float64m2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m2_tLj512EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_float64m2_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m2_tLj1024EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_float64m2_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m2_tLj2048EE

// CHECK-64: @_ZTI9__RVV_VLSIu14__rvv_int8m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu14__rvv_int8m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu14__rvv_int8m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu14__rvv_int8m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu14__rvv_int8m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int16m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int16m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int16m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int16m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int16m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int32m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int32m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int32m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int32m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int32m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int64m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int64m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int64m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int64m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int64m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_uint8m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_uint8m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_uint8m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_uint8m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_uint8m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint16m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint16m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint16m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint16m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint16m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint32m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint32m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint32m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint32m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint32m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint64m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint64m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint64m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint64m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint64m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_float32m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_float32m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_float32m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_float32m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_float32m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_float64m4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m4_tLj256EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_float64m4_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m4_tLj512EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_float64m4_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m4_tLj1024EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_float64m4_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m4_tLj2048EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_float64m4_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m4_tLj4096EE

// CHECK-64: @_ZTI9__RVV_VLSIu14__rvv_int8m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu14__rvv_int8m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu14__rvv_int8m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu14__rvv_int8m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu14__rvv_int8m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu14__rvv_int8m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int16m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int16m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int16m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int16m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int16m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int16m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int32m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int32m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int32m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int32m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int32m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int32m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int64m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int64m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int64m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int64m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int64m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int64m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_uint8m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_uint8m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_uint8m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_uint8m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_uint8m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_uint8m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint16m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint16m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint16m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint16m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint16m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint16m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint32m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint32m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint32m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint32m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint32m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint32m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint64m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint64m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint64m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint64m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint64m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint64m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_float32m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_float32m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_float32m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_float32m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_float32m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float32m8_tLj8192EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_float64m8_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m8_tLj512EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_float64m8_tLj1024EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m8_tLj1024EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_float64m8_tLj2048EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m8_tLj2048EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_float64m8_tLj4096EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m8_tLj4096EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_float64m8_tLj8192EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_float64m8_tLj8192EE
//
// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int8mf2_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf2_tLj32EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int8mf2_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf2_tLj64EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int8mf2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf2_tLj128EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int8mf2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf2_tLj256EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int8mf2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf2_tLj512EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_int16mf2_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf2_tLj32EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_int16mf2_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf2_tLj64EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_int16mf2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf2_tLj128EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_int16mf2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf2_tLj256EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_int16mf2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf2_tLj512EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_int32mf2_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int32mf2_tLj32EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_int32mf2_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int32mf2_tLj64EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_int32mf2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int32mf2_tLj128EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_int32mf2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int32mf2_tLj256EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_int32mf2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int32mf2_tLj512EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint8mf2_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf2_tLj32EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint8mf2_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf2_tLj64EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint8mf2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf2_tLj128EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint8mf2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf2_tLj256EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint8mf2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf2_tLj512EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_uint16mf2_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf2_tLj32EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_uint16mf2_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf2_tLj64EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_uint16mf2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf2_tLj128EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_uint16mf2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf2_tLj256EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_uint16mf2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf2_tLj512EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_uint32mf2_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint32mf2_tLj32EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_uint32mf2_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint32mf2_tLj64EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_uint32mf2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint32mf2_tLj128EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_uint32mf2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint32mf2_tLj256EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_uint32mf2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint32mf2_tLj512EE

// CHECK-64: @_ZTI9__RVV_VLSIu18__rvv_float32mf2_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu18__rvv_float32mf2_tLj32EE
// CHECK-128: @_ZTI9__RVV_VLSIu18__rvv_float32mf2_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu18__rvv_float32mf2_tLj64EE
// CHECK-256: @_ZTI9__RVV_VLSIu18__rvv_float32mf2_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu18__rvv_float32mf2_tLj128EE
// CHECK-512: @_ZTI9__RVV_VLSIu18__rvv_float32mf2_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu18__rvv_float32mf2_tLj256EE
// CHECK-1024: @_ZTI9__RVV_VLSIu18__rvv_float32mf2_tLj512EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu18__rvv_float32mf2_tLj512EE
//
// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int8mf4_tLj16EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf4_tLj16EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int8mf4_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf4_tLj32EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int8mf4_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf4_tLj64EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int8mf4_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf4_tLj128EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int8mf4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf4_tLj256EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_int16mf4_tLj16EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf4_tLj16EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_int16mf4_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf4_tLj32EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_int16mf4_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf4_tLj64EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_int16mf4_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf4_tLj128EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_int16mf4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_int16mf4_tLj256EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint8mf4_tLj16EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf4_tLj16EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint8mf4_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf4_tLj32EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint8mf4_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf4_tLj64EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint8mf4_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf4_tLj128EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint8mf4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf4_tLj256EE

// CHECK-64: @_ZTI9__RVV_VLSIu17__rvv_uint16mf4_tLj16EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf4_tLj16EE
// CHECK-128: @_ZTI9__RVV_VLSIu17__rvv_uint16mf4_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf4_tLj32EE
// CHECK-256: @_ZTI9__RVV_VLSIu17__rvv_uint16mf4_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf4_tLj64EE
// CHECK-512: @_ZTI9__RVV_VLSIu17__rvv_uint16mf4_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf4_tLj128EE
// CHECK-1024: @_ZTI9__RVV_VLSIu17__rvv_uint16mf4_tLj256EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu17__rvv_uint16mf4_tLj256EE
//
// CHECK-64: @_ZTI9__RVV_VLSIu15__rvv_int8mf8_tLj8EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf8_tLj8EE
// CHECK-128: @_ZTI9__RVV_VLSIu15__rvv_int8mf8_tLj16EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf8_tLj16EE
// CHECK-256: @_ZTI9__RVV_VLSIu15__rvv_int8mf8_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf8_tLj32EE
// CHECK-512: @_ZTI9__RVV_VLSIu15__rvv_int8mf8_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf8_tLj64EE
// CHECK-1024: @_ZTI9__RVV_VLSIu15__rvv_int8mf8_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu15__rvv_int8mf8_tLj128EE

// CHECK-64: @_ZTI9__RVV_VLSIu16__rvv_uint8mf8_tLj8EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf8_tLj8EE
// CHECK-128: @_ZTI9__RVV_VLSIu16__rvv_uint8mf8_tLj16EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf8_tLj16EE
// CHECK-256: @_ZTI9__RVV_VLSIu16__rvv_uint8mf8_tLj32EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf8_tLj32EE
// CHECK-512: @_ZTI9__RVV_VLSIu16__rvv_uint8mf8_tLj64EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf8_tLj64EE
// CHECK-1024: @_ZTI9__RVV_VLSIu16__rvv_uint8mf8_tLj128EE = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTS9__RVV_VLSIu16__rvv_uint8mf8_tLj128EE
