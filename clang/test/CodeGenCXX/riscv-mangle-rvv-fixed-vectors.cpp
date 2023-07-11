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

template <typename T> struct S {};

// CHECK-64: _Z2f11SI9__RVV_VLSIu14__rvv_int8m1_tLj64EEE
// CHECK-128: _Z2f11SI9__RVV_VLSIu14__rvv_int8m1_tLj128EEE
// CHECK-256: _Z2f11SI9__RVV_VLSIu14__rvv_int8m1_tLj256EEE
// CHECK-512: _Z2f11SI9__RVV_VLSIu14__rvv_int8m1_tLj512EEE
// CHECK-1024: _Z2f11SI9__RVV_VLSIu14__rvv_int8m1_tLj1024EEE
void f1(S<fixed_int8m1_t>) {}

// CHECK-64: _Z2f21SI9__RVV_VLSIu15__rvv_int16m1_tLj64EEE
// CHECK-128: _Z2f21SI9__RVV_VLSIu15__rvv_int16m1_tLj128EEE
// CHECK-256: _Z2f21SI9__RVV_VLSIu15__rvv_int16m1_tLj256EEE
// CHECK-512: _Z2f21SI9__RVV_VLSIu15__rvv_int16m1_tLj512EEE
// CHECK-1024: _Z2f21SI9__RVV_VLSIu15__rvv_int16m1_tLj1024EEE
void f2(S<fixed_int16m1_t>) {}

// CHECK-64: _Z2f31SI9__RVV_VLSIu15__rvv_int32m1_tLj64EEE
// CHECK-128: _Z2f31SI9__RVV_VLSIu15__rvv_int32m1_tLj128EEE
// CHECK-256: _Z2f31SI9__RVV_VLSIu15__rvv_int32m1_tLj256EEE
// CHECK-512: _Z2f31SI9__RVV_VLSIu15__rvv_int32m1_tLj512EEE
// CHECK-1024: _Z2f31SI9__RVV_VLSIu15__rvv_int32m1_tLj1024EEE
void f3(S<fixed_int32m1_t>) {}

// CHECK-64: _Z2f41SI9__RVV_VLSIu15__rvv_int64m1_tLj64EEE
// CHECK-128: _Z2f41SI9__RVV_VLSIu15__rvv_int64m1_tLj128EEE
// CHECK-256: _Z2f41SI9__RVV_VLSIu15__rvv_int64m1_tLj256EEE
// CHECK-512: _Z2f41SI9__RVV_VLSIu15__rvv_int64m1_tLj512EEE
// CHECK-1024: _Z2f41SI9__RVV_VLSIu15__rvv_int64m1_tLj1024EEE
void f4(S<fixed_int64m1_t>) {}

// CHECK-64: _Z2f51SI9__RVV_VLSIu15__rvv_uint8m1_tLj64EEE
// CHECK-128: _Z2f51SI9__RVV_VLSIu15__rvv_uint8m1_tLj128EEE
// CHECK-256: _Z2f51SI9__RVV_VLSIu15__rvv_uint8m1_tLj256EEE
// CHECK-512: _Z2f51SI9__RVV_VLSIu15__rvv_uint8m1_tLj512EEE
// CHECK-1024: _Z2f51SI9__RVV_VLSIu15__rvv_uint8m1_tLj1024EEE
void f5(S<fixed_uint8m1_t>) {}

// CHECK-64: _Z2f61SI9__RVV_VLSIu16__rvv_uint16m1_tLj64EEE
// CHECK-128: _Z2f61SI9__RVV_VLSIu16__rvv_uint16m1_tLj128EEE
// CHECK-256: _Z2f61SI9__RVV_VLSIu16__rvv_uint16m1_tLj256EEE
// CHECK-512: _Z2f61SI9__RVV_VLSIu16__rvv_uint16m1_tLj512EEE
// CHECK-1024: _Z2f61SI9__RVV_VLSIu16__rvv_uint16m1_tLj1024EEE
void f6(S<fixed_uint16m1_t>) {}

// CHECK-64: _Z2f71SI9__RVV_VLSIu16__rvv_uint32m1_tLj64EEE
// CHECK-128: _Z2f71SI9__RVV_VLSIu16__rvv_uint32m1_tLj128EEE
// CHECK-256: _Z2f71SI9__RVV_VLSIu16__rvv_uint32m1_tLj256EEE
// CHECK-512: _Z2f71SI9__RVV_VLSIu16__rvv_uint32m1_tLj512EEE
// CHECK-1024: _Z2f71SI9__RVV_VLSIu16__rvv_uint32m1_tLj1024EEE
void f7(S<fixed_uint32m1_t>) {}

// CHECK-64: _Z2f81SI9__RVV_VLSIu16__rvv_uint64m1_tLj64EEE
// CHECK-128: _Z2f81SI9__RVV_VLSIu16__rvv_uint64m1_tLj128EEE
// CHECK-256: _Z2f81SI9__RVV_VLSIu16__rvv_uint64m1_tLj256EEE
// CHECK-512: _Z2f81SI9__RVV_VLSIu16__rvv_uint64m1_tLj512EEE
// CHECK-1024: _Z2f81SI9__RVV_VLSIu16__rvv_uint64m1_tLj1024EEE
void f8(S<fixed_uint64m1_t>) {}

// CHECK-64: _Z2f91SI9__RVV_VLSIu17__rvv_float32m1_tLj64EEE
// CHECK-128: _Z2f91SI9__RVV_VLSIu17__rvv_float32m1_tLj128EEE
// CHECK-256: _Z2f91SI9__RVV_VLSIu17__rvv_float32m1_tLj256EEE
// CHECK-512: _Z2f91SI9__RVV_VLSIu17__rvv_float32m1_tLj512EEE
// CHECK-1024: _Z2f91SI9__RVV_VLSIu17__rvv_float32m1_tLj1024EEE
void f9(S<fixed_float32m1_t>) {}

// CHECK-64: _Z3f101SI9__RVV_VLSIu17__rvv_float64m1_tLj64EEE
// CHECK-128: _Z3f101SI9__RVV_VLSIu17__rvv_float64m1_tLj128EEE
// CHECK-256: _Z3f101SI9__RVV_VLSIu17__rvv_float64m1_tLj256EEE
// CHECK-512: _Z3f101SI9__RVV_VLSIu17__rvv_float64m1_tLj512EEE
// CHECK-1024: _Z3f101SI9__RVV_VLSIu17__rvv_float64m1_tLj1024EEE
void f10(S<fixed_float64m1_t>) {}

// CHECK-64: _Z4m2f11SI9__RVV_VLSIu14__rvv_int8m2_tLj128EEE
// CHECK-128: _Z4m2f11SI9__RVV_VLSIu14__rvv_int8m2_tLj256EEE
// CHECK-256: _Z4m2f11SI9__RVV_VLSIu14__rvv_int8m2_tLj512EEE
// CHECK-512: _Z4m2f11SI9__RVV_VLSIu14__rvv_int8m2_tLj1024EEE
// CHECK-1024: _Z4m2f11SI9__RVV_VLSIu14__rvv_int8m2_tLj2048EEE
void m2f1(S<fixed_int8m2_t>) {}

// CHECK-64: _Z4m2f21SI9__RVV_VLSIu15__rvv_int16m2_tLj128EEE
// CHECK-128: _Z4m2f21SI9__RVV_VLSIu15__rvv_int16m2_tLj256EEE
// CHECK-256: _Z4m2f21SI9__RVV_VLSIu15__rvv_int16m2_tLj512EEE
// CHECK-512: _Z4m2f21SI9__RVV_VLSIu15__rvv_int16m2_tLj1024EEE
// CHECK-1024: _Z4m2f21SI9__RVV_VLSIu15__rvv_int16m2_tLj2048EEE
void m2f2(S<fixed_int16m2_t>) {}

// CHECK-64: _Z4m2f31SI9__RVV_VLSIu15__rvv_int32m2_tLj128EEE
// CHECK-128: _Z4m2f31SI9__RVV_VLSIu15__rvv_int32m2_tLj256EEE
// CHECK-256: _Z4m2f31SI9__RVV_VLSIu15__rvv_int32m2_tLj512EEE
// CHECK-512: _Z4m2f31SI9__RVV_VLSIu15__rvv_int32m2_tLj1024EEE
// CHECK-1024: _Z4m2f31SI9__RVV_VLSIu15__rvv_int32m2_tLj2048EEE
void m2f3(S<fixed_int32m2_t>) {}

// CHECK-64: _Z4m2f41SI9__RVV_VLSIu15__rvv_int64m2_tLj128EEE
// CHECK-128: _Z4m2f41SI9__RVV_VLSIu15__rvv_int64m2_tLj256EEE
// CHECK-256: _Z4m2f41SI9__RVV_VLSIu15__rvv_int64m2_tLj512EEE
// CHECK-512: _Z4m2f41SI9__RVV_VLSIu15__rvv_int64m2_tLj1024EEE
// CHECK-1024: _Z4m2f41SI9__RVV_VLSIu15__rvv_int64m2_tLj2048EEE
void m2f4(S<fixed_int64m2_t>) {}

// CHECK-64: _Z4m2f51SI9__RVV_VLSIu15__rvv_uint8m2_tLj128EEE
// CHECK-128: _Z4m2f51SI9__RVV_VLSIu15__rvv_uint8m2_tLj256EEE
// CHECK-256: _Z4m2f51SI9__RVV_VLSIu15__rvv_uint8m2_tLj512EEE
// CHECK-512: _Z4m2f51SI9__RVV_VLSIu15__rvv_uint8m2_tLj1024EEE
// CHECK-1024: _Z4m2f51SI9__RVV_VLSIu15__rvv_uint8m2_tLj2048EEE
void m2f5(S<fixed_uint8m2_t>) {}

// CHECK-64: _Z4m2f61SI9__RVV_VLSIu16__rvv_uint16m2_tLj128EEE
// CHECK-128: _Z4m2f61SI9__RVV_VLSIu16__rvv_uint16m2_tLj256EEE
// CHECK-256: _Z4m2f61SI9__RVV_VLSIu16__rvv_uint16m2_tLj512EEE
// CHECK-512: _Z4m2f61SI9__RVV_VLSIu16__rvv_uint16m2_tLj1024EEE
// CHECK-1024: _Z4m2f61SI9__RVV_VLSIu16__rvv_uint16m2_tLj2048EEE
void m2f6(S<fixed_uint16m2_t>) {}

// CHECK-64: _Z4m2f71SI9__RVV_VLSIu16__rvv_uint32m2_tLj128EEE
// CHECK-128: _Z4m2f71SI9__RVV_VLSIu16__rvv_uint32m2_tLj256EEE
// CHECK-256: _Z4m2f71SI9__RVV_VLSIu16__rvv_uint32m2_tLj512EEE
// CHECK-512: _Z4m2f71SI9__RVV_VLSIu16__rvv_uint32m2_tLj1024EEE
// CHECK-1024: _Z4m2f71SI9__RVV_VLSIu16__rvv_uint32m2_tLj2048EEE
void m2f7(S<fixed_uint32m2_t>) {}

// CHECK-64: _Z4m2f81SI9__RVV_VLSIu16__rvv_uint64m2_tLj128EEE
// CHECK-128: _Z4m2f81SI9__RVV_VLSIu16__rvv_uint64m2_tLj256EEE
// CHECK-256: _Z4m2f81SI9__RVV_VLSIu16__rvv_uint64m2_tLj512EEE
// CHECK-512: _Z4m2f81SI9__RVV_VLSIu16__rvv_uint64m2_tLj1024EEE
// CHECK-1024: _Z4m2f81SI9__RVV_VLSIu16__rvv_uint64m2_tLj2048EEE
void m2f8(S<fixed_uint64m2_t>) {}

// CHECK-64: _Z4m2f91SI9__RVV_VLSIu17__rvv_float32m2_tLj128EEE
// CHECK-128: _Z4m2f91SI9__RVV_VLSIu17__rvv_float32m2_tLj256EEE
// CHECK-256: _Z4m2f91SI9__RVV_VLSIu17__rvv_float32m2_tLj512EEE
// CHECK-512: _Z4m2f91SI9__RVV_VLSIu17__rvv_float32m2_tLj1024EEE
// CHECK-1024: _Z4m2f91SI9__RVV_VLSIu17__rvv_float32m2_tLj2048EEE
void m2f9(S<fixed_float32m2_t>) {}

// CHECK-64: _Z5m2f101SI9__RVV_VLSIu17__rvv_float64m2_tLj128EEE
// CHECK-128: _Z5m2f101SI9__RVV_VLSIu17__rvv_float64m2_tLj256EEE
// CHECK-256: _Z5m2f101SI9__RVV_VLSIu17__rvv_float64m2_tLj512EEE
// CHECK-512: _Z5m2f101SI9__RVV_VLSIu17__rvv_float64m2_tLj1024EEE
// CHECK-1024: _Z5m2f101SI9__RVV_VLSIu17__rvv_float64m2_tLj2048EEE
void m2f10(S<fixed_float64m2_t>) {}

// CHECK-64: _Z4m4f11SI9__RVV_VLSIu14__rvv_int8m4_tLj256EEE
// CHECK-128: _Z4m4f11SI9__RVV_VLSIu14__rvv_int8m4_tLj512EEE
// CHECK-256: _Z4m4f11SI9__RVV_VLSIu14__rvv_int8m4_tLj1024EEE
// CHECK-512: _Z4m4f11SI9__RVV_VLSIu14__rvv_int8m4_tLj2048EEE
// CHECK-1024: _Z4m4f11SI9__RVV_VLSIu14__rvv_int8m4_tLj4096EEE
void m4f1(S<fixed_int8m4_t>) {}

// CHECK-64: _Z4m4f21SI9__RVV_VLSIu15__rvv_int16m4_tLj256EEE
// CHECK-128: _Z4m4f21SI9__RVV_VLSIu15__rvv_int16m4_tLj512EEE
// CHECK-256: _Z4m4f21SI9__RVV_VLSIu15__rvv_int16m4_tLj1024EEE
// CHECK-512: _Z4m4f21SI9__RVV_VLSIu15__rvv_int16m4_tLj2048EEE
// CHECK-1024: _Z4m4f21SI9__RVV_VLSIu15__rvv_int16m4_tLj4096EEE
void m4f2(S<fixed_int16m4_t>) {}

// CHECK-64: _Z4m4f31SI9__RVV_VLSIu15__rvv_int32m4_tLj256EEE
// CHECK-128: _Z4m4f31SI9__RVV_VLSIu15__rvv_int32m4_tLj512EEE
// CHECK-256: _Z4m4f31SI9__RVV_VLSIu15__rvv_int32m4_tLj1024EEE
// CHECK-512: _Z4m4f31SI9__RVV_VLSIu15__rvv_int32m4_tLj2048EEE
// CHECK-1024: _Z4m4f31SI9__RVV_VLSIu15__rvv_int32m4_tLj4096EEE
void m4f3(S<fixed_int32m4_t>) {}

// CHECK-64: _Z4m4f41SI9__RVV_VLSIu15__rvv_int64m4_tLj256EEE
// CHECK-128: _Z4m4f41SI9__RVV_VLSIu15__rvv_int64m4_tLj512EEE
// CHECK-256: _Z4m4f41SI9__RVV_VLSIu15__rvv_int64m4_tLj1024EEE
// CHECK-512: _Z4m4f41SI9__RVV_VLSIu15__rvv_int64m4_tLj2048EEE
// CHECK-1024: _Z4m4f41SI9__RVV_VLSIu15__rvv_int64m4_tLj4096EEE
void m4f4(S<fixed_int64m4_t>) {}

// CHECK-64: _Z4m4f51SI9__RVV_VLSIu15__rvv_uint8m4_tLj256EEE
// CHECK-128: _Z4m4f51SI9__RVV_VLSIu15__rvv_uint8m4_tLj512EEE
// CHECK-256: _Z4m4f51SI9__RVV_VLSIu15__rvv_uint8m4_tLj1024EEE
// CHECK-512: _Z4m4f51SI9__RVV_VLSIu15__rvv_uint8m4_tLj2048EEE
// CHECK-1024: _Z4m4f51SI9__RVV_VLSIu15__rvv_uint8m4_tLj4096EEE
void m4f5(S<fixed_uint8m4_t>) {}

// CHECK-64: _Z4m4f61SI9__RVV_VLSIu16__rvv_uint16m4_tLj256EEE
// CHECK-128: _Z4m4f61SI9__RVV_VLSIu16__rvv_uint16m4_tLj512EEE
// CHECK-256: _Z4m4f61SI9__RVV_VLSIu16__rvv_uint16m4_tLj1024EEE
// CHECK-512: _Z4m4f61SI9__RVV_VLSIu16__rvv_uint16m4_tLj2048EEE
// CHECK-1024: _Z4m4f61SI9__RVV_VLSIu16__rvv_uint16m4_tLj4096EEE
void m4f6(S<fixed_uint16m4_t>) {}

// CHECK-64: _Z4m4f71SI9__RVV_VLSIu16__rvv_uint32m4_tLj256EEE
// CHECK-128: _Z4m4f71SI9__RVV_VLSIu16__rvv_uint32m4_tLj512EEE
// CHECK-256: _Z4m4f71SI9__RVV_VLSIu16__rvv_uint32m4_tLj1024EEE
// CHECK-512: _Z4m4f71SI9__RVV_VLSIu16__rvv_uint32m4_tLj2048EEE
// CHECK-1024: _Z4m4f71SI9__RVV_VLSIu16__rvv_uint32m4_tLj4096EEE
void m4f7(S<fixed_uint32m4_t>) {}

// CHECK-64: _Z4m4f81SI9__RVV_VLSIu16__rvv_uint64m4_tLj256EEE
// CHECK-128: _Z4m4f81SI9__RVV_VLSIu16__rvv_uint64m4_tLj512EEE
// CHECK-256: _Z4m4f81SI9__RVV_VLSIu16__rvv_uint64m4_tLj1024EEE
// CHECK-512: _Z4m4f81SI9__RVV_VLSIu16__rvv_uint64m4_tLj2048EEE
// CHECK-1024: _Z4m4f81SI9__RVV_VLSIu16__rvv_uint64m4_tLj4096EEE
void m4f8(S<fixed_uint64m4_t>) {}

// CHECK-64: _Z4m4f91SI9__RVV_VLSIu17__rvv_float32m4_tLj256EEE
// CHECK-128: _Z4m4f91SI9__RVV_VLSIu17__rvv_float32m4_tLj512EEE
// CHECK-256: _Z4m4f91SI9__RVV_VLSIu17__rvv_float32m4_tLj1024EEE
// CHECK-512: _Z4m4f91SI9__RVV_VLSIu17__rvv_float32m4_tLj2048EEE
// CHECK-1024: _Z4m4f91SI9__RVV_VLSIu17__rvv_float32m4_tLj4096EEE
void m4f9(S<fixed_float32m4_t>) {}

// CHECK-64: _Z5m4f101SI9__RVV_VLSIu17__rvv_float64m4_tLj256EEE
// CHECK-128: _Z5m4f101SI9__RVV_VLSIu17__rvv_float64m4_tLj512EEE
// CHECK-256: _Z5m4f101SI9__RVV_VLSIu17__rvv_float64m4_tLj1024EEE
// CHECK-512: _Z5m4f101SI9__RVV_VLSIu17__rvv_float64m4_tLj2048EEE
// CHECK-1024: _Z5m4f101SI9__RVV_VLSIu17__rvv_float64m4_tLj4096EEE
void m4f10(S<fixed_float64m4_t>) {}

// CHECK-64: _Z4m8f11SI9__RVV_VLSIu14__rvv_int8m8_tLj512EEE
// CHECK-128: _Z4m8f11SI9__RVV_VLSIu14__rvv_int8m8_tLj1024EEE
// CHECK-256: _Z4m8f11SI9__RVV_VLSIu14__rvv_int8m8_tLj2048EEE
// CHECK-512: _Z4m8f11SI9__RVV_VLSIu14__rvv_int8m8_tLj4096EEE
// CHECK-1024: _Z4m8f11SI9__RVV_VLSIu14__rvv_int8m8_tLj8192EEE
void m8f1(S<fixed_int8m8_t>) {}

// CHECK-64: _Z4m8f21SI9__RVV_VLSIu15__rvv_int16m8_tLj512EEE
// CHECK-128: _Z4m8f21SI9__RVV_VLSIu15__rvv_int16m8_tLj1024EEE
// CHECK-256: _Z4m8f21SI9__RVV_VLSIu15__rvv_int16m8_tLj2048EEE
// CHECK-512: _Z4m8f21SI9__RVV_VLSIu15__rvv_int16m8_tLj4096EEE
// CHECK-1024: _Z4m8f21SI9__RVV_VLSIu15__rvv_int16m8_tLj8192EEE
void m8f2(S<fixed_int16m8_t>) {}

// CHECK-64: _Z4m8f31SI9__RVV_VLSIu15__rvv_int32m8_tLj512EEE
// CHECK-128: _Z4m8f31SI9__RVV_VLSIu15__rvv_int32m8_tLj1024EEE
// CHECK-256: _Z4m8f31SI9__RVV_VLSIu15__rvv_int32m8_tLj2048EEE
// CHECK-512: _Z4m8f31SI9__RVV_VLSIu15__rvv_int32m8_tLj4096EEE
// CHECK-1024: _Z4m8f31SI9__RVV_VLSIu15__rvv_int32m8_tLj8192EEE
void m8f3(S<fixed_int32m8_t>) {}

// CHECK-64: _Z4m8f41SI9__RVV_VLSIu15__rvv_int64m8_tLj512EEE
// CHECK-128: _Z4m8f41SI9__RVV_VLSIu15__rvv_int64m8_tLj1024EEE
// CHECK-256: _Z4m8f41SI9__RVV_VLSIu15__rvv_int64m8_tLj2048EEE
// CHECK-512: _Z4m8f41SI9__RVV_VLSIu15__rvv_int64m8_tLj4096EEE
// CHECK-1024: _Z4m8f41SI9__RVV_VLSIu15__rvv_int64m8_tLj8192EEE
void m8f4(S<fixed_int64m8_t>) {}

// CHECK-64: _Z4m8f51SI9__RVV_VLSIu15__rvv_uint8m8_tLj512EEE
// CHECK-128: _Z4m8f51SI9__RVV_VLSIu15__rvv_uint8m8_tLj1024EEE
// CHECK-256: _Z4m8f51SI9__RVV_VLSIu15__rvv_uint8m8_tLj2048EEE
// CHECK-512: _Z4m8f51SI9__RVV_VLSIu15__rvv_uint8m8_tLj4096EEE
// CHECK-1024: _Z4m8f51SI9__RVV_VLSIu15__rvv_uint8m8_tLj8192EEE
void m8f5(S<fixed_uint8m8_t>) {}

// CHECK-64: _Z4m8f61SI9__RVV_VLSIu16__rvv_uint16m8_tLj512EEE
// CHECK-128: _Z4m8f61SI9__RVV_VLSIu16__rvv_uint16m8_tLj1024EEE
// CHECK-256: _Z4m8f61SI9__RVV_VLSIu16__rvv_uint16m8_tLj2048EEE
// CHECK-512: _Z4m8f61SI9__RVV_VLSIu16__rvv_uint16m8_tLj4096EEE
// CHECK-1024: _Z4m8f61SI9__RVV_VLSIu16__rvv_uint16m8_tLj8192EEE
void m8f6(S<fixed_uint16m8_t>) {}

// CHECK-64: _Z4m8f71SI9__RVV_VLSIu16__rvv_uint32m8_tLj512EEE
// CHECK-128: _Z4m8f71SI9__RVV_VLSIu16__rvv_uint32m8_tLj1024EEE
// CHECK-256: _Z4m8f71SI9__RVV_VLSIu16__rvv_uint32m8_tLj2048EEE
// CHECK-512: _Z4m8f71SI9__RVV_VLSIu16__rvv_uint32m8_tLj4096EEE
// CHECK-1024: _Z4m8f71SI9__RVV_VLSIu16__rvv_uint32m8_tLj8192EEE
void m8f7(S<fixed_uint32m8_t>) {}

// CHECK-64: _Z4m8f81SI9__RVV_VLSIu16__rvv_uint64m8_tLj512EEE
// CHECK-128: _Z4m8f81SI9__RVV_VLSIu16__rvv_uint64m8_tLj1024EEE
// CHECK-256: _Z4m8f81SI9__RVV_VLSIu16__rvv_uint64m8_tLj2048EEE
// CHECK-512: _Z4m8f81SI9__RVV_VLSIu16__rvv_uint64m8_tLj4096EEE
// CHECK-1024: _Z4m8f81SI9__RVV_VLSIu16__rvv_uint64m8_tLj8192EEE
void m8f8(S<fixed_uint64m8_t>) {}

// CHECK-64: _Z4m8f91SI9__RVV_VLSIu17__rvv_float32m8_tLj512EEE
// CHECK-128: _Z4m8f91SI9__RVV_VLSIu17__rvv_float32m8_tLj1024EEE
// CHECK-256: _Z4m8f91SI9__RVV_VLSIu17__rvv_float32m8_tLj2048EEE
// CHECK-512: _Z4m8f91SI9__RVV_VLSIu17__rvv_float32m8_tLj4096EEE
// CHECK-1024: _Z4m8f91SI9__RVV_VLSIu17__rvv_float32m8_tLj8192EEE
void m8f9(S<fixed_float32m8_t>) {}

// CHECK-64: _Z5m8f101SI9__RVV_VLSIu17__rvv_float64m8_tLj512EEE
// CHECK-128: _Z5m8f101SI9__RVV_VLSIu17__rvv_float64m8_tLj1024EEE
// CHECK-256: _Z5m8f101SI9__RVV_VLSIu17__rvv_float64m8_tLj2048EEE
// CHECK-512: _Z5m8f101SI9__RVV_VLSIu17__rvv_float64m8_tLj4096EEE
// CHECK-1024: _Z5m8f101SI9__RVV_VLSIu17__rvv_float64m8_tLj8192EEE
void m8f10(S<fixed_float64m8_t>) {}

// CHECK-64: _Z5mf2f11SI9__RVV_VLSIu15__rvv_int8mf2_tLj32EEE
// CHECK-128: _Z5mf2f11SI9__RVV_VLSIu15__rvv_int8mf2_tLj64EEE
// CHECK-256: _Z5mf2f11SI9__RVV_VLSIu15__rvv_int8mf2_tLj128EEE
// CHECK-512: _Z5mf2f11SI9__RVV_VLSIu15__rvv_int8mf2_tLj256EEE
// CHECK-1024: _Z5mf2f11SI9__RVV_VLSIu15__rvv_int8mf2_tLj512EEE
void mf2f1(S<fixed_int8mf2_t>) {}

// CHECK-64: _Z5mf2f21SI9__RVV_VLSIu16__rvv_int16mf2_tLj32EEE
// CHECK-128: _Z5mf2f21SI9__RVV_VLSIu16__rvv_int16mf2_tLj64EEE
// CHECK-256: _Z5mf2f21SI9__RVV_VLSIu16__rvv_int16mf2_tLj128EEE
// CHECK-512: _Z5mf2f21SI9__RVV_VLSIu16__rvv_int16mf2_tLj256EEE
// CHECK-1024: _Z5mf2f21SI9__RVV_VLSIu16__rvv_int16mf2_tLj512EEE
void mf2f2(S<fixed_int16mf2_t>) {}

// CHECK-64: _Z5mf2f31SI9__RVV_VLSIu16__rvv_int32mf2_tLj32EEE
// CHECK-128: _Z5mf2f31SI9__RVV_VLSIu16__rvv_int32mf2_tLj64EEE
// CHECK-256: _Z5mf2f31SI9__RVV_VLSIu16__rvv_int32mf2_tLj128EEE
// CHECK-512: _Z5mf2f31SI9__RVV_VLSIu16__rvv_int32mf2_tLj256EEE
// CHECK-1024: _Z5mf2f31SI9__RVV_VLSIu16__rvv_int32mf2_tLj512EEE
void mf2f3(S<fixed_int32mf2_t>) {}

// CHECK-64: _Z5mf2f51SI9__RVV_VLSIu16__rvv_uint8mf2_tLj32EEE
// CHECK-128: _Z5mf2f51SI9__RVV_VLSIu16__rvv_uint8mf2_tLj64EEE
// CHECK-256: _Z5mf2f51SI9__RVV_VLSIu16__rvv_uint8mf2_tLj128EEE
// CHECK-512: _Z5mf2f51SI9__RVV_VLSIu16__rvv_uint8mf2_tLj256EEE
// CHECK-1024: _Z5mf2f51SI9__RVV_VLSIu16__rvv_uint8mf2_tLj512EEE
void mf2f5(S<fixed_uint8mf2_t>) {}

// CHECK-64: _Z5mf2f61SI9__RVV_VLSIu17__rvv_uint16mf2_tLj32EEE
// CHECK-128: _Z5mf2f61SI9__RVV_VLSIu17__rvv_uint16mf2_tLj64EEE
// CHECK-256: _Z5mf2f61SI9__RVV_VLSIu17__rvv_uint16mf2_tLj128EEE
// CHECK-512: _Z5mf2f61SI9__RVV_VLSIu17__rvv_uint16mf2_tLj256EEE
// CHECK-1024: _Z5mf2f61SI9__RVV_VLSIu17__rvv_uint16mf2_tLj512EEE
void mf2f6(S<fixed_uint16mf2_t>) {}

// CHECK-64: _Z5mf2f71SI9__RVV_VLSIu17__rvv_uint32mf2_tLj32EEE
// CHECK-128: _Z5mf2f71SI9__RVV_VLSIu17__rvv_uint32mf2_tLj64EEE
// CHECK-256: _Z5mf2f71SI9__RVV_VLSIu17__rvv_uint32mf2_tLj128EEE
// CHECK-512: _Z5mf2f71SI9__RVV_VLSIu17__rvv_uint32mf2_tLj256EEE
// CHECK-1024: _Z5mf2f71SI9__RVV_VLSIu17__rvv_uint32mf2_tLj512EEE
void mf2f7(S<fixed_uint32mf2_t>) {}

// CHECK-64: _Z5mf2f91SI9__RVV_VLSIu18__rvv_float32mf2_tLj32EEE
// CHECK-128: _Z5mf2f91SI9__RVV_VLSIu18__rvv_float32mf2_tLj64EEE
// CHECK-256: _Z5mf2f91SI9__RVV_VLSIu18__rvv_float32mf2_tLj128EEE
// CHECK-512: _Z5mf2f91SI9__RVV_VLSIu18__rvv_float32mf2_tLj256EEE
// CHECK-1024: _Z5mf2f91SI9__RVV_VLSIu18__rvv_float32mf2_tLj512EEE
void mf2f9(S<fixed_float32mf2_t>) {}

// CHECK-64: _Z5mf4f11SI9__RVV_VLSIu15__rvv_int8mf4_tLj16EEE
// CHECK-128: _Z5mf4f11SI9__RVV_VLSIu15__rvv_int8mf4_tLj32EEE
// CHECK-256: _Z5mf4f11SI9__RVV_VLSIu15__rvv_int8mf4_tLj64EEE
// CHECK-512: _Z5mf4f11SI9__RVV_VLSIu15__rvv_int8mf4_tLj128EEE
// CHECK-1024: _Z5mf4f11SI9__RVV_VLSIu15__rvv_int8mf4_tLj256EEE
void mf4f1(S<fixed_int8mf4_t>) {}

// CHECK-64: _Z5mf4f21SI9__RVV_VLSIu16__rvv_int16mf4_tLj16EEE
// CHECK-128: _Z5mf4f21SI9__RVV_VLSIu16__rvv_int16mf4_tLj32EEE
// CHECK-256: _Z5mf4f21SI9__RVV_VLSIu16__rvv_int16mf4_tLj64EEE
// CHECK-512: _Z5mf4f21SI9__RVV_VLSIu16__rvv_int16mf4_tLj128EEE
// CHECK-1024: _Z5mf4f21SI9__RVV_VLSIu16__rvv_int16mf4_tLj256EEE
void mf4f2(S<fixed_int16mf4_t>) {}

// CHECK-64: _Z5mf4f51SI9__RVV_VLSIu16__rvv_uint8mf4_tLj16EEE
// CHECK-128: _Z5mf4f51SI9__RVV_VLSIu16__rvv_uint8mf4_tLj32EEE
// CHECK-256: _Z5mf4f51SI9__RVV_VLSIu16__rvv_uint8mf4_tLj64EEE
// CHECK-512: _Z5mf4f51SI9__RVV_VLSIu16__rvv_uint8mf4_tLj128EEE
// CHECK-1024: _Z5mf4f51SI9__RVV_VLSIu16__rvv_uint8mf4_tLj256EEE
void mf4f5(S<fixed_uint8mf4_t>) {}

// CHECK-64: _Z5mf4f61SI9__RVV_VLSIu17__rvv_uint16mf4_tLj16EEE
// CHECK-128: _Z5mf4f61SI9__RVV_VLSIu17__rvv_uint16mf4_tLj32EEE
// CHECK-256: _Z5mf4f61SI9__RVV_VLSIu17__rvv_uint16mf4_tLj64EEE
// CHECK-512: _Z5mf4f61SI9__RVV_VLSIu17__rvv_uint16mf4_tLj128EEE
// CHECK-1024: _Z5mf4f61SI9__RVV_VLSIu17__rvv_uint16mf4_tLj256EEE
void mf4f6(S<fixed_uint16mf4_t>) {}

// CHECK-64: _Z5mf8f11SI9__RVV_VLSIu15__rvv_int8mf8_tLj8EEE
// CHECK-128: _Z5mf8f11SI9__RVV_VLSIu15__rvv_int8mf8_tLj16EEE
// CHECK-256: _Z5mf8f11SI9__RVV_VLSIu15__rvv_int8mf8_tLj32EEE
// CHECK-512: _Z5mf8f11SI9__RVV_VLSIu15__rvv_int8mf8_tLj64EEE
// CHECK-1024: _Z5mf8f11SI9__RVV_VLSIu15__rvv_int8mf8_tLj128EEE
void mf8f1(S<fixed_int8mf8_t>) {}

// CHECK-64: _Z5mf8f51SI9__RVV_VLSIu16__rvv_uint8mf8_tLj8EEE
// CHECK-128: _Z5mf8f51SI9__RVV_VLSIu16__rvv_uint8mf8_tLj16EEE
// CHECK-256: _Z5mf8f51SI9__RVV_VLSIu16__rvv_uint8mf8_tLj32EEE
// CHECK-512: _Z5mf8f51SI9__RVV_VLSIu16__rvv_uint8mf8_tLj64EEE
// CHECK-1024: _Z5mf8f51SI9__RVV_VLSIu16__rvv_uint8mf8_tLj128EEE
void mf8f5(S<fixed_uint8mf8_t>) {}
