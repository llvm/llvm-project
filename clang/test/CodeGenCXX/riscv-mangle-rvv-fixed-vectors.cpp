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
