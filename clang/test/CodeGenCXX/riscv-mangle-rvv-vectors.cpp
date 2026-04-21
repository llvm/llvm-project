// RUN: %clang_cc1 -triple riscv32-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +experimental-zvfofp8min -target-feature +zve64x | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +experimental-zvfofp8min -target-feature +zve64x | FileCheck %s

typedef __rvv_float8e4m3mf8_t vfloat8e4m3mf8_t;
typedef __rvv_float8e4m3mf4_t vfloat8e4m3mf4_t;
typedef __rvv_float8e4m3mf2_t vfloat8e4m3mf2_t;
typedef __rvv_float8e4m3m1_t  vfloat8e4m3m1_t;
typedef __rvv_float8e4m3m2_t  vfloat8e4m3m2_t;
typedef __rvv_float8e4m3m4_t  vfloat8e4m3m4_t;
typedef __rvv_float8e4m3m8_t  vfloat8e4m3m8_t;

typedef __rvv_float8e5m2mf8_t vfloat8e5m2mf8_t;
typedef __rvv_float8e5m2mf4_t vfloat8e5m2mf4_t;
typedef __rvv_float8e5m2mf2_t vfloat8e5m2mf2_t;
typedef __rvv_float8e5m2m1_t  vfloat8e5m2m1_t;
typedef __rvv_float8e5m2m2_t  vfloat8e5m2m2_t;
typedef __rvv_float8e5m2m4_t  vfloat8e5m2m4_t;
typedef __rvv_float8e5m2m8_t  vfloat8e5m2m8_t;

// CHECK: _Z8fe4m3mf8u21__rvv_float8e4m3mf8_t
void fe4m3mf8(vfloat8e4m3mf8_t) {}

// CHECK: _Z8fe4m3mf4u21__rvv_float8e4m3mf4_t
void fe4m3mf4(vfloat8e4m3mf4_t) {}

// CHECK: _Z8fe4m3mf2u21__rvv_float8e4m3mf2_t
void fe4m3mf2(vfloat8e4m3mf2_t) {}

// CHECK: _Z7fe4m3m1u20__rvv_float8e4m3m1_t
void fe4m3m1(vfloat8e4m3m1_t) {}

// CHECK: _Z7fe4m3m2u20__rvv_float8e4m3m2_t
void fe4m3m2(vfloat8e4m3m2_t) {}

// CHECK: _Z7fe4m3m4u20__rvv_float8e4m3m4_t
void fe4m3m4(vfloat8e4m3m4_t) {}

// CHECK: _Z7fe4m3m8u20__rvv_float8e4m3m8_t
void fe4m3m8(vfloat8e4m3m8_t) {}

// CHECK: _Z8fe5m2mf8u21__rvv_float8e5m2mf8_t
void fe5m2mf8(vfloat8e5m2mf8_t) {}

// CHECK: _Z8fe5m2mf4u21__rvv_float8e5m2mf4_t
void fe5m2mf4(vfloat8e5m2mf4_t) {}

// CHECK: _Z8fe5m2mf2u21__rvv_float8e5m2mf2_t
void fe5m2mf2(vfloat8e5m2mf2_t) {}

// CHECK: _Z7fe5m2m1u20__rvv_float8e5m2m1_t
void fe5m2m1(vfloat8e5m2m1_t) {}

// CHECK: _Z7fe5m2m2u20__rvv_float8e5m2m2_t
void fe5m2m2(vfloat8e5m2m2_t) {}

// CHECK: _Z7fe5m2m4u20__rvv_float8e5m2m4_t
void fe5m2m4(vfloat8e5m2m4_t) {}

// CHECK: _Z7fe5m2m8u20__rvv_float8e5m2m8_t
void fe5m2m8(vfloat8e5m2m8_t) {}
