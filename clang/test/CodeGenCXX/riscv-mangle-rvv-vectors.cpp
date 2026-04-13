// RUN: %clang_cc1 -triple riscv32-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +experimental-zvfofp8min | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:  -target-feature +experimental-zvfofp8min | FileCheck %s

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

template <typename T> struct S {};

// CHECK: _Z8fe4m3mf81SIu21__rvv_float8e4m3mf8_tE
void fe4m3mf8(S<vfloat8e4m3mf8_t>) {}

// CHECK: _Z8fe4m3mf41SIu21__rvv_float8e4m3mf4_tE
void fe4m3mf4(S<vfloat8e4m3mf4_t>) {}

// CHECK: _Z8fe4m3mf21SIu21__rvv_float8e4m3mf2_tE
void fe4m3mf2(S<vfloat8e4m3mf2_t>) {}

// CHECK: _Z7fe4m3m11SIu20__rvv_float8e4m3m1_tE
void fe4m3m1(S<vfloat8e4m3m1_t>) {}

// CHECK: _Z7fe4m3m21SIu20__rvv_float8e4m3m2_tE
void fe4m3m2(S<vfloat8e4m3m2_t>) {}

// CHECK: _Z7fe4m3m41SIu20__rvv_float8e4m3m4_tE
void fe4m3m4(S<vfloat8e4m3m4_t>) {}

// CHECK: _Z7fe4m3m81SIu20__rvv_float8e4m3m8_tE
void fe4m3m8(S<vfloat8e4m3m8_t>) {}

// CHECK: _Z8fe5m2mf81SIu21__rvv_float8e5m2mf8_tE
void fe5m2mf8(S<vfloat8e5m2mf8_t>) {}

// CHECK: _Z8fe5m2mf41SIu21__rvv_float8e5m2mf4_tE
void fe5m2mf4(S<vfloat8e5m2mf4_t>) {}

// CHECK: _Z8fe5m2mf21SIu21__rvv_float8e5m2mf2_tE
void fe5m2mf2(S<vfloat8e5m2mf2_t>) {}

// CHECK: _Z7fe5m2m11SIu20__rvv_float8e5m2m1_tE
void fe5m2m1(S<vfloat8e5m2m1_t>) {}

// CHECK: _Z7fe5m2m21SIu20__rvv_float8e5m2m2_tE
void fe5m2m2(S<vfloat8e5m2m2_t>) {}

// CHECK: _Z7fe5m2m41SIu20__rvv_float8e5m2m4_tE
void fe5m2m4(S<vfloat8e5m2m4_t>) {}

// CHECK: _Z7fe5m2m81SIu20__rvv_float8e5m2m8_tE
void fe5m2m8(S<vfloat8e5m2m8_t>) {}
