// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -c -O2 -emit-llvm %S/external_library.c -o %t
// RUN: %clang -Wall -Wextra -Wripple -Wpedantic -g -O2 -fenable-ripple -emit-llvm -S -ffast-math -fripple-lib %t %s -ferror-limit=500 2>%t.err; FileCheck %s --input-file=%t.err

#include <ripple.h>

// We need to allow ripple functions returning multiple values to support these
#define sincosf16(x, y, z) __builtin_ripple_sincosf16((x), (y), (z))
#define sincosf(x, y, z) __builtin_ripple_sincosf((x), (y), (z))
#define sincos(x, y, z) __builtin_ripple_sincos((x), (y), (z))
#define sincosl(x, y, z) __builtin_ripple_sincosl((x), (y), (z))

#define modff16(x, y) __builtin_ripple_modff16((x), (y))
#define modff(x, y) __builtin_ripple_modff((x), (y))
#define modf(x, y) __builtin_ripple_modf((x), (y))
#define modfl(x, y) __builtin_ripple_modfl((x), (y))

#define frexpf16(x, y) __builtin_ripple_frexpf16((x), (y))
#define frexpf(x, y) __builtin_ripple_frexpf((x), (y))
#define frexp(x, y) __builtin_ripple_frexp((x), (y))
#define frexpl(x, y) __builtin_ripple_frexpl((x), (y))

#define N 128


void check_sincosf16(const _Float16 x[N], _Float16 y[N], _Float16 z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  sincosf16(x[v0], &y[v0], &z[v0]);
}


void check_sincosf(const float x[N], float y[N], float z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  sincosf(x[v0], &y[v0], &z[v0]);
}


void check_sincos(const double x[N], double y[N], double z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  sincos(x[v0], &y[v0], &z[v0]);
}


void check_sincosl(const long double x[N], long double y[N], long double z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  sincosl(x[v0], &y[v0], &z[v0]);
}



void check_modff16(const _Float16 x[N], _Float16 y[N], _Float16 z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = modff16(x[v0], &y[v0]);
}


void check_modff(const float x[N], float y[N], float z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = modff(x[v0], &y[v0]);
}


void check_modf(const double x[N], double y[N], double z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = modf(x[v0], &y[v0]);
}


void check_modfl(const long double x[N], long double y[N], long double z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = modfl(x[v0], &y[v0]);
}



void check_frexpf16(const _Float16 x[N], int y[N], _Float16 z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = frexpf16(x[v0], &y[v0]);
}


void check_frexpf(const float x[N], int y[N], float z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = frexpf(x[v0], &y[v0]);
}


void check_frexp(const double x[N], int y[N], double z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = frexp(x[v0], &y[v0]);
}


void check_frexpl(const long double x[N], int y[N], long double z[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, N);
  int v0 = ripple_id(BS, 0);
  z[v0] = frexpl(x[v0], &y[v0]);
}

// CHECK: 108:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 101:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 94:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 87:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 79:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 72:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 65:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 58:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 50:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 43:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 36:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
// CHECK: 29:{{.*}}Ripple cannot create a vector type from this instruction's structure type; Allowed vector element types are integer, floating point and pointer
