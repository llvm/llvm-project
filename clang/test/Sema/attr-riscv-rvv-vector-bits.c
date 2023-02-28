// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -ffreestanding -fsyntax-only -verify %s

// TODO: Support for a arm_sve_vector_bits like attribute will come in the future.

#include <stdint.h>

#define N 64

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

// GNU vector types
typedef int8_t gnu_int8_t __attribute__((vector_size(N / 8)));
typedef int16_t gnu_int16_t __attribute__((vector_size(N / 8)));
typedef int32_t gnu_int32_t __attribute__((vector_size(N / 8)));
typedef int64_t gnu_int64_t __attribute__((vector_size(N / 8)));

typedef uint8_t gnu_uint8_t __attribute__((vector_size(N / 8)));
typedef uint16_t gnu_uint16_t __attribute__((vector_size(N / 8)));
typedef uint32_t gnu_uint32_t __attribute__((vector_size(N / 8)));
typedef uint64_t gnu_uint64_t __attribute__((vector_size(N / 8)));

typedef float gnu_float32_t __attribute__((vector_size(N / 8)));
typedef double gnu_float64_t __attribute__((vector_size(N / 8)));


void f(int c) {
  vint8m1_t ss8;
  gnu_int8_t gs8;

  // Check conditional expressions where the result is ambiguous are
  // ill-formed.
  void *sel __attribute__((unused));

  sel = c ? gs8 : ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}
  sel = c ? ss8 : gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  // Check binary expressions where the result is ambiguous are ill-formed.
  ss8 = ss8 + gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  gs8 = gs8 + ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  ss8 += gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  gs8 += ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  ss8 = ss8 == gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  gs8 = gs8 == ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  ss8 = ss8 & gs8; // expected-error {{invalid operands to binary expression ('vint8m1_t' (aka '__rvv_int8m1_t') and 'gnu_int8_t' (vector of 8 'int8_t' values))}}

  gs8 = gs8 & ss8; // expected-error {{invalid operands to binary expression ('gnu_int8_t' (vector of 8 'int8_t' values) and 'vint8m1_t' (aka '__rvv_int8m1_t'))}}
}

// --------------------------------------------------------------------------//
// Implicit casts

gnu_int8_t to_gnu_int8_t_from_vint8m1_t_(vint8m1_t x) { return x; } // expected-error {{returning 'vint8m1_t' (aka '__rvv_int8m1_t') from a function with incompatible result type 'gnu_int8_t' (vector of 8 'int8_t' values)}}
vint8m1_t from_gnu_int8_t_to_vint8m1_t(gnu_int8_t x) { return x; } // expected-error {{returning 'gnu_int8_t' (vector of 8 'int8_t' values) from a function with incompatible result type 'vint8m1_t' (aka '__rvv_int8m1_t')}}

// --------------------------------------------------------------------------//
// Test passing GNU vector scalable function

vint32m1_t __attribute__((overloadable)) vfunc(vint32m1_t op1, vint32m1_t op2);
vfloat64m1_t __attribute__((overloadable)) vfunc(vfloat64m1_t op1, vfloat64m1_t op2);

gnu_int32_t call_int32_ff(gnu_int32_t op1, gnu_int32_t op2) {
  return vfunc(op1, op2); // expected-error {{no matching function for call to 'vfunc'}}
                          // expected-note@-5 {{candidate function not viable: no known conversion from 'gnu_int32_t' (vector of 2 'int32_t' values) to 'vint32m1_t' (aka '__rvv_int32m1_t') for 1st argument}}
                          // expected-note@-5 {{candidate function not viable: no known conversion from 'gnu_int32_t' (vector of 2 'int32_t' values) to 'vfloat64m1_t' (aka '__rvv_float64m1_t') for 1st argument}}
}
