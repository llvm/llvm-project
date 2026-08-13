// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -fcuda-is-device \
// RUN:   -fsyntax-only -verify %s

#define __global__ __attribute__((global))

typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
struct Agg { int a, b; };

__global__ void ok(f32x8 *o, i32x4 *v, int *s) {
  *o = __builtin_amdgcn_pin_vgpr(*o, 256);
  *v = __builtin_amdgcn_pin_agpr(*v, 8);
  *s = __builtin_amdgcn_pin_vgpr(*s, 0);
  // The result has the type of the value, so it composes without a cast.
  *o = __builtin_amdgcn_pin_vgpr(*o, 264) + *o;
}

__global__ void bad(f32x8 *o, int n) {
  __builtin_amdgcn_pin_vgpr(*o);           // expected-error {{too few arguments to function call, expected 2, have 1}}
  __builtin_amdgcn_pin_vgpr(*o, 256, 1);   // expected-error {{too many arguments to function call, expected 2, have 3}}
  __builtin_amdgcn_pin_vgpr(*o, n);        // expected-error {{argument to '__builtin_amdgcn_pin_vgpr' must be a constant integer}}

  // A register number the file cannot name is rejected; whether a given target
  // and occupancy can reach one that it can name is left to the backend.
  __builtin_amdgcn_pin_vgpr(*o, 1024);     // expected-error {{argument value 1024 is outside the valid range [0, 1023]}}
  __builtin_amdgcn_pin_agpr(*o, 256);      // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  __builtin_amdgcn_pin_vgpr(*o, -1);       // expected-error {{argument value -1 is outside the valid range [0, 1023]}}

  Agg a{1, 2};
  __builtin_amdgcn_pin_vgpr(a, 0);         // expected-error {{argument to '__builtin_amdgcn_pin_vgpr' must be a scalar or vector, but 'Agg' is not}}

  short h = 1;
  __builtin_amdgcn_pin_vgpr(h, 0);         // expected-error {{argument to '__builtin_amdgcn_pin_vgpr' must occupy a whole number of 32-bit registers, but 'short' is 16 bits}}
}

// A dependent argument is only checked once the template is instantiated.
template <int Reg, typename T> __global__ void tmpl(T *o) {
  *o = __builtin_amdgcn_pin_vgpr(*o, Reg); // expected-error {{argument value 2048 is outside the valid range [0, 1023]}}
}

template __global__ void tmpl<256, f32x8>(f32x8 *);
template __global__ void tmpl<2048, f32x8>(f32x8 *); // expected-note {{in instantiation of function template specialization 'tmpl<2048, float __attribute__((ext_vector_type(8)))>' requested here}}
