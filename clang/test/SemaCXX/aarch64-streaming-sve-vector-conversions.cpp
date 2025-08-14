// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=1 -mvscale-max=1 -mvscale-streaming-min=2 -mvscale-streaming-max=2 -flax-vector-conversions=integer -ffreestanding -fsyntax-only -verify %s
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

#define SVE_BITS 128
#define SVE_FIXED_ATTR __attribute__((arm_sve_vector_bits(SVE_BITS)))
#define GNU_FIXED_ATTR __attribute__((vector_size(SVE_BITS / 8)))
#define GNU_BOOL_FIXED_ATTR __attribute__((vector_size(SVE_BITS / 64)))
#define STREAMING_BITS 256
#define GNU_FIXED_STREAMING_ATTR __attribute__((vector_size(STREAMING_BITS / 8)))
#define GNU_BOOL_FIXED_STREAMING_ATTR __attribute__((vector_size(STREAMING_BITS / 64)))

typedef svfloat32_t sve_fixed_float32_t SVE_FIXED_ATTR;
typedef svint32_t sve_fixed_int32_t SVE_FIXED_ATTR;
typedef svbool_t sve_fixed_bool_t SVE_FIXED_ATTR;
typedef float gnu_fixed_float32_t GNU_FIXED_ATTR;
typedef int gnu_fixed_int32_t GNU_FIXED_ATTR;
typedef int8_t gnu_fixed_bool_t GNU_BOOL_FIXED_ATTR;

typedef float gnu_fixed_float32_t_streaming GNU_FIXED_STREAMING_ATTR;
typedef int gnu_fixed_int32_t_streaming GNU_FIXED_STREAMING_ATTR;
typedef int8_t gnu_fixed_bool_t_streaming GNU_BOOL_FIXED_STREAMING_ATTR;

void sve_fixed() {
  gnu_fixed_int32_t fi;
  gnu_fixed_float32_t_streaming fi_wrong;
  gnu_fixed_float32_t ff;
  gnu_fixed_float32_t_streaming ff_wrong;
  gnu_fixed_bool_t fb;
  gnu_fixed_bool_t_streaming fb_wrong;
  *(volatile svint32_t*)0 = fi;
  *(volatile svint32_t*)0 = fi_wrong; // expected-error {{incompatible}}
  *(volatile svfloat32_t*)0 = ff;
  *(volatile svfloat32_t*)0 = ff_wrong; // expected-error {{incompatible}}
  *(volatile svbool_t*)0 = fb;
  *(volatile svbool_t*)0 = fb_wrong; // expected-error {{incompatible}}
}

__arm_locally_streaming void streaming_fixed() {
  gnu_fixed_int32_t_streaming fi;
  gnu_fixed_float32_t fi_wrong;
  gnu_fixed_float32_t_streaming ff;
  gnu_fixed_float32_t ff_wrong;
  gnu_fixed_bool_t_streaming fb;
  gnu_fixed_bool_t fb_wrong;
  *(volatile svint32_t*)0 = fi;
  *(volatile svint32_t*)0 = fi_wrong; // expected-error {{incompatible}}
  *(volatile svfloat32_t*)0 = ff;
  *(volatile svfloat32_t*)0 = ff_wrong; // expected-error {{incompatible}}
  *(volatile svbool_t*)0 = fb;
  *(volatile svbool_t*)0 = fb_wrong; // expected-error {{incompatible}}
}

void streaming_compatible() __arm_streaming_compatible {
  gnu_fixed_int32_t fi_ns;
  gnu_fixed_float32_t_streaming fi_s;
  gnu_fixed_float32_t ff_ns;
  gnu_fixed_float32_t_streaming ff_s;
  gnu_fixed_bool_t fb_ns;
  gnu_fixed_bool_t_streaming fb_s;
  *(volatile svint32_t*)0 = fi_ns; // expected-error {{incompatible}}
  *(volatile svint32_t*)0 = fi_s; // expected-error {{incompatible}}
  *(volatile svfloat32_t*)0 = ff_ns; // expected-error {{incompatible}}
  *(volatile svfloat32_t*)0 = ff_s; // expected-error {{incompatible}}
  *(volatile svbool_t*)0 = fb_ns; // expected-error {{incompatible}}
  *(volatile svbool_t*)0 = fb_s; // expected-error {{incompatible}}
}

