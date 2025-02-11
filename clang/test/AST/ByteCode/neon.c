// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -disable-O0-optnone -emit-llvm -o - %s -fexperimental-new-constant-interpreter

// REQUIRES: aarch64-registered-target

/// This just tests that we're not crashing with a non-primitive vector element type.

typedef __mfp8 mfloat8_t;
typedef __bf16 bfloat16_t;

typedef __attribute__((neon_vector_type(8))) mfloat8_t mfloat8x8_t;
typedef __attribute__((neon_vector_type(8))) bfloat16_t bfloat16x8_t;

typedef __UINT64_TYPE__ fpm_t;
#define __ai static __inline__ __attribute__((__always_inline__, __nodebug__))
__ai __attribute__((target("fp8,neon"))) bfloat16x8_t vcvt1_bf16_mf8_fpm(mfloat8x8_t __p0, fpm_t __p1) {
  bfloat16x8_t __ret;
  __ret = (bfloat16x8_t) __builtin_neon_vcvt1_bf16_mf8_fpm(__p0, __p1);
  return __ret;
}
