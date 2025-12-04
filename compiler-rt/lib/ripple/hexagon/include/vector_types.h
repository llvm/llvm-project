#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h> // C99 and above

#ifdef __cplusplus
}
#endif

#define __decl_f16_vec_t(VEC_SIZE, VEC_BYTE_WIDTH)                             \
  typedef _Float16 f16t##VEC_SIZE __attribute__((vector_size(VEC_BYTE_WIDTH)))

__decl_f16_vec_t(64, 128);

#undef __decl_f16_vec_t

#define __decl_f32_vec_t(VEC_SIZE, VEC_BYTE_WIDTH)                             \
  typedef float f32t##VEC_SIZE __attribute__((vector_size(VEC_BYTE_WIDTH)))

__decl_f32_vec_t(32, 128);

#undef __decl_f32_vec_t

#define __decl_int_vec_t(EL_WIDTH, VEC_SIZE, VEC_BYTE_WIDTH)                   \
  typedef int##EL_WIDTH##_t i##EL_WIDTH##t##VEC_SIZE                           \
      __attribute__((vector_size(VEC_BYTE_WIDTH)))

__decl_int_vec_t(32, 32, 128);
__decl_int_vec_t(16, 64, 128);

#undef __decl_int_vec_t

#define __decl_uint_vec_t(EL_WIDTH, VEC_SIZE, VEC_BYTE_WIDTH)                  \
  typedef uint##EL_WIDTH##_t u##EL_WIDTH##t##VEC_SIZE                          \
      __attribute__((vector_size(VEC_BYTE_WIDTH)))

__decl_uint_vec_t(32, 32, 128);
__decl_uint_vec_t(16, 64, 128);

#undef __decl_uint_vec_t
