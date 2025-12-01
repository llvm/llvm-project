//===------- __ripple_vec.h: High-level interface to ripple-builtins ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

enum { __ripple_char_is_signed = (char)-1 < (char)0 };

#define ripple_id(x, y) __builtin_ripple_get_index((x), (y))
#define ripple_get_block_size(x, y) __builtin_ripple_get_size((x), (y))

#define __ripple_take_ten(a, b, c, d, e, f, g, h, i, j, ...)                   \
  a, b, c, d, e, f, g, h, i, j

typedef struct ripple_block_shape *ripple_block_t;

#define ripple_set_block_shape(PEId, Size, ...)                                \
  ((ripple_block_t)__builtin_ripple_set_shape(                                 \
      (PEId),                                                                  \
      __ripple_take_ten((Size), ##__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)))

#define ripple_set_block_size(...)                                             \
  _Pragma("GCC error \"ripple_set_block_size has been deprecated; use "        \
          "ripple_set_block_shape() instead\"")

/**
 * A descriptive loop annotation to split a loop across the "dim"-th dimension
 * work-items.
 */
#define RIPPLE_PARALLEL_STRINGIFY(x) #x
#define ripple_parallel(BlockShape, ...)                                       \
  _Pragma(RIPPLE_PARALLEL_STRINGIFY(ripple parallel Block(BlockShape)          \
                                        Dims(__VA_ARGS__) IgnoreNullStmts))

/**
 * A descriptive loop annotation to split a loop across the "dim"-th dimension
 * work-items, when it is known that the loop fully occupies all the
 * block elements it maps to.
 */
#define ripple_parallel_full(BlockShape, ...)                                  \
  _Pragma(RIPPLE_PARALLEL_STRINGIFY(ripple parallel Block(BlockShape) Dims(    \
      __VA_ARGS__) IgnoreNullStmts NoRemainder))

/**
 * Request the index of the block being processed by an enclosing ripple
 * parallel loop. The index is an integer value between 0 and the number of
 * iterations of the loop divided by the parallel block size.
 *
 * For example:
 * ```.c
 * ripple_block_t BS = ripple_set_block_shape(0, 32, 4);
 * #pragma ripple parallel Block(BS) Dims(1)
 * for (size_t i = 0; i < X; ++i)
 *   #pragma ripple parallel Block(BS) Dims(0)
 *   for (size_t j = 0; j < Y; ++j) {
 *      // Values between 0 and X / ripple_get_block_size(BS, 1)
 *      size_t block_for_i = ripple_parallel_idx(BS, 1);
 *      // Values between 0 and Y / ripple_get_block_size(BS, 0)
 *      size_t block_for_j = ripple_parallel_idx(BS, 0);
 *   }
 * ```
 */
#define ripple_parallel_idx(...) __builtin_ripple_parallel_idx(__VA_ARGS__)

// Use the definitions exposed by the targets in clang/lib/Basic/Targets/*.cpp
// to infer _Float16 and __bf16 support.
// Ripple only supports ARM/AArch64/X86[_64]/Hexagon for now
#if (defined(__hexagon__) && __HEXAGON_ARCH__ >= 68) ||                        \
    __ARM_FEATURE_FP16_SCALAR_ARITHMETIC || __arm64ec__ || __aarch64__ ||      \
    __SSE2__
#define __has_Float16__ 1
#else
#define __has_Float16__ 0
#endif

#if __STDCPP_BFLOAT16_T__ || __ARM_FEATURE_BF16 || __SSE2__ || __AVX10_2__ ||  \
    (defined(__hexagon__) && __HEXAGON_ARCH__ >= 81)
#define __has_bf16__ 1
#else
#define __has_bf16__ 0
#endif

#if defined(__hexagon__)
#define __has_soft_bf16__ 1
#else
#define __has_soft_bf16__ 0
#endif

/* ______________________ Interaction with C vector type ____________________ */

#ifdef __cplusplus
/// \brief Converts a <N_EL x T> vector to a 1-d Ripple block of T
/// for processing element BS.
/// Ripple block size assumed to be less than or equal to N_EL.
template <size_t N_EL, typename T>
[[gnu::always_inline]] T __attribute__((vector_size(sizeof(T) * N_EL)))
ripple_to_vec(ripple_block_t BS, T x) {
  __attribute__((aligned(__alignof__(
      T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];
  tmp[ripple_id(BS, 0)] = x;
  return *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp);
}

/// \brief Converts a 1-d Ripple block of T to a <N_EL x T> vector.
/// Ripple block size assumed to be less than or equal to N_EL.
template <size_t N_EL, typename T>
[[gnu::always_inline]] T
vec_to_ripple(ripple_block_t BS,
              T __attribute__((vector_size(sizeof(T) * N_EL))) x) {
  __attribute__((aligned(__alignof__(
      T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];
  *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp) = x;
  return tmp[ripple_id(BS, 0)];
}

/// \brief Converts a <N_EL x T> vector to a 2-d Ripple block of T.
/// Ripple block size assumed to be less than or equal to N_EL.
template <size_t N_EL, typename T>
[[gnu::always_inline]] T __attribute__((vector_size(sizeof(T) * N_EL)))
ripple_to_vec_2d(ripple_block_t BS, T x) {
  __attribute__((aligned(__alignof__(
      T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];
  tmp[ripple_get_block_size(BS, 0) * ripple_id(BS, 1) + ripple_id(BS, 0)] = x;
  return *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp);
}

/// \brief Converts a 2-d Ripple block of T to a <N_EL x T> vector.
/// Ripple block size assumed to be less than or equal to N_EL.
template <size_t N_EL, typename T>
[[gnu::always_inline]] T
vec_to_ripple_2d(ripple_block_t BS,
                 T __attribute__((vector_size(sizeof(T) * N_EL))) x) {
  __attribute__((aligned(__alignof__(
      T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];
  *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp) = x;
  return tmp[ripple_get_block_size(BS, 0) * ripple_id(BS, 1) +
             ripple_id(BS, 0)];
}

/// \brief Converts a <N_EL x T> vector to a 3-d Ripple block of T.
/// Ripple block size assumed to be less than or equal to N_EL.
template <size_t N_EL, typename T>
[[gnu::always_inline]] T __attribute__((vector_size(sizeof(T) * N_EL)))
ripple_to_vec_3d(ripple_block_t BS, T x) {
  __attribute__((aligned(__alignof__(
      T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];
  tmp[(ripple_get_block_size(BS, 1) * ripple_get_block_size(BS, 0)) *
          ripple_id(BS, 2) +
      ripple_get_block_size(BS, 0) * ripple_id(BS, 1) + ripple_id(BS, 0)] = x;
  return *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp);
}

/// \brief Converts a 3-d Ripple block of T to a <N_EL x T> vector.
/// Ripple block size assumed to be less than or equal to N_EL.
template <size_t N_EL, typename T>
[[gnu::always_inline]] T
vec_to_ripple_3d(ripple_block_t BS,
                 T __attribute__((vector_size(sizeof(T) * N_EL))) x) {
  __attribute__((aligned(__alignof__(
      T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];
  *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp) = x;
  return tmp[(ripple_get_block_size(BS, 1) * ripple_get_block_size(BS, 0)) *
                 ripple_id(BS, 2) +
             ripple_get_block_size(BS, 0) * ripple_id(BS, 1) +
             ripple_id(BS, 0)];
}

#else // ! defined __cplusplus

/// \brief Converts a <N_EL x T> vector to a 1-d Ripple block of T
/// for processing element BS.
/// Ripple block size assumed to be less than or equal to N_EL.
#define ripple_to_vec(N_EL, T, BS, x)                                          \
  ({                                                                           \
    __attribute__((aligned(__alignof__(                                        \
        T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];      \
    tmp[ripple_id(BS, 0)] = x;                                                 \
    *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp);                \
  })

/// \brief Converts a 1-d Ripple block of T to a <N_EL x T> vector.
/// Ripple block size assumed to be less than or equal to N_EL.
#define vec_to_ripple(N_EL, T, BS, x)                                          \
  ({                                                                           \
    __attribute__((aligned(__alignof__(                                        \
        T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];      \
    *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp) = x;            \
    tmp[ripple_id(BS, 0)];                                                     \
  })

/// \brief Converts a <N_EL x T> vector to a 2-d Ripple block of T.
/// Ripple block size assumed to be less than or equal to N_EL.
#define ripple_to_vec_2d(N_EL, T, BS, x)                                       \
  ({                                                                           \
    __attribute__((aligned(__alignof__(                                        \
        T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];      \
    tmp[ripple_get_block_size(BS, 0) * ripple_id(BS, 1) + ripple_id(BS, 0)] =  \
        x;                                                                     \
    *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp);                \
  })

/// \brief Converts a 2-d Ripple block of T to a <N_EL x T> vector.
/// Ripple block size assumed to be less than or equal to N_EL.
#define vec_to_ripple_2d(N_EL, T, BS, x)                                       \
  ({                                                                           \
    __attribute__((aligned(__alignof__(                                        \
        T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];      \
    *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp) = x;            \
    tmp[ripple_get_block_size(BS, 0) * ripple_id(BS, 1) + ripple_id(BS, 0)];   \
  })

/// \brief Converts a <N_EL x T> vector to a 3-d Ripple block of T.
/// Ripple block size assumed to be less than or equal to N_EL.
#define ripple_to_vec_3d(N_EL, T, BS, x)                                       \
  ({                                                                           \
    __attribute__((aligned(__alignof__(                                        \
        T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];      \
    tmp[(ripple_get_block_size(BS, 1) * ripple_get_block_size(BS, 0)) *        \
            ripple_id(BS, 2) +                                                 \
        ripple_get_block_size(BS, 0) * ripple_id(BS, 1) + ripple_id(BS, 0)] =  \
        x;                                                                     \
    *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp);                \
  })

/// \brief Converts a 3-d Ripple block of T to a <N_EL x T> vector.
/// Ripple block size assumed to be less than or equal to N_EL.
#define vec_to_ripple_3d(N_EL, T, BS, x)                                       \
  ({                                                                           \
    __attribute__((aligned(__alignof__(                                        \
        T __attribute__((vector_size(sizeof(T) * N_EL))))))) T tmp[N_EL];      \
    *((T __attribute__((vector_size(sizeof(T) * N_EL))) *)tmp) = x;            \
    tmp[(ripple_get_block_size(BS, 1) * ripple_get_block_size(BS, 0)) *        \
            ripple_id(BS, 2) +                                                 \
        ripple_get_block_size(BS, 0) * ripple_id(BS, 1) + ripple_id(BS, 0)];   \
  })

#endif // __cplusplus

/* _______________________________ Reductions ________________________________*/

#define RIPPLE_DISABLE_GENERIC_WARNING                                         \
  _Pragma("GCC diagnostic push")                                               \
      _Pragma("GCC diagnostic ignored \"-Wpedantic\"")
#define RIPPLE_REENABLE_GENERIC_WARNING _Pragma("GCC diagnostic pop")

#ifndef __cplusplus

// TODO: __bf16 is only supported as non-native on Hexagon.
// TODO: Hence we use f32 emulation.
// TODO: Update as the backend evolves to support native __bf16.

#if __has_bf16__
#if __has_soft_bf16__
#define __extra_bf16_ripple_reduceadd(bitmask, val)                            \
  , __bf16 : (__bf16)__builtin_ripple_reduceadd_f32((bitmask), (float)(val))
#define __extra_bf16_ripple_reducemul(bitmask, val)                            \
  , __bf16 : (__bf16)__builtin_ripple_reducemul_f32((bitmask), (float)(val))
#define __extra_bf16_ripple_reducemin(bitmask, val)                            \
  , __bf16 : (__bf16)__builtin_ripple_reducemin_f32((bitmask), (float)(val))
#define __extra_bf16_ripple_reducemax(bitmask, val)                            \
  , __bf16 : (__bf16)__builtin_ripple_reducemax_f32((bitmask), (float)(val))
#define __extra_bf16_ripple_reduceminimum(bitmask, val)                        \
  , __bf16 : (__bf16)__builtin_ripple_reduceminimum_f32((bitmask), (float)(val))
#define __extra_bf16_ripple_reducemaximum(bitmask, val)                        \
  , __bf16 : (__bf16)__builtin_ripple_reducemaximum_f32((bitmask), (float)(val))
#else // !__has_soft_bf16__
#define __extra_bf16_ripple_reduceadd(bitmask, val)                            \
  , __bf16 : __builtin_ripple_reduceadd_bf16((bitmask), (val))
#define __extra_bf16_ripple_reducemul(bitmask, val)                            \
  , __bf16 : __builtin_ripple_reducemul_bf16((bitmask), (val))
#define __extra_bf16_ripple_reducemin(bitmask, val)                            \
  , __bf16 : __builtin_ripple_reducemin_bf16((bitmask), (val))
#define __extra_bf16_ripple_reducemax(bitmask, val)                            \
  , __bf16 : __builtin_ripple_reducemax_bf16((bitmask), (val))
#define __extra_bf16_ripple_reduceminimum(bitmask, val)                        \
  , __bf16 : __builtin_ripple_reduceminimum_bf16((bitmask), (val))
#define __extra_bf16_ripple_reducemaximum(bitmask, val)                        \
  , __bf16 : __builtin_ripple_reducemaximum_bf16((bitmask), (val))
#endif // __has_soft_bf16__
#else
#define __extra_bf16_ripple_reduceadd(bitmask, val)
#define __extra_bf16_ripple_reducemul(bitmask, val)
#define __extra_bf16_ripple_reducemin(bitmask, val)
#define __extra_bf16_ripple_reducemax(bitmask, val)
#define __extra_bf16_ripple_reduceminimum(bitmask, val)
#define __extra_bf16_ripple_reducemaximum(bitmask, val)
#endif

#if __has_Float16__
#define __extra_f16_ripple_reduceadd(bitmask, val)                             \
  , _Float16 : __builtin_ripple_reduceadd_f16((bitmask), (val))
#define __extra_f16_ripple_reducemul(bitmask, val)                             \
  , _Float16 : __builtin_ripple_reducemul_f16((bitmask), (val))
#define __extra_f16_ripple_reducemin(bitmask, val)                             \
  , _Float16 : __builtin_ripple_reducemin_f16((bitmask), (val))
#define __extra_f16_ripple_reducemax(bitmask, val)                             \
  , _Float16 : __builtin_ripple_reducemax_f16((bitmask), (val))
#define __extra_f16_ripple_reduceminimum(bitmask, val)                         \
  , _Float16 : __builtin_ripple_reduceminimum_f16((bitmask), (val))
#define __extra_f16_ripple_reducemaximum(bitmask, val)                         \
  , _Float16 : __builtin_ripple_reducemaximum_f16((bitmask), (val))
#else
#define __extra_f16_ripple_reduceadd(bitmask, val)
#define __extra_f16_ripple_reducemul(bitmask, val)
#define __extra_f16_ripple_reducemin(bitmask, val)
#define __extra_f16_ripple_reducemax(bitmask, val)
#define __extra_f16_ripple_reduceminimum(bitmask, val)
#define __extra_f16_ripple_reducemaximum(bitmask, val)
#endif

#define __ripple_reduce_any_int(X, Type, T, bitmask, val)                      \
  ((Type)(sizeof(Type) == 1                                                    \
              ? __builtin_ripple_reduce##X##_##T##8(bitmask, val)              \
              : (sizeof(Type) == 2                                             \
                     ? __builtin_ripple_reduce##X##_##T##16(bitmask, val)      \
                     : (sizeof(Type) == 4                                      \
                            ? __builtin_ripple_reduce##X##_##T##32(bitmask,    \
                                                                   val)        \
                            : __builtin_ripple_reduce##X##_##T##64(bitmask,    \
                                                                   val)))))

#define ripple_reduceadd(bitmask, val)                                         \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_reduce_any_int(add, char, i, (bitmask), (val))     \
                 : __ripple_reduce_any_int(add, char, u, (bitmask), (val))),   \
      signed char: __ripple_reduce_any_int(add, signed char, i, (bitmask),     \
                                           (val)),                             \
      unsigned char: __ripple_reduce_any_int(add, unsigned char, u, (bitmask), \
                                             (val)),                           \
      signed short: __ripple_reduce_any_int(add, signed short, i, (bitmask),   \
                                            (val)),                            \
      unsigned short: __ripple_reduce_any_int(add, unsigned short, u,          \
                                              (bitmask), (val)),               \
      signed int: __ripple_reduce_any_int(add, signed int, i, (bitmask),       \
                                          (val)),                              \
      unsigned int: __ripple_reduce_any_int(add, unsigned int, u, (bitmask),   \
                                            (val)),                            \
      signed long: __ripple_reduce_any_int(add, signed long, i, (bitmask),     \
                                           (val)),                             \
      unsigned long: __ripple_reduce_any_int(add, unsigned long, u, (bitmask), \
                                             (val)),                           \
      signed long long: __ripple_reduce_any_int(add, signed long long, i,      \
                                                (bitmask), (val)),             \
      unsigned long long: __ripple_reduce_any_int(add, unsigned long long, u,  \
                                                  (bitmask), (val)),           \
      float: __builtin_ripple_reduceadd_f32((bitmask), (val)),                 \
      double: __builtin_ripple_reduceadd_f64((bitmask), (val))                 \
          __extra_bf16_ripple_reduceadd((bitmask), (val))                      \
              __extra_f16_ripple_reduceadd((bitmask), (val)))                  \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_reducemul(bitmask, val)                                         \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_reduce_any_int(mul, char, i, (bitmask), (val))     \
                 : __ripple_reduce_any_int(mul, char, u, (bitmask), (val))),   \
      signed char: __ripple_reduce_any_int(mul, signed char, i, (bitmask),     \
                                           (val)),                             \
      unsigned char: __ripple_reduce_any_int(mul, unsigned char, u, (bitmask), \
                                             (val)),                           \
      signed short: __ripple_reduce_any_int(mul, signed short, i, (bitmask),   \
                                            (val)),                            \
      unsigned short: __ripple_reduce_any_int(mul, unsigned short, u,          \
                                              (bitmask), (val)),               \
      signed int: __ripple_reduce_any_int(mul, signed int, i, (bitmask),       \
                                          (val)),                              \
      unsigned int: __ripple_reduce_any_int(mul, unsigned int, u, (bitmask),   \
                                            (val)),                            \
      signed long: __ripple_reduce_any_int(mul, signed long, i, (bitmask),     \
                                           (val)),                             \
      unsigned long: __ripple_reduce_any_int(mul, unsigned long, u, (bitmask), \
                                             (val)),                           \
      signed long long: __ripple_reduce_any_int(mul, signed long long, i,      \
                                                (bitmask), (val)),             \
      unsigned long long: __ripple_reduce_any_int(mul, unsigned long long, u,  \
                                                  (bitmask), (val)),           \
      float: __builtin_ripple_reducemul_f32((bitmask), (val)),                 \
      double: __builtin_ripple_reducemul_f64((bitmask), (val))                 \
          __extra_bf16_ripple_reducemul((bitmask), (val))                      \
              __extra_f16_ripple_reducemul((bitmask), (val)))                  \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_reducemin(bitmask, val)                                         \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_reduce_any_int(min, char, i, (bitmask), (val))     \
                 : __ripple_reduce_any_int(min, char, u, (bitmask), (val))),   \
      signed char: __ripple_reduce_any_int(min, signed char, i, (bitmask),     \
                                           (val)),                             \
      unsigned char: __ripple_reduce_any_int(min, unsigned char, u, (bitmask), \
                                             (val)),                           \
      signed short: __ripple_reduce_any_int(min, signed short, i, (bitmask),   \
                                            (val)),                            \
      unsigned short: __ripple_reduce_any_int(min, unsigned short, u,          \
                                              (bitmask), (val)),               \
      signed int: __ripple_reduce_any_int(min, signed int, i, (bitmask),       \
                                          (val)),                              \
      unsigned int: __ripple_reduce_any_int(min, unsigned int, u, (bitmask),   \
                                            (val)),                            \
      signed long: __ripple_reduce_any_int(min, signed long, i, (bitmask),     \
                                           (val)),                             \
      unsigned long: __ripple_reduce_any_int(min, unsigned long, u, (bitmask), \
                                             (val)),                           \
      signed long long: __ripple_reduce_any_int(min, signed long long, i,      \
                                                (bitmask), (val)),             \
      unsigned long long: __ripple_reduce_any_int(min, unsigned long long, u,  \
                                                  (bitmask), (val)),           \
      float: __builtin_ripple_reducemin_f32((bitmask), (val)),                 \
      double: __builtin_ripple_reducemin_f64((bitmask), (val))                 \
          __extra_bf16_ripple_reducemin((bitmask), (val))                      \
              __extra_f16_ripple_reducemin(bitmask, (val)))                    \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_reduceminimum(bitmask, val)                                     \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      float: __builtin_ripple_reduceminimum_f32((bitmask), (val)),             \
      double: __builtin_ripple_reduceminimum_f64((bitmask), (val))             \
          __extra_bf16_ripple_reduceminimum((bitmask), (val))                  \
              __extra_f16_ripple_reduceminimum(bitmask, (val)))                \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_reducemax(bitmask, val)                                         \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_reduce_any_int(max, char, i, (bitmask), (val))     \
                 : __ripple_reduce_any_int(max, char, u, (bitmask), (val))),   \
      signed char: __ripple_reduce_any_int(max, signed char, i, (bitmask),     \
                                           (val)),                             \
      unsigned char: __ripple_reduce_any_int(max, unsigned char, u, (bitmask), \
                                             (val)),                           \
      signed short: __ripple_reduce_any_int(max, signed short, i, (bitmask),   \
                                            (val)),                            \
      unsigned short: __ripple_reduce_any_int(max, unsigned short, u,          \
                                              (bitmask), (val)),               \
      signed int: __ripple_reduce_any_int(max, signed int, i, (bitmask),       \
                                          (val)),                              \
      unsigned int: __ripple_reduce_any_int(max, unsigned int, u, (bitmask),   \
                                            (val)),                            \
      signed long: __ripple_reduce_any_int(max, signed long, i, (bitmask),     \
                                           (val)),                             \
      unsigned long: __ripple_reduce_any_int(max, unsigned long, u, (bitmask), \
                                             (val)),                           \
      signed long long: __ripple_reduce_any_int(max, signed long long, i,      \
                                                (bitmask), (val)),             \
      unsigned long long: __ripple_reduce_any_int(max, unsigned long long, u,  \
                                                  (bitmask), (val)),           \
      float: __builtin_ripple_reducemax_f32((bitmask), (val)),                 \
      double: __builtin_ripple_reducemax_f64((bitmask), (val))                 \
          __extra_bf16_ripple_reducemax((bitmask), (val))                      \
              __extra_f16_ripple_reducemax((bitmask), (val)))                  \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_reducemaximum(bitmask, val)                                     \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      float: __builtin_ripple_reducemaximum_f32((bitmask), (val)),             \
      double: __builtin_ripple_reducemaximum_f64((bitmask), (val))             \
          __extra_bf16_ripple_reducemaximum((bitmask), (val))                  \
              __extra_f16_ripple_reducemaximum(bitmask, (val)))                \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_reduceand(bitmask, val)                                         \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_reduce_any_int(and, char, i, (bitmask), (val))     \
                 : __ripple_reduce_any_int(and, char, u, (bitmask), (val))),   \
      signed char: __ripple_reduce_any_int(and, signed char, i, (bitmask),     \
                                           (val)),                             \
      unsigned char: __ripple_reduce_any_int(and, unsigned char, u, (bitmask), \
                                             (val)),                           \
      signed short: __ripple_reduce_any_int(and, signed short, i, (bitmask),   \
                                            (val)),                            \
      unsigned short: __ripple_reduce_any_int(and, unsigned short, u,          \
                                              (bitmask), (val)),               \
      signed int: __ripple_reduce_any_int(and, signed int, i, (bitmask),       \
                                          (val)),                              \
      unsigned int: __ripple_reduce_any_int(and, unsigned int, u, (bitmask),   \
                                            (val)),                            \
      signed long: __ripple_reduce_any_int(and, signed long, i, (bitmask),     \
                                           (val)),                             \
      unsigned long: __ripple_reduce_any_int(and, unsigned long, u, (bitmask), \
                                             (val)),                           \
      signed long long: __ripple_reduce_any_int(and, signed long long, i,      \
                                                (bitmask), (val)),             \
      unsigned long long: __ripple_reduce_any_int(and, unsigned long long, u,  \
                                                  (bitmask), (val)))           \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_reduceor(bitmask, val)                                          \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_reduce_any_int(or, char, i, (bitmask), (val))      \
                 : __ripple_reduce_any_int(or, char, u, (bitmask), (val))),    \
      signed char: __ripple_reduce_any_int(or, signed char, i, (bitmask),      \
                                           (val)),                             \
      unsigned char: __ripple_reduce_any_int(or, unsigned char, u, (bitmask),  \
                                             (val)),                           \
      signed short: __ripple_reduce_any_int(or, signed short, i, (bitmask),    \
                                            (val)),                            \
      unsigned short: __ripple_reduce_any_int(or, unsigned short, u,           \
                                              (bitmask), (val)),               \
      signed int: __ripple_reduce_any_int(or, signed int, i, (bitmask),        \
                                          (val)),                              \
      unsigned int: __ripple_reduce_any_int(or, unsigned int, u, (bitmask),    \
                                            (val)),                            \
      signed long: __ripple_reduce_any_int(or, signed long, i, (bitmask),      \
                                           (val)),                             \
      unsigned long: __ripple_reduce_any_int(or, unsigned long, u, (bitmask),  \
                                             (val)),                           \
      signed long long: __ripple_reduce_any_int(or, signed long long, i,       \
                                                (bitmask), (val)),             \
      unsigned long long: __ripple_reduce_any_int(or, unsigned long long, u,   \
                                                  (bitmask), (val)))           \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_reducexor(bitmask, val)                                         \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_reduce_any_int(xor, char, i, (bitmask), (val))     \
                 : __ripple_reduce_any_int(xor, char, u, (bitmask), (val))),   \
      signed char: __ripple_reduce_any_int(xor, signed char, i, (bitmask),     \
                                           (val)),                             \
      unsigned char: __ripple_reduce_any_int(xor, unsigned char, u, (bitmask), \
                                             (val)),                           \
      signed short: __ripple_reduce_any_int(xor, signed short, i, (bitmask),   \
                                            (val)),                            \
      unsigned short: __ripple_reduce_any_int(xor, unsigned short, u,          \
                                              (bitmask), (val)),               \
      signed int: __ripple_reduce_any_int(xor, signed int, i, (bitmask),       \
                                          (val)),                              \
      unsigned int: __ripple_reduce_any_int(xor, unsigned int, u, (bitmask),   \
                                            (val)),                            \
      signed long: __ripple_reduce_any_int(xor, signed long, i, (bitmask),     \
                                           (val)),                             \
      unsigned long: __ripple_reduce_any_int(xor, unsigned long, u, (bitmask), \
                                             (val)),                           \
      signed long long: __ripple_reduce_any_int(xor, signed long long, i,      \
                                                (bitmask), (val)),             \
      unsigned long long: __ripple_reduce_any_int(xor, unsigned long long, u,  \
                                                  (bitmask), (val)))           \
      RIPPLE_REENABLE_GENERIC_WARNING

#else // defined(_cplusplus)

// We don't have <type_traits> in llvm, do a simple implementation for Ripple!
namespace __RippleTyTraits {
// is_same to check types equality

template <typename T, typename U> struct is_same {
  static const bool value = false;
};
template <typename T> struct is_same<T, T> {
  static const bool value = true;
};

// remove_cv to remove const and volatile
template <typename T> struct remove_cv {
  typedef T type;
};
template <typename T> struct remove_cv<const T> {
  typedef T type;
};
template <typename T> struct remove_cv<volatile T> {
  typedef T type;
};
template <typename T> struct remove_cv<const volatile T> {
  typedef T type;
};

// remove_reference
template <typename T> struct remove_reference {
  typedef T type;
};
template <typename T> struct remove_reference<T &> {
  typedef T type;
};
template <typename T> struct remove_reference<T &&> {
  typedef T type;
};
}; // namespace __RippleTyTraits

#define spec_reduce(op, CT, T)                                                 \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static CT ripple_reduce##op(const CT Val) {   \
    return __builtin_ripple_reduce##op##_##T(msk, (Val));                      \
  }

#define spec_reduce_int(op, CT, UI)                                            \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static CT ripple_reduce##op(const CT Val) {   \
    switch (sizeof(Val)) {                                                     \
    case 1:                                                                    \
      return __builtin_ripple_reduce##op##_##UI##8(msk, (Val));                \
    case 2:                                                                    \
      return __builtin_ripple_reduce##op##_##UI##16(msk, (Val));               \
    case 4:                                                                    \
      return __builtin_ripple_reduce##op##_##UI##32(msk, (Val));               \
    case 8:                                                                    \
      return __builtin_ripple_reduce##op##_##UI##64(msk, (Val));               \
    }                                                                          \
  }

#define spec_reduce_char(op)                                                   \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static char ripple_reduce##op(                \
      const char Val) {                                                        \
    return __ripple_char_is_signed                                             \
               ? __builtin_ripple_reduce##op##_i8(msk, Val)                    \
               : __builtin_ripple_reduce##op##_u8(msk, Val);                   \
  }

#if __has_bf16__
#if __has_soft_bf16__
#define emulate_spec_reduce_bf16(op)                                           \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static __bf16 ripple_reduce##op(              \
      const __bf16 Val) {                                                      \
    return (__bf16)__builtin_ripple_reduce##op##_f32(msk, (float)(Val));       \
  }
#else // !__has_soft_bf16__
#define emulate_spec_reduce_bf16(op)                                           \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static __bf16 ripple_reduce##op(              \
      const __bf16 Val) {                                                      \
    return __builtin_ripple_reduce##op##_bf16(msk, (Val));                     \
  }
#endif // __has_soft_bf16__
#endif // __has_bf16__

#if __has_bf16__ && __has_Float16__
#define spec_reduce_ftypes(op)                                                 \
  spec_reduce(op, _Float16, f16);                                              \
  emulate_spec_reduce_bf16(op);                                                \
  spec_reduce(op, float, f32);                                                 \
  spec_reduce(op, double, f64);
#elif __has_bf16__
#define spec_reduce_ftypes(op)                                                 \
  emulate_spec_reduce_bf16(op);                                                \
  spec_reduce(op, float, f32);                                                 \
  spec_reduce(op, double, f64);
#elif __has_Float16__
#define spec_reduce_ftypes(op)                                                 \
  spec_reduce(op, _Float16, f16);                                              \
  spec_reduce(op, float, f32);                                                 \
  spec_reduce(op, double, f64)
#else // __has_Float16__
#define spec_reduce_ftypes(op)                                                 \
  spec_reduce(op, float, f32);                                                 \
  spec_reduce(op, double, f64)
#endif

#define spec_reduce_itypes(op)                                                 \
  spec_reduce_char(op);                                                        \
  spec_reduce_int(op, signed char, i);                                         \
  spec_reduce_int(op, unsigned char, u);                                       \
  spec_reduce_int(op, short, i);                                               \
  spec_reduce_int(op, unsigned short, u);                                      \
  spec_reduce_int(op, int, i);                                                 \
  spec_reduce_int(op, unsigned int, u);                                        \
  spec_reduce_int(op, long, i);                                                \
  spec_reduce_int(op, unsigned long, u);                                       \
  spec_reduce_int(op, long long, i);                                           \
  spec_reduce_int(op, unsigned long long, u);

spec_reduce_itypes(add);
spec_reduce_ftypes(add);
spec_reduce_itypes(mul);
spec_reduce_ftypes(mul);
spec_reduce_itypes(max);
spec_reduce_ftypes(max);
spec_reduce_ftypes(maximum);
spec_reduce_itypes(min);
spec_reduce_ftypes(min);
spec_reduce_ftypes(minimum);

spec_reduce_itypes(and);
spec_reduce_itypes(or);
spec_reduce_itypes(xor);

// Expose the same API as C for all reductions
#define ripple_reduceadd(mask, val) ripple_reduceadd<mask>(val)
#define ripple_reducemul(mask, val) ripple_reducemul<mask>(val)
#define ripple_reducemax(mask, val) ripple_reducemax<mask>(val)
#define ripple_reducemin(mask, val) ripple_reducemin<mask>(val)
#define ripple_reducemaximum(mask, val) ripple_reducemaximum<mask>(val)
#define ripple_reduceminimum(mask, val) ripple_reduceminimum<mask>(val)
#define ripple_reduceand(mask, val) ripple_reduceand<mask>(val)
#define ripple_reduceor(mask, val) ripple_reduceor<mask>(val)
#define ripple_reducexor(mask, val) ripple_reducexor<mask>(val)

#undef spec_reduce
#undef spec_reduce_char
#undef spec_reduce_int
#undef spec_reduce_itypes
#undef spec_reduce_ftypes

#endif // defined(cplusplus)

/* ___________________________ Shuffle Ops ___________________________________*/

#define __ripple_shuffle_impl_int(Type, UI, Val1, Val2, IsPair, IdxFun)        \
  ((Type)(sizeof(Type) == 1                                                    \
              ? __builtin_ripple_shuffle_##UI##8((Val1), (Val2), (IsPair),     \
                                                 (IdxFun))                     \
              : (sizeof(Type) == 2                                             \
                     ? __builtin_ripple_shuffle_##UI##16((Val1), (Val2),       \
                                                         (IsPair), (IdxFun))   \
                     : (sizeof(Type) == 4                                      \
                            ? __builtin_ripple_shuffle_##UI##32(               \
                                  (Val1), (Val2), (IsPair), (IdxFun))          \
                            : __builtin_ripple_shuffle_##UI##64(               \
                                  (Val1), (Val2), (IsPair), (IdxFun))))))

#ifndef __cplusplus

#if __has_bf16__
#if __has_soft_bf16__
#define __extra_bf16_ripple_shuffle(Val1, Val2, IsPair, indexfunc)             \
  , __bf16 : (__bf16)__builtin_ripple_shuffle_f32(                             \
                 (float)(Val1), (float)(Val2), (IsPair), (indexfunc))
#else // __has_soft_bf16__
#define __extra_bf16_ripple_shuffle(Val1, Val2, IsPair, indexfunc)             \
  , __bf16                                                                     \
      : __builtin_ripple_shuffle_bf16((Val1), (Val2), (IsPair), (indexfunc))
#endif // __has_soft_bf16__
#else
#define __extra_bf16_ripple_shuffle(Val1, Val2, IsPair, indexfunc)
#endif

#if __has_Float16__
#define __extra_f16_ripple_shuffle(Val1, Val2, IsPair, indexfunc)              \
  , _Float16                                                                   \
      : __builtin_ripple_shuffle_f16((Val1), (Val2), (IsPair), (indexfunc))
#else
#define __extra_f16_ripple_shuffle(Val, Val2, IsPair, indexfunc)
#endif

#define __ripple_shuffle_impl(Val1, Val2, IsPair, IdxFun)                      \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((Val1),                                                             \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_shuffle_impl_int(char, i, (Val1), (Val2),          \
                                             (IsPair), (IdxFun))               \
                 : __ripple_shuffle_impl_int(char, u, (Val1), (Val2),          \
                                             (IsPair), (IdxFun))),             \
      signed char: __ripple_shuffle_impl_int(signed char, i, (Val1), (Val2),   \
                                             (IsPair), (IdxFun)),              \
      unsigned char: __ripple_shuffle_impl_int(unsigned char, u, (Val1),       \
                                               (Val2), (IsPair), (IdxFun)),    \
      signed short: __ripple_shuffle_impl_int(signed short, i, (Val1), (Val2), \
                                              (IsPair), (IdxFun)),             \
      unsigned short: __ripple_shuffle_impl_int(unsigned short, u, (Val1),     \
                                                (Val2), (IsPair), (IdxFun)),   \
      signed int: __ripple_shuffle_impl_int(signed int, i, (Val1), (Val2),     \
                                            (IsPair), (IdxFun)),               \
      unsigned int: __ripple_shuffle_impl_int(unsigned int, u, (Val1), (Val2), \
                                              (IsPair), (IdxFun)),             \
      signed long: __ripple_shuffle_impl_int(signed long, i, (Val1), (Val2),   \
                                             (IsPair), (IdxFun)),              \
      unsigned long: __ripple_shuffle_impl_int(unsigned long, u, (Val1),       \
                                               (Val2), (IsPair), (IdxFun)),    \
      signed long long: __ripple_shuffle_impl_int(signed long long, i, (Val1), \
                                                  (Val2), (IsPair), (IdxFun)), \
      unsigned long long: __ripple_shuffle_impl_int(                           \
               unsigned long long, u, (Val1), (Val2), (IsPair), (IdxFun)),     \
      float: __builtin_ripple_shuffle_f32((Val1), (Val2), (IsPair), (IdxFun)), \
      double: __builtin_ripple_shuffle_f64((Val1), (Val2), (IsPair), (IdxFun)) \
          __extra_bf16_ripple_shuffle((Val1), (Val2), (IsPair), (IdxFun))      \
              __extra_f16_ripple_shuffle((Val1), (Val2), (IsPair), (IdxFun)))  \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_shuffle(Val, IdxFun)                                            \
  __ripple_shuffle_impl((Val), (Val), 0, (IdxFun))

#define ripple_shuffle_pair(Val1, Val2, IdxFun)                                \
  __ripple_shuffle_impl((Val1), (Val2), 1, (IdxFun))

#define ripple_shuffle_p(Val, IdxFun)                                          \
  (__typeof__(Val))__builtin_ripple_shuffle_p((Val), (Val), 0, (IdxFun))

#define ripple_shuffle_pair_p(Val1, Val2, IdxFun)                              \
  (__typeof__(Val1))__builtin_ripple_shuffle_p((Val1), (Val2), 1, (IdxFun))

#else // defined(__cplusplus)

#if __has_soft_bf16__

#define __emulate_shuffle_bf16(Val1, Val2, IsPair, indexfunc)                  \
  (__bf16)__builtin_ripple_shuffle_f32((float)(Val1), (float)(Val2), (IsPair), \
                                       (indexfunc))

#else // ! __has_soft_bf16__

#define __emulate_shuffle_bf16(Val1, Val2, IsPair, indexfunc)                  \
  __builtin_ripple_shuffle_bf16((Val1), (Val2), (IsPair), (indexfunc))

#endif // __has_soft_bf16__

#define __ripple_shuffle_impl_int_cpp(Type, UI)                                \
  static __attribute__((always_inline)) Type ripple_shuffle(                   \
      Type x, size_t (*index_func)(size_t, size_t)) {                          \
    return __ripple_shuffle_impl_int(Type, UI, x, x, false, index_func);       \
  }

static __attribute__((always_inline)) char
ripple_shuffle(char x, size_t (*index_func)(size_t, size_t)) {
  return __ripple_char_is_signed
             ? __ripple_shuffle_impl_int(char, i, x, x, false, index_func)
             : __ripple_shuffle_impl_int(char, u, x, x, false, index_func);
}

template <typename T>
static __attribute__((always_inline)) T *
ripple_shuffle_p(T *x, size_t (*index_func)(size_t, size_t)) {
  return static_cast<T *>(__builtin_ripple_shuffle_p(x, x, false, index_func));
}

__ripple_shuffle_impl_int_cpp(signed char, i);
__ripple_shuffle_impl_int_cpp(unsigned char, u);
__ripple_shuffle_impl_int_cpp(signed short, i);
__ripple_shuffle_impl_int_cpp(unsigned short, u);
__ripple_shuffle_impl_int_cpp(signed int, i);
__ripple_shuffle_impl_int_cpp(unsigned int, u);
__ripple_shuffle_impl_int_cpp(signed long, i);
__ripple_shuffle_impl_int_cpp(unsigned long, u);
__ripple_shuffle_impl_int_cpp(signed long long, i);
__ripple_shuffle_impl_int_cpp(unsigned long long, u);

#undef __ripple_shuffle_impl_int_cpp

#if __has_Float16__
static __attribute__((always_inline)) _Float16
ripple_shuffle(_Float16 x, size_t (*index_func)(size_t, size_t)) {
  return __builtin_ripple_shuffle_f16(x, x, false, index_func);
}
#endif
#if __has_bf16__
static __attribute__((always_inline)) __bf16
ripple_shuffle(__bf16 x, size_t (*index_func)(size_t, size_t)) {
  return __emulate_shuffle_bf16(x, x, false, index_func);
}
#endif
static __attribute__((always_inline)) float
ripple_shuffle(float x, size_t (*index_func)(size_t, size_t)) {
  return __builtin_ripple_shuffle_f32(x, x, false, index_func);
}
static __attribute__((always_inline)) double
ripple_shuffle(double x, size_t (*index_func)(size_t, size_t)) {
  return __builtin_ripple_shuffle_f64(x, x, false, index_func);
}

#define __ripple_shuffle_pair_impl_int_cpp(Type, UI)                           \
  static __attribute__((always_inline)) Type ripple_shuffle_pair(              \
      Type x, Type y, size_t (*index_func)(size_t, size_t)) {                  \
    return __ripple_shuffle_impl_int(Type, UI, x, y, true, index_func);        \
  }

static __attribute__((always_inline)) char
ripple_shuffle_pair(char x, char y, size_t (*index_func)(size_t, size_t)) {
  return __ripple_char_is_signed
             ? __ripple_shuffle_impl_int(char, i, x, y, true, index_func)
             : __ripple_shuffle_impl_int(char, u, x, y, true, index_func);
}

template <typename T>
static __attribute__((always_inline)) T *
ripple_shuffle_pair_p(T *x, T *y, size_t (*index_func)(size_t, size_t)) {
  return static_cast<T *>(__builtin_ripple_shuffle_p(x, y, true, index_func));
}

__ripple_shuffle_pair_impl_int_cpp(signed char, i);
__ripple_shuffle_pair_impl_int_cpp(unsigned char, u);
__ripple_shuffle_pair_impl_int_cpp(signed short, i);
__ripple_shuffle_pair_impl_int_cpp(unsigned short, u);
__ripple_shuffle_pair_impl_int_cpp(signed int, i);
__ripple_shuffle_pair_impl_int_cpp(unsigned int, u);
__ripple_shuffle_pair_impl_int_cpp(signed long, i);
__ripple_shuffle_pair_impl_int_cpp(unsigned long, u);
__ripple_shuffle_pair_impl_int_cpp(signed long long, i);
__ripple_shuffle_pair_impl_int_cpp(unsigned long long, u);

#undef __ripple_shuffle_pair_impl_int_cpp

#if __has_Float16__
static __attribute__((always_inline)) _Float16
ripple_shuffle_pair(_Float16 x, _Float16 y,
                    size_t (*index_func)(size_t, size_t)) {
  return __builtin_ripple_shuffle_f16(x, y, true, index_func);
}
#endif
#if __has_bf16__
static __attribute__((always_inline)) __bf16
ripple_shuffle_pair(__bf16 x, __bf16 y, size_t (*index_func)(size_t, size_t)) {
  return __emulate_shuffle_bf16(x, y, true, index_func);
}
#endif
static __attribute__((always_inline)) float
ripple_shuffle_pair(float x, float y, size_t (*index_func)(size_t, size_t)) {
  return __builtin_ripple_shuffle_f32(x, y, true, index_func);
}
static __attribute__((always_inline)) double
ripple_shuffle_pair(double x, double y, size_t (*index_func)(size_t, size_t)) {
  return __builtin_ripple_shuffle_f64(x, y, true, index_func);
}

#endif // __cplusplus

/* _________________________ Saturation Ops _________________________________*/

#ifndef __cplusplus

#define __ripple_TYPEID(X)                                                     \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((X),                                                                \
      char: 0,                                                                 \
      signed char: 1,                                                          \
      unsigned char: 2,                                                        \
      short: 3,                                                                \
      unsigned short: 4,                                                       \
      int: 5,                                                                  \
      unsigned int: 6,                                                         \
      long: 7,                                                                 \
      unsigned long: 8,                                                        \
      long long: 9,                                                            \
      unsigned long long: 10) RIPPLE_REENABLE_GENERIC_WARNING

#define __ripple_sat_any(Type, UI, Op, X, Y)                                   \
  ((Type)(sizeof(Type) == 1                                                    \
              ? __builtin_ripple_##Op##_sat_##UI##8(X, Y)                      \
              : (sizeof(Type) == 2                                             \
                     ? __builtin_ripple_##Op##_sat_##UI##16(X, Y)              \
                     : (sizeof(Type) == 4                                      \
                            ? __builtin_ripple_##Op##_sat_##UI##32(X, Y)       \
                            : __builtin_ripple_##Op##_sat_##UI##64(X, Y)))))

#define __ripple_sat_any_generic(Op, X, Y)                                     \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((X),                                                                \
      char: (__ripple_char_is_signed ? __ripple_sat_any(char, i, Op, X, Y)     \
                                     : __ripple_sat_any(char, u, Op, X, Y)),   \
      signed char: __ripple_sat_any(signed char, i, Op, X, Y),                 \
      unsigned char: __ripple_sat_any(unsigned char, u, Op, X, Y),             \
      signed short: __ripple_sat_any(signed short, i, Op, X, Y),               \
      unsigned short: __ripple_sat_any(unsigned short, u, Op, X, Y),           \
      signed int: __ripple_sat_any(signed int, i, Op, X, Y),                   \
      unsigned int: __ripple_sat_any(unsigned int, u, Op, X, Y),               \
      signed long: __ripple_sat_any(signed long, i, Op, X, Y),                 \
      unsigned long: __ripple_sat_any(unsigned long, u, Op, X, Y),             \
      signed long long: __ripple_sat_any(signed long long, i, Op, X, Y),       \
      unsigned long long: __ripple_sat_any(unsigned long long, u, Op, X, Y))   \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_add_sat(X, Y)                                                   \
  RIPPLE_DISABLE_GENERIC_WARNING({                                             \
    _Static_assert(__ripple_TYPEID(X) == __ripple_TYPEID(Y),                   \
                   "Type mismatch in add_sat.");                               \
    __ripple_sat_any_generic(add, X, Y);                                       \
  })                                                                           \
  RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_sub_sat(X, Y)                                                   \
  RIPPLE_DISABLE_GENERIC_WARNING({                                             \
    _Static_assert(__ripple_TYPEID(X) == __ripple_TYPEID(Y),                   \
                   "Type mismatch in sub_sat.");                               \
    __ripple_sat_any_generic(sub, X, Y);                                       \
  })                                                                           \
  RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_shl_sat(X, Y)                                                   \
  RIPPLE_DISABLE_GENERIC_WARNING({                                             \
    _Static_assert(__ripple_TYPEID(X) == __ripple_TYPEID(Y),                   \
                   "Type mismatch in shl_sat.");                               \
    __ripple_sat_any_generic(shl, X, Y);                                       \
  })                                                                           \
  RIPPLE_REENABLE_GENERIC_WARNING

#endif // !__cplusplus

/* _______________________________ Broadcasts ________________________________*/

#define __ripple_broadcast_any_int(Type, UI, BS, Mask, Val)                    \
  ((Type)(sizeof(Type) == 1                                                    \
              ? __builtin_ripple_broadcast_##UI##8(BS, Mask, Val)              \
              : (sizeof(Type) == 2                                             \
                     ? __builtin_ripple_broadcast_##UI##16(BS, Mask, Val)      \
                     : (sizeof(Type) == 4                                      \
                            ? __builtin_ripple_broadcast_##UI##32(BS, Mask,    \
                                                                  Val)         \
                            : __builtin_ripple_broadcast_##UI##64(BS, Mask,    \
                                                                  Val)))))

#ifndef __cplusplus

#if __has_bf16__
#if __has_soft_bf16__
#define __extra_bf16_ripple_broadcast(PE, Mask, Val)                           \
  , __bf16 : (__bf16)__builtin_ripple_broadcast_f32((PE), (Mask), (float)(Val))
#else // __has_soft_bf16__
#define __extra_bf16_ripple_broadcast(PE, Mask, Val)                           \
  , __bf16 : __builtin_ripple_broadcast_bf16((PE), (Mask), (Val))
#endif // __has_soft_bf16__
#else
#define __extra_bf16_ripple_broadcast(PE, Mask, Val)
#endif

#if __has_Float16__
#define __extra_f16_ripple_broadcast(PE, Mask, Val)                            \
  , _Float16 : __builtin_ripple_broadcast_f16((PE), (Mask), (Val))
#else
#define __extra_f16_ripple_broadcast(PE, Mask, Val)
#endif

#define ripple_broadcast(PE, Mask, Val)                                        \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((Val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_broadcast_any_int(char, i, (PE), (Mask), (Val))    \
                 : __ripple_broadcast_any_int(char, u, (PE), (Mask), (Val))),  \
      signed char: __ripple_broadcast_any_int(signed char, i, (PE), (Mask),    \
                                              (Val)),                          \
      unsigned char: __ripple_broadcast_any_int(unsigned char, u, (PE),        \
                                                (Mask), (Val)),                \
      signed short: __ripple_broadcast_any_int(signed short, i, (PE), (Mask),  \
                                               (Val)),                         \
      unsigned short: __ripple_broadcast_any_int(unsigned short, u, (PE),      \
                                                 (Mask), (Val)),               \
      signed int: __ripple_broadcast_any_int(signed int, i, (PE), (Mask),      \
                                             (Val)),                           \
      unsigned int: __ripple_broadcast_any_int(unsigned int, u, (PE), (Mask),  \
                                               (Val)),                         \
      signed long: __ripple_broadcast_any_int(signed long, i, (PE), (Mask),    \
                                              (Val)),                          \
      unsigned long: __ripple_broadcast_any_int(unsigned long, u, (PE),        \
                                                (Mask), (Val)),                \
      signed long long: __ripple_broadcast_any_int(signed long long, i, (PE),  \
                                                   (Mask), (Val)),             \
      unsigned long long: __ripple_broadcast_any_int(unsigned long long, u,    \
                                                     (PE), (Mask), (Val)),     \
      float: __builtin_ripple_broadcast_f32((PE), (Mask), (Val)),              \
      double: __builtin_ripple_broadcast_f64((PE), (Mask), (Val))              \
          __extra_f16_ripple_broadcast((PE), (Mask), (Val))                    \
              __extra_bf16_ripple_broadcast((PE), (Mask), (Val)))              \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_broadcast_ptr(PE, Mask, Ptr)                                    \
  ((__typeof__(Ptr))__builtin_ripple_broadcast_p((PE), (Mask), (void *)(Ptr)))

#else // defined(__cplusplus)

#define spec_bcast_int(CT, IU)                                                 \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static CT ripple_broadcast(ripple_block_t BS, \
                                                            const CT Val) {    \
    return __ripple_broadcast_any_int(CT, IU, BS, msk, Val);                   \
  }

#define spec_bcast(CT, T)                                                      \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static CT ripple_broadcast(ripple_block_t BS, \
                                                            const CT Val) {    \
    return __builtin_ripple_broadcast_##T(BS, msk, Val);                       \
  }

#if __has_bf16__
#if __has_soft_bf16__
#define emulate_spec_bcast_bf16()                                              \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static __bf16 ripple_broadcast(               \
      ripple_block_t BS, const __bf16 Val) {                                   \
    return (__bf16)__builtin_ripple_broadcast_f32(BS, msk, (float)(Val));      \
  }
#else
#define emulate_spec_bcast_bf16()                                              \
  template <uint64_t msk>                                                      \
  __attribute__((always_inline)) static __bf16 ripple_broadcast(               \
      ripple_block_t BS, const __bf16 Val) {                                   \
    return __builtin_ripple_broadcast_bf16(BS, msk, Val);                      \
  }
#endif
#endif // __has_bf16__

#if __has_bf16__ && __has_Float16__
#define spec_bcast_ftypes()                                                    \
  spec_bcast(_Float16, f16);                                                   \
  emulate_spec_bcast_bf16();                                                   \
  spec_bcast(float, f32);                                                      \
  spec_bcast(double, f64);
#elif __has_Float16__
#define spec_bcast_ftypes()                                                    \
  spec_bcast(_Float16, f16);                                                   \
  spec_bcast(float, f32);                                                      \
  spec_bcast(double, f64);
#elif __has_bf16__
#define spec_bcast_ftypes()                                                    \
  emulate_spec_bcast_bf16();                                                   \
  spec_bcast(float, f32);                                                      \
  spec_bcast(double, f64);
#else // __has_Float16__
#define spec_bcast_ftypes()                                                    \
  spec_bcast(float, f32);                                                      \
  spec_bcast(double, f64);
#endif

#define spec_bcast_itypes()                                                    \
  spec_bcast_int(signed char, i);                                              \
  spec_bcast_int(unsigned char, u);                                            \
  spec_bcast_int(signed short, i);                                             \
  spec_bcast_int(unsigned short, u);                                           \
  spec_bcast_int(signed int, i);                                               \
  spec_bcast_int(unsigned int, u);                                             \
  spec_bcast_int(signed long, i);                                              \
  spec_bcast_int(unsigned long, u);                                            \
  spec_bcast_int(signed long long, i);                                         \
  spec_bcast_int(unsigned long long, u);

template <uint64_t msk>
__attribute__((always_inline)) static char ripple_broadcast(ripple_block_t BS,
                                                            const char Val) {
  return __ripple_char_is_signed ? __builtin_ripple_broadcast_i8(BS, msk, Val)
                                 : __builtin_ripple_broadcast_u8(BS, msk, Val);
}

spec_bcast_itypes();
spec_bcast_ftypes();

// Expose the same API as C for broadcasts
#define ripple_broadcast(PE, mask, val) ripple_broadcast<mask>(PE, val)
#define ripple_broadcast_ptr(PE, mask, ptr)                                    \
  (decltype(ptr))__builtin_ripple_broadcast_p(PE, mask, (void *)(ptr))

#undef spec_bcast
#undef spec_bcast_int
#undef spec_bcast_itypes
#undef spec_bcast_ftypes
#undef __ripple_broadcast_any_int

#endif // defined(__cplusplus)

/* _____________________________ Block slicing _______________________________*/

#define __ripple_slice_any_int(Type, UI, Val, ...)                             \
  ((Type)(sizeof(Type) == 1                                                    \
              ? __builtin_ripple_slice_##UI##8(Val, __VA_ARGS__)               \
              : (sizeof(Type) == 2                                             \
                     ? __builtin_ripple_slice_##UI##16(Val, __VA_ARGS__)       \
                     : (sizeof(Type) == 4 ? __builtin_ripple_slice_##UI##32(   \
                                                Val, __VA_ARGS__)              \
                                          : __builtin_ripple_slice_##UI##64(   \
                                                Val, __VA_ARGS__)))))

#ifndef __cplusplus

#if __has_bf16__
#if __has_soft_bf16__
#define __extra_bf16_ripple_slice(Val, ...)                                    \
  , __bf16 : (__bf16)__builtin_ripple_slice_f32((float)(Val), __VA_ARGS__)
#else // !__has_soft_bf16__
#define __extra_bf16_ripple_slice(Val, ...)                                    \
  , __bf16 : __builtin_ripple_slice_bf16((Val), __VA_ARGS__)
#endif // __has_soft_bf16__
#else
#define __extra_bf16_ripple_slice(Val, ...)
#endif

#if __has_Float16__
#define __extra_f16_ripple_slice(Val, ...)                                     \
  , _Float16 : __builtin_ripple_slice_f16((Val), __VA_ARGS__)
#else
#define __extra_f16_ripple_slice(Val, ...)
#endif

#define ripple_slice(Val, ...)                                                 \
  RIPPLE_DISABLE_GENERIC_WARNING                                               \
  _Generic((Val),                                                              \
      char: (__ripple_char_is_signed                                           \
                 ? __ripple_slice_any_int(char, i, (Val), __VA_ARGS__)         \
                 : __ripple_slice_any_int(char, u, (Val), __VA_ARGS__)),       \
      signed char: __ripple_slice_any_int(signed char, i, (Val), __VA_ARGS__), \
      unsigned char: __ripple_slice_any_int(unsigned char, u, (Val),           \
                                            __VA_ARGS__),                      \
      signed short: __ripple_slice_any_int(signed short, i, (Val),             \
                                           __VA_ARGS__),                       \
      unsigned short: __ripple_slice_any_int(unsigned short, u, (Val),         \
                                             __VA_ARGS__),                     \
      signed int: __ripple_slice_any_int(signed int, i, (Val), __VA_ARGS__),   \
      unsigned int: __ripple_slice_any_int(unsigned int, u, (Val),             \
                                           __VA_ARGS__),                       \
      signed long: __ripple_slice_any_int(signed long, i, (Val), __VA_ARGS__), \
      unsigned long: __ripple_slice_any_int(unsigned long, u, (Val),           \
                                            __VA_ARGS__),                      \
      signed long long: __ripple_slice_any_int(signed long long, i, (Val),     \
                                               __VA_ARGS__),                   \
      unsigned long long: __ripple_slice_any_int(unsigned long long, u, (Val), \
                                                 __VA_ARGS__),                 \
      float: __builtin_ripple_slice_f32((Val), __VA_ARGS__),                   \
      double: __builtin_ripple_slice_f64((Val), __VA_ARGS__)                   \
          __extra_f16_ripple_slice((Val), __VA_ARGS__)                         \
              __extra_bf16_ripple_slice((Val), __VA_ARGS__))                   \
      RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_slice_ptr(Ptr, ...)                                             \
  (__typeof__(Ptr))__builtin_ripple_slice_p((void *)(Ptr), __VA_ARGS__)

#else // defined(__cplusplus)

#define spec_slice(CT, T)                                                      \
  template <int arg0, int arg1 = -1, int arg2 = -1, int arg3 = -1,             \
            int arg4 = -1, int arg5 = -1, int arg6 = -1, int arg7 = -1,        \
            int arg8 = -1, int arg9 = -1>                                      \
  __attribute__((always_inline)) static CT ripple_slice(const CT Val) {        \
    return __builtin_ripple_slice_##T((Val), arg0, arg1, arg2, arg3, arg4,     \
                                      arg5, arg6, arg7, arg8, arg9);           \
  }

#define spec_slice_int(CT, UI)                                                 \
  template <int arg0, int arg1 = -1, int arg2 = -1, int arg3 = -1,             \
            int arg4 = -1, int arg5 = -1, int arg6 = -1, int arg7 = -1,        \
            int arg8 = -1, int arg9 = -1>                                      \
  __attribute__((always_inline)) static CT ripple_slice(const CT Val) {        \
    return __ripple_slice_any_int(CT, UI, (Val), arg0, arg1, arg2, arg3, arg4, \
                                  arg5, arg6, arg7, arg8, arg9);               \
  }

#if __has_bf16__
#if __has_soft_bf16__
#define emulate_spec_slice_bf16()                                              \
  template <int arg0, int arg1 = -1, int arg2 = -1, int arg3 = -1,             \
            int arg4 = -1, int arg5 = -1, int arg6 = -1, int arg7 = -1,        \
            int arg8 = -1, int arg9 = -1>                                      \
  __attribute__((always_inline)) static __bf16 ripple_slice(                   \
      const __bf16 Val) {                                                      \
    return (__bf16)__builtin_ripple_slice_f32((float)(Val), arg0, arg1, arg2,  \
                                              arg3, arg4, arg5, arg6, arg7,    \
                                              arg8, arg9);                     \
  }

#else
#define emulate_spec_slice_bf16()                                              \
  template <int arg0, int arg1 = -1, int arg2 = -1, int arg3 = -1,             \
            int arg4 = -1, int arg5 = -1, int arg6 = -1, int arg7 = -1,        \
            int arg8 = -1, int arg9 = -1>                                      \
  __attribute__((always_inline)) static __bf16 ripple_slice(                   \
      const __bf16 Val) {                                                      \
    return __builtin_ripple_slice_bf16((Val), arg0, arg1, arg2, arg3, arg4,    \
                                       arg5, arg6, arg7, arg8, arg9);          \
  }
#endif
#endif

#if __has_bf16__ && __has_Float16__
#define spec_slice_ftypes()                                                    \
  spec_slice(_Float16, f16);                                                   \
  emulate_spec_slice_bf16();                                                   \
  spec_slice(float, f32);                                                      \
  spec_slice(double, f64)
#elif __has_Float16__
#define spec_slice_ftypes()                                                    \
  spec_slice(_Float16, f16);                                                   \
  spec_slice(float, f32);                                                      \
  spec_slice(double, f64);
#elif __has_bf16__
#define spec_slice_ftypes()                                                    \
  emulate_spec_slice_bf16();                                                   \
  spec_slice(float, f32);                                                      \
  spec_slice(double, f64);
#else // __has_Float16__
#define spec_slice_ftypes()                                                    \
  spec_slice(float, f32);                                                      \
  spec_slice(double, f64);
#endif

#define spec_slice_itypes()                                                    \
  spec_slice_int(signed char, i);                                              \
  spec_slice_int(unsigned char, u);                                            \
  spec_slice_int(signed short, i);                                             \
  spec_slice_int(unsigned short, u);                                           \
  spec_slice_int(signed int, i);                                               \
  spec_slice_int(unsigned int, u);                                             \
  spec_slice_int(signed long, i);                                              \
  spec_slice_int(unsigned long, u);                                            \
  spec_slice_int(signed long long, i);                                         \
  spec_slice_int(unsigned long long, u);

template <int arg0, int arg1 = -1, int arg2 = -1, int arg3 = -1, int arg4 = -1,
          int arg5 = -1, int arg6 = -1, int arg7 = -1, int arg8 = -1,
          int arg9 = -1>
__attribute__((always_inline)) static char ripple_slice(const char Val) {
  return __ripple_char_is_signed
             ? __ripple_slice_any_int(char, i, (Val), arg0, arg1, arg2, arg3,
                                      arg4, arg5, arg6, arg7, arg8, arg9)
             : __ripple_slice_any_int(char, u, (Val), arg0, arg1, arg2, arg3,
                                      arg4, arg5, arg6, arg7, arg8, arg9);
}

spec_slice_itypes();
spec_slice_ftypes();

// Expose the same API as C for broadcasts
#define ripple_slice(Val, ...) ripple_slice<__VA_ARGS__>(Val)
#define ripple_slice_ptr(Ptr, ...)                                             \
  (__typeof__(Ptr))__builtin_ripple_slice_p((void *)(Ptr), __VA_ARGS__)

#undef spec_slice
#undef spec_slice_int
#undef spec_slice_itypes
#undef spec_slice_ftypes
#undef spec_slice_p
#undef spec_slice_ptypes
#undef __ripple_slice_any_int

#endif // defined(__cplusplus)

/// Ripple pointer alignment assumptions hints

#ifdef __cplusplus
#define __ripple_decltype_like(x) decltype(x)
#else
#define __ripple_decltype_like(x) __typeof__(x)
#endif

#define ripple_ptr_alignment_slice(AddrTensor, Alignment, SliceIdx, ...)       \
  RIPPLE_DISABLE_GENERIC_WARNING({                                             \
    __ripple_decltype_like(AddrTensor) AddrTsor = (AddrTensor);                \
    (void)__builtin_assume_aligned(                                            \
        ripple_slice_ptr(AddrTsor,                                             \
                         __ripple_take_ten((SliceIdx), ##__VA_ARGS__, 0, 0, 0, \
                                           0, 0, 0, 0, 0, 0, 0)),              \
        (Alignment));                                                          \
    AddrTsor;                                                                  \
  })                                                                           \
  RIPPLE_REENABLE_GENERIC_WARNING

#define ripple_ptr_alignment(AddrTensor, Alignment)                            \
  ripple_ptr_alignment_slice((AddrTensor), (Alignment), 0)
