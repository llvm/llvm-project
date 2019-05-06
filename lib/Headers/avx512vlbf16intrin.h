/*===--------- avx512vlbf16intrin.h - AVX512_BF16 intrinsics ---------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error "Never use <avx512vlbf16intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VLBF16INTRIN_H
#define __AVX512VLBF16INTRIN_H

typedef short __m128bh __attribute__((__vector_size__(16), __aligned__(16)));

#define __DEFAULT_FN_ATTRS128 \
  __attribute__((__always_inline__, __nodebug__, \
                 __target__("avx512vl, avx512bf16"), __min_vector_width__(128)))
#define __DEFAULT_FN_ATTRS256 \
  __attribute__((__always_inline__, __nodebug__, \
                 __target__("avx512vl, avx512bf16"), __min_vector_width__(256)))

/// Convert Two Packed Single Data to One Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNE2PS2BF16 </c> instructions.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [8 x bfloat] whose lower 64 bits come from
///    convertion of src2, and higher 64 bits come from conversion of src1.
static __inline__ __m128bh __DEFAULT_FN_ATTRS128
_mm_cvtne2ps_pbh(__m128 __A, __m128 __B) {
  return (__m128bh)__builtin_ia32_cvtne2ps2bf16_128((__v4sf) __A,
                                                    (__v4sf) __B);
}

/// Convert Two Packed Single Data to One Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNE2PS2BF16 </c> instructions.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \param __B
///    A 128-bit vector of [4 x float].
/// \param __W
///    A 128-bit vector of [8 x bfloat].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A or __B, 0 means __W.
/// \returns A 128-bit vector of [8 x bfloat] whose lower 64 bits come from
///    convertion of src2, and higher 64 bits come from conversion of src1.
static __inline__ __m128bh __DEFAULT_FN_ATTRS128
_mm_mask_cvtne2ps_pbh(__m128bh __W, __mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128bh)__builtin_ia32_selectw_128((__mmask8)__U,
                                             (__v8hi)_mm_cvtne2ps_pbh(__A, __B),
                                             (__v8hi)__W);
}

/// Convert Two Packed Single Data to One Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNE2PS2BF16 </c> instructions.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \param __B
///    A 128-bit vector of [4 x float].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A or __B, 0 means zero.
/// \returns A 128-bit vector of [8 x bfloat] whose lower 64 bits come from
///    convertion of src2, and higher 64 bits come from conversion of src1.
static __inline__ __m128bh __DEFAULT_FN_ATTRS128
_mm_maskz_cvtne2ps_pbh(__mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128bh)__builtin_ia32_selectw_128((__mmask8)__U,
                                             (__v8hi)_mm_cvtne2ps_pbh(__A, __B),
                                             (__v8hi)_mm_setzero_si128());
}

/// Convert Two Packed Single Data to One Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNE2PS2BF16 </c> instructions.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns A 256-bit vector of [16 x bfloat] whose lower 128 bits come from
///    convertion of src2, and higher 128 bits come from conversion of src1.
static __inline__ __m256bh __DEFAULT_FN_ATTRS256
_mm256_cvtne2ps_pbh(__m256 __A, __m256 __B) {
  return (__m256bh)__builtin_ia32_cvtne2ps2bf16_256((__v8sf) __A,
                                                    (__v8sf) __B);
}

/// Convert Two Packed Single Data to One Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNE2PS2BF16 </c> instructions.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \param __B
///    A 256-bit vector of [8 x float].
/// \param __W
///    A 256-bit vector of [16 x bfloat].
/// \param __U
///    An immediate value containing an 16-bit value specifying which element
///    is choosed. 1 means __A or __B, 0 means __W.
/// \returns A 256-bit vector of [16 x bfloat] whose lower 128 bits come from
///    convertion of src2, and higher 128 bits come from conversion of src1.
static __inline__ __m256bh __DEFAULT_FN_ATTRS256
_mm256_mask_cvtne2ps_pbh(__m256bh __W, __mmask16 __U, __m256 __A, __m256 __B) {
  return (__m256bh)__builtin_ia32_selectw_256((__mmask16)__U,
                                         (__v16hi)_mm256_cvtne2ps_pbh(__A, __B),
                                         (__v16hi)__W);
}

/// Convert Two Packed Single Data to One Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNE2PS2BF16 </c> instructions.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \param __B
///    A 256-bit vector of [8 x float].
/// \param __U
///    An immediate value containing an 16-bit value specifying which element
///    is choosed. 1 means __A or __B, 0 means zero.
/// \returns A 256-bit vector of [16 x bfloat] whose lower 128 bits come from
///    convertion of src2, and higher 128 bits come from conversion of src1.
static __inline__ __m256bh __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtne2ps_pbh(__mmask16 __U, __m256 __A, __m256 __B) {
  return (__m256bh)__builtin_ia32_selectw_256((__mmask16)__U,
                                         (__v16hi)_mm256_cvtne2ps_pbh(__A, __B),
                                         (__v16hi)_mm256_setzero_si256());
}

/// Convert Packed Single Data to Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNEPS2BF16 </c> instructions.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [8 x bfloat] whose lower 64 bits come from
///    convertion of src, and higher 64 bits are 0.
static __inline__ __m128bh __DEFAULT_FN_ATTRS128
_mm_cvtneps_pbh(__m128 __A) {
  return (__m128bh)__builtin_ia32_cvtneps2bf16_128_mask((__v4sf) __A,
                                                  (__v8hi)_mm_undefined_si128(),
                                                  (__mmask8)-1);
}

/// Convert Packed Single Data to Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNEPS2BF16 </c> instructions.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \param __W
///    A 128-bit vector of [8 x bfloat].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A, 0 means __W.
/// \returns A 128-bit vector of [8 x bfloat] whose lower 64 bits come from
///    convertion of src, and higher 64 bits are 0.
static __inline__ __m128bh __DEFAULT_FN_ATTRS128
_mm_mask_cvtneps_pbh(__m128bh __W, __mmask8 __U, __m128 __A) {
  return (__m128bh)__builtin_ia32_cvtneps2bf16_128_mask((__v4sf) __A,
                                                        (__v8hi)__W,
                                                        (__mmask8)__U);
}

/// Convert Packed Single Data to Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNEPS2BF16 </c> instructions.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A, 0 means 0.
/// \returns A 128-bit vector of [8 x bfloat] whose lower 64 bits come from
///    convertion of src, and higher 64 bits are 0.
static __inline__ __m128bh __DEFAULT_FN_ATTRS128
_mm_maskz_cvtneps_pbh(__mmask8 __U, __m128 __A) {
  return (__m128bh)__builtin_ia32_cvtneps2bf16_128_mask((__v4sf) __A,
                                                    (__v8hi)_mm_setzero_si128(),
                                                    (__mmask8)__U);
}

/// Convert Packed Single Data to Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNEPS2BF16 </c> instructions.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns A 128-bit vector of [8 x bfloat] comes from convertion of src.
static __inline__ __m128bh __DEFAULT_FN_ATTRS256
_mm256_cvtneps_pbh(__m256 __A) {
  return (__m128bh)__builtin_ia32_cvtneps2bf16_256((__v8sf)__A);
}

/// Convert Packed Single Data to Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNEPS2BF16 </c> instructions.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \param __W
///    A 256-bit vector of [8 x bfloat].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A, 0 means __W.
/// \returns A 128-bit vector of [8 x bfloat] comes from convertion of src.
static __inline__ __m128bh __DEFAULT_FN_ATTRS256
_mm256_mask_cvtneps_pbh(__m128bh __W, __mmask8 __U, __m256 __A) {
  return (__m128bh)__builtin_ia32_selectw_128((__mmask8)__U,
                                              (__v8hi)_mm256_cvtneps_pbh(__A),
                                              (__v8hi)__W);
}

/// Convert Packed Single Data to Packed BF16 Data.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VCVTNEPS2BF16 </c> instructions.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A, 0 means __W.
/// \returns A 128-bit vector of [8 x bfloat] comes from convertion of src.
static __inline__ __m128bh __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtneps_pbh(__mmask8 __U, __m256 __A) {
  return (__m128bh)__builtin_ia32_selectw_128((__mmask8)__U,
                                              (__v8hi)_mm256_cvtneps_pbh(__A),
                                              (__v8hi)_mm_setzero_si128());
}

/// Dot Product of BF16 Pairs Accumulated into Packed Single Precision.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VDPBF16PS </c> instructions.
///
/// \param __A
///    A 128-bit vector of [8 x bfloat].
/// \param __B
///    A 128-bit vector of [8 x bfloat].
/// \param __D
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] comes from  Dot Product of
///  __A, __B and __D
static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_dpbf16_ps(__m128 __D, __m128bh __A, __m128bh __B) {
  return (__m128)__builtin_ia32_dpbf16ps_128((__v4sf)__D,
                                             (__v4si)__A,
                                             (__v4si)__B);
}

/// Dot Product of BF16 Pairs Accumulated into Packed Single Precision.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VDPBF16PS </c> instructions.
///
/// \param __A
///    A 128-bit vector of [8 x bfloat].
/// \param __B
///    A 128-bit vector of [8 x bfloat].
/// \param __D
///    A 128-bit vector of [4 x float].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A and __B's dot product, 0 means __D.
/// \returns A 128-bit vector of [4 x float] comes from  Dot Product of
///  __A, __B and __D
static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_mask_dpbf16_ps(__m128 __D, __mmask8 __U, __m128bh __A, __m128bh __B) {
  return (__m128)__builtin_ia32_selectps_128((__mmask8)__U,
                                           (__v4sf)_mm_dpbf16_ps(__D, __A, __B),
                                           (__v4sf)__D);
}

/// Dot Product of BF16 Pairs Accumulated into Packed Single Precision.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VDPBF16PS </c> instructions.
///
/// \param __A
///    A 128-bit vector of [8 x bfloat].
/// \param __B
///    A 128-bit vector of [8 x bfloat].
/// \param __D
///    A 128-bit vector of [4 x float].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A and __B's dot product, 0 means 0.
/// \returns A 128-bit vector of [4 x float] comes from  Dot Product of
///  __A, __B and __D
static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_maskz_dpbf16_ps(__mmask8 __U, __m128 __D, __m128bh __A, __m128bh __B) {
  return (__m128)__builtin_ia32_selectps_128((__mmask8)__U,
                                           (__v4sf)_mm_dpbf16_ps(__D, __A, __B),
                                           (__v4sf)_mm_setzero_si128());
}

/// Dot Product of BF16 Pairs Accumulated into Packed Single Precision.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VDPBF16PS </c> instructions.
///
/// \param __A
///    A 256-bit vector of [16 x bfloat].
/// \param __B
///    A 256-bit vector of [16 x bfloat].
/// \param __D
///    A 256-bit vector of [8 x float].
/// \returns A 256-bit vector of [8 x float] comes from  Dot Product of
///  __A, __B and __D
static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_dpbf16_ps(__m256 __D, __m256bh __A, __m256bh __B) {
  return (__m256)__builtin_ia32_dpbf16ps_256((__v8sf)__D,
                                             (__v8si)__A,
                                             (__v8si)__B);
}

/// Dot Product of BF16 Pairs Accumulated into Packed Single Precision.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VDPBF16PS </c> instructions.
///
/// \param __A
///    A 256-bit vector of [16 x bfloat].
/// \param __B
///    A 256-bit vector of [16 x bfloat].
/// \param __D
///    A 256-bit vector of [8 x float].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A and __B's dot product, 0 means __D.
/// \returns A 256-bit vector of [8 x float] comes from  Dot Product of
///  __A, __B and __D
static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_mask_dpbf16_ps(__m256 __D, __mmask8 __U, __m256bh __A, __m256bh __B) {
  return (__m256)__builtin_ia32_selectps_256((__mmask8)__U,
                                        (__v8sf)_mm256_dpbf16_ps(__D, __A, __B),
                                        (__v8sf)__D);
}

/// Dot Product of BF16 Pairs Accumulated into Packed Single Precision.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VDPBF16PS </c> instructions.
///
/// \param __A
///    A 256-bit vector of [16 x bfloat].
/// \param __B
///    A 256-bit vector of [16 x bfloat].
/// \param __D
///    A 256-bit vector of [8 x float].
/// \param __U
///    An immediate value containing an 8-bit value specifying which element
///    is choosed. 1 means __A and __B's dot product, 0 means 0.
/// \returns A 256-bit vector of [8 x float] comes from  Dot Product of
///  __A, __B and __D
static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_maskz_dpbf16_ps(__mmask8 __U, __m256 __D, __m256bh __A, __m256bh __B) {
  return (__m256)__builtin_ia32_selectps_256((__mmask8)__U,
                                        (__v8sf)_mm256_dpbf16_ps(__D, __A, __B),
                                        (__v8sf)_mm256_setzero_si256());
}
#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256

#endif
