//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Bitwise operations on floating-point numbers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_BITWISE_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_BITWISE_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h"

#ifdef LIBC_TARGET_CPU_HAS_SSE2
#include <immintrin.h>
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif // LIBC_TARGET_CPU_HAS_SSE2

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

// Bitwise AND of two floating-point values (a & b), performed directly in
// FP/SIMD registers when SSE2 or AArch64 NEON is available to avoid GPR
// round-trips.
template <typename T>
LIBC_ALWAYS_INLINE constexpr cpp::enable_if_t<cpp::is_floating_point_v<T>, T>
fp_and(T a, T b) {
  if (!cpp::is_constant_evaluated()) {
#ifdef LIBC_TARGET_CPU_HAS_SSE2
    if constexpr (cpp::is_same_v<T, float>)
      return _mm_cvtss_f32(_mm_and_ps(_mm_set_ss(a), _mm_set_ss(b)));
    if constexpr (cpp::is_same_v<T, double>)
      return _mm_cvtsd_f64(_mm_and_pd(_mm_set_sd(a), _mm_set_sd(b)));
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64) && defined(__ARM_NEON)
    if constexpr (cpp::is_same_v<T, float>)
      return vget_lane_f32(
          vreinterpret_f32_u32(vand_u32(vreinterpret_u32_f32(vdup_n_f32(a)),
                                        vreinterpret_u32_f32(vdup_n_f32(b)))),
          0);
    if constexpr (cpp::is_same_v<T, double>)
      return vget_lane_f64(
          vreinterpret_f64_u64(vand_u64(vreinterpret_u64_f64(vdup_n_f64(a)),
                                        vreinterpret_u64_f64(vdup_n_f64(b)))),
          0);
#endif
  }
  using StorageType = typename FPBits<T>::StorageType;
  return FPBits<T>(static_cast<StorageType>(FPBits<T>(a).uintval() &
                                            FPBits<T>(b).uintval()))
      .get_val();
}

// Bitwise OR of two floating-point values (a | b), performed directly in
// FP/SIMD registers when SSE2 or AArch64 NEON is available to avoid GPR
// round-trips.
template <typename T>
LIBC_ALWAYS_INLINE constexpr cpp::enable_if_t<cpp::is_floating_point_v<T>, T>
fp_or(T a, T b) {
  if (!cpp::is_constant_evaluated()) {
#ifdef LIBC_TARGET_CPU_HAS_SSE2
    if constexpr (cpp::is_same_v<T, float>)
      return _mm_cvtss_f32(_mm_or_ps(_mm_set_ss(a), _mm_set_ss(b)));
    if constexpr (cpp::is_same_v<T, double>)
      return _mm_cvtsd_f64(_mm_or_pd(_mm_set_sd(a), _mm_set_sd(b)));
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64) && defined(__ARM_NEON)
    if constexpr (cpp::is_same_v<T, float>)
      return vget_lane_f32(
          vreinterpret_f32_u32(vorr_u32(vreinterpret_u32_f32(vdup_n_f32(a)),
                                        vreinterpret_u32_f32(vdup_n_f32(b)))),
          0);
    if constexpr (cpp::is_same_v<T, double>)
      return vget_lane_f64(
          vreinterpret_f64_u64(vorr_u64(vreinterpret_u64_f64(vdup_n_f64(a)),
                                        vreinterpret_u64_f64(vdup_n_f64(b)))),
          0);
#endif
  }
  using StorageType = typename FPBits<T>::StorageType;
  return FPBits<T>(static_cast<StorageType>(FPBits<T>(a).uintval() |
                                            FPBits<T>(b).uintval()))
      .get_val();
}

// Bitwise XOR of two floating-point values (a ^ b), performed directly in
// FP/SIMD registers when SSE2 or AArch64 NEON is available to avoid GPR
// round-trips.
template <typename T>
LIBC_ALWAYS_INLINE constexpr cpp::enable_if_t<cpp::is_floating_point_v<T>, T>
fp_xor(T a, T b) {
  if (!cpp::is_constant_evaluated()) {
#ifdef LIBC_TARGET_CPU_HAS_SSE2
    if constexpr (cpp::is_same_v<T, float>)
      return _mm_cvtss_f32(_mm_xor_ps(_mm_set_ss(a), _mm_set_ss(b)));
    if constexpr (cpp::is_same_v<T, double>)
      return _mm_cvtsd_f64(_mm_xor_pd(_mm_set_sd(a), _mm_set_sd(b)));
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64) && defined(__ARM_NEON)
    if constexpr (cpp::is_same_v<T, float>)
      return vget_lane_f32(
          vreinterpret_f32_u32(veor_u32(vreinterpret_u32_f32(vdup_n_f32(a)),
                                        vreinterpret_u32_f32(vdup_n_f32(b)))),
          0);
    if constexpr (cpp::is_same_v<T, double>)
      return vget_lane_f64(
          vreinterpret_f64_u64(veor_u64(vreinterpret_u64_f64(vdup_n_f64(a)),
                                        vreinterpret_u64_f64(vdup_n_f64(b)))),
          0);
#endif
  }
  using StorageType = typename FPBits<T>::StorageType;
  return FPBits<T>(static_cast<StorageType>(FPBits<T>(a).uintval() ^
                                            FPBits<T>(b).uintval()))
      .get_val();
}

// Bitwise AND-NOT of two floating-point values (a & ~b), clearing the bits set
// in b from a. Performed directly in FP/SIMD registers when SSE2 or AArch64
// NEON is available to avoid GPR round-trips.
template <typename T>
LIBC_ALWAYS_INLINE constexpr cpp::enable_if_t<cpp::is_floating_point_v<T>, T>
fp_andnot(T a, T b) {
  if (!cpp::is_constant_evaluated()) {
#ifdef LIBC_TARGET_CPU_HAS_SSE2
    if constexpr (cpp::is_same_v<T, float>)
      return _mm_cvtss_f32(_mm_andnot_ps(_mm_set_ss(b), _mm_set_ss(a)));
    if constexpr (cpp::is_same_v<T, double>)
      return _mm_cvtsd_f64(_mm_andnot_pd(_mm_set_sd(b), _mm_set_sd(a)));
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64) && defined(__ARM_NEON)
    if constexpr (cpp::is_same_v<T, float>)
      return vget_lane_f32(
          vreinterpret_f32_u32(vbic_u32(vreinterpret_u32_f32(vdup_n_f32(a)),
                                        vreinterpret_u32_f32(vdup_n_f32(b)))),
          0);
    if constexpr (cpp::is_same_v<T, double>)
      return vget_lane_f64(
          vreinterpret_f64_u64(vbic_u64(vreinterpret_u64_f64(vdup_n_f64(a)),
                                        vreinterpret_u64_f64(vdup_n_f64(b)))),
          0);
#endif
  }
  using StorageType = typename FPBits<T>::StorageType;
  return FPBits<T>(static_cast<StorageType>(FPBits<T>(a).uintval() &
                                            ~FPBits<T>(b).uintval()))
      .get_val();
}

// Bitwise NOT of a floating-point value (~a), performed directly in FP/SIMD
// registers when SSE2 or AArch64 NEON is available to avoid GPR round-trips.
template <typename T>
LIBC_ALWAYS_INLINE constexpr cpp::enable_if_t<cpp::is_floating_point_v<T>, T>
fp_not(T a) {
  if (!cpp::is_constant_evaluated()) {
#ifdef LIBC_TARGET_CPU_HAS_SSE2
    if constexpr (cpp::is_same_v<T, float>)
      return _mm_cvtss_f32(
          _mm_xor_ps(_mm_set_ss(a), _mm_castsi128_ps(_mm_set1_epi32(-1))));
    if constexpr (cpp::is_same_v<T, double>)
      return _mm_cvtsd_f64(
          _mm_xor_pd(_mm_set_sd(a), _mm_castsi128_pd(_mm_set1_epi64x(-1))));
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64) && defined(__ARM_NEON)
    if constexpr (cpp::is_same_v<T, float>)
      return vget_lane_f32(
          vreinterpret_f32_u32(vmvn_u32(vreinterpret_u32_f32(vdup_n_f32(a)))),
          0);
    if constexpr (cpp::is_same_v<T, double>)
      return vget_lane_f64(
          vreinterpret_f64_u32(vmvn_u32(vreinterpret_u32_f64(vdup_n_f64(a)))),
          0);
#endif
  }
  using StorageType = typename FPBits<T>::StorageType;
  FPBits<T> res(a);
  res.set_uintval(static_cast<StorageType>(~res.uintval()));
  return res.get_val();
}

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_BITWISE_H
