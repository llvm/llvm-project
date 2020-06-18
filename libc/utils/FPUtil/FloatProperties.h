//===-- Properties of floating point numbers --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_FLOAT_PROPERTIES_H
#define LLVM_LIBC_UTILS_FPUTIL_FLOAT_PROPERTIES_H

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

template <typename T> struct FloatProperties {};

template <> struct FloatProperties<float> {
  typedef uint32_t BitsType;
  static_assert(sizeof(BitsType) == sizeof(float),
                "Unexpected size of 'float' type.");

  static constexpr uint32_t bitWidth = sizeof(BitsType) << 3;

  static constexpr uint32_t mantissaWidth = 23;
  static constexpr BitsType mantissaMask = 0x007fffffU;
  static constexpr BitsType signMask = 0x80000000U;
  static constexpr BitsType exponentMask = ~(signMask | mantissaMask);
  static constexpr uint32_t exponentOffset = 127;

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType quietNaNMask = 0x00400000U;
};

template <> struct FloatProperties<double> {
  typedef uint64_t BitsType;
  static_assert(sizeof(BitsType) == sizeof(double),
                "Unexpected size of 'double' type.");

  static constexpr uint32_t bitWidth = sizeof(BitsType) << 3;

  static constexpr uint32_t mantissaWidth = 52;
  static constexpr BitsType mantissaMask = 0x000fffffffffffffU;
  static constexpr BitsType signMask = 0x8000000000000000ULL;
  static constexpr BitsType exponentMask = ~(signMask | mantissaMask);
  static constexpr uint32_t exponentOffset = 1023;

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType quietNaNMask = 0x0008000000000000ULL;
};

// Define the float type corresponding to the BitsType.
template <typename BitsType> struct FloatType;

template <> struct FloatType<uint32_t> {
  static_assert(sizeof(uint32_t) == sizeof(float),
                "Unexpected size of 'float' type.");
  typedef float Type;
};

template <> struct FloatType<uint64_t> {
  static_assert(sizeof(uint64_t) == sizeof(double),
                "Unexpected size of 'double' type.");
  typedef double Type;
};

template <typename BitsType>
using FloatTypeT = typename FloatType<BitsType>::Type;

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_FLOAT_PROPERTIES_H
