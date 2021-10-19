//===-- Properties of floating point numbers --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_FLOAT_PROPERTIES_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_FLOAT_PROPERTIES_H

#include "PlatformDefs.h"
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
  static constexpr uint32_t exponentWidth = 8;
  static constexpr BitsType mantissaMask = (BitsType(1) << mantissaWidth) - 1;
  static constexpr BitsType signMask = BitsType(1)
                                       << (exponentWidth + mantissaWidth);
  static constexpr BitsType exponentMask = ~(signMask | mantissaMask);
  static constexpr uint32_t exponentBias = 127;

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
  static constexpr uint32_t exponentWidth = 11;
  static constexpr BitsType mantissaMask = (BitsType(1) << mantissaWidth) - 1;
  static constexpr BitsType signMask = BitsType(1)
                                       << (exponentWidth + mantissaWidth);
  static constexpr BitsType exponentMask = ~(signMask | mantissaMask);
  static constexpr uint32_t exponentBias = 1023;

  // If a number x is a NAN, then it is a quiet NAN if:
  //   QuietNaNMask & bits(x) != 0
  // Else, it is a signalling NAN.
  static constexpr BitsType quietNaNMask = 0x0008000000000000ULL;
};

#if defined(LONG_DOUBLE_IS_DOUBLE)
// Properties for numbers represented in 64 bits long double on Windows
// platform.
template <> struct FloatProperties<long double> {
  typedef uint64_t BitsType;
  static_assert(sizeof(BitsType) == sizeof(double),
                "Unexpected size of 'double' type.");

  static constexpr uint32_t bitWidth = FloatProperties<double>::bitWidth;

  static constexpr uint32_t mantissaWidth =
      FloatProperties<double>::mantissaWidth;
  static constexpr uint32_t exponentWidth =
      FloatProperties<double>::exponentWidth;
  static constexpr BitsType mantissaMask =
      FloatProperties<double>::mantissaMask;
  static constexpr BitsType signMask = FloatProperties<double>::signMask;
  static constexpr BitsType exponentMask =
      FloatProperties<double>::exponentMask;
  static constexpr uint32_t exponentBias =
      FloatProperties<double>::exponentBias;
};
#elif defined(SPECIAL_X86_LONG_DOUBLE)
// Properties for numbers represented in 80 bits long double on non-Windows x86
// platforms.
template <> struct FloatProperties<long double> {
  typedef __uint128_t BitsType;
  static_assert(sizeof(BitsType) == sizeof(long double),
                "Unexpected size of 'long double' type.");

  static constexpr uint32_t bitWidth = (sizeof(BitsType) << 3) - 48;

  static constexpr uint32_t mantissaWidth = 63;
  static constexpr uint32_t exponentWidth = 15;
  static constexpr BitsType mantissaMask = (BitsType(1) << mantissaWidth) - 1;
  static constexpr BitsType signMask = BitsType(1)
                                       << (exponentWidth + mantissaWidth + 1);
  static constexpr BitsType exponentMask = ((BitsType(1) << exponentWidth) - 1)
                                           << (mantissaWidth + 1);
  static constexpr uint32_t exponentBias = 16383;
};
#else
// Properties for numbers represented in 128 bits long double on non x86
// platform.
template <> struct FloatProperties<long double> {
  typedef __uint128_t BitsType;
  static_assert(sizeof(BitsType) == sizeof(long double),
                "Unexpected size of 'long double' type.");

  static constexpr uint32_t bitWidth = sizeof(BitsType) << 3;

  static constexpr uint32_t mantissaWidth = 112;
  static constexpr uint32_t exponentWidth = 15;
  static constexpr BitsType mantissaMask = (BitsType(1) << mantissaWidth) - 1;
  static constexpr BitsType signMask = BitsType(1)
                                       << (exponentWidth + mantissaWidth);
  static constexpr BitsType exponentMask = ~(signMask | mantissaMask);
  static constexpr uint32_t exponentBias = 16383;
};
#endif

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

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FLOAT_PROPERTIES_H
