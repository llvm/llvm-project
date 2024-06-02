//===-- Implementation of fmul function------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmul.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

namespace Fmul {
uint64_t maxu(uint64_t A, uint64_t B) { return A > B ? A : B; }

uint64_t mul(uint64_t a, uint64_t b) {
  __uint128_t product =
      static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  return static_cast<uint64_t>(product >> 64);
}

uint64_t mullow(uint64_t a, uint64_t b) {
  __uint128_t product =
      static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  return static_cast<uint64_t>(product);
}

uint64_t nlz(uint64_t x) {
  uint64_t z = 0;

  if (x == 0)
    return 64;
  if (x <= 0x00000000FFFFFFFF) {
    z = z + 32;
    x = x << 32;
  }
  if (x <= 0x0000FFFFFFFFFFFF) {
    z = z + 16;
    x = x << 16;
  }
  if (x <= 0x00FFFFFFFFFFFFFF) {
    z = z + 8;
    x = x << 8;
  }
  if (x <= 0x0FFFFFFFFFFFFFFF) {
    z = z + 4;
    x = x << 4;
  }
  if (x <= 0x3FFFFFFFFFFFFFFF) {
    z = z + 2;
    x = x << 2;
  }
  if (x <= 0x7FFFFFFFFFFFFFFF) {
    z = z + 1;
  }
  return z;
}

float fmul(double x, double y) {

  auto x_bits = fputil::FPBits<double>(x);
  uint64_t x_u = x_bits.uintval();

  auto y_bits = fputil::FPBits<double>(y);
  uint64_t y_u = y_bits.uintval();

  uint64_t absx = x_u & 0x7FFFFFFFFFFFFFFF;
  uint64_t absy = y_u & 0x7FFFFFFFFFFFFFFF;

  uint64_t exponent_x = absx >> 52;
  uint64_t exponent_y = absy >> 52;

  uint64_t mx, my;

  mx = maxu(nlz(absx), 11);

  my = maxu(nlz(absy), 11);

  int32_t dm1;
  uint64_t mpx, mpy, highs, lows, b;
  uint64_t g, hight, lowt, morlowt, c, m;
  mpx = (x_u << mx) | 0x8000000000000000;
  mpy = (y_u << my) | 0x8000000000000000;
  highs = mul(mpx, mpy);
  c = highs >= 0x8000000000000000;
  lows = mullow(mpx, mpy);

  lowt = (lows != 0);

  g = (highs >> (38 + c)) & 1;
  hight = (highs << (55 - c)) != 0;

  int32_t exint = static_cast<int32_t>(exponent_x);
  int32_t eyint = static_cast<int32_t>(exponent_y);
  int32_t cint = static_cast<int32_t>(c);
  dm1 = ((exint + eyint) - 1919) + cint;

  if (dm1 >= 255) {
    dm1 = 255;

    m = 0;
  } else if (dm1 <= 0) {
    m = static_cast<uint32_t>((highs >> (39 + c)) >> (1 - dm1));
    dm1 = 0;
    morlowt = m | lowt;
    b = g & (morlowt | hight);
  } else {
    m = static_cast<uint32_t>(highs >> (39 + c));
    morlowt = m | lowt;
    b = g & (morlowt | hight);
  }

  uint32_t sr = static_cast<uint32_t>((x_u ^ y_u) & 0x8000000000000000);

  uint32_t exp16 = sr | (dm1 << 23);
  if (dm1 == 0) {
    uint32_t m2 = static_cast<uint32_t>(m);
    uint32_t result =
        (static_cast<uint32_t>(exp16) + m2) + static_cast<uint32_t>(b);

    float result32 = cpp::bit_cast<float>(result);

    return result32;

  } else {
    constexpr uint32_t FLOAT32_MANTISSA_MASK =
        0b00000000011111111111111111111111;
    uint32_t m2 = static_cast<uint32_t>(m) & FLOAT32_MANTISSA_MASK;

    uint32_t result =
        (static_cast<uint32_t>(exp16) + m2) + static_cast<uint32_t>(b);

    float result16 = cpp::bit_cast<float>(result);

    return result16;
  }
}
} // namespace Fmul

LLVM_LIBC_FUNCTION(float, fmul, (double x, double y)) {
  return Fmul::fmul(x, y);
}

} // namespace LIBC_NAMESPACE
