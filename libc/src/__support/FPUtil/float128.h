//===-- Utilities for Float128 data type  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H

#include "FPBits.h"
#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/uint128.h"

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct Float128 {
  UInt128 bits;
  // Testing
  constexpr Float128() = default;
  /* TODO: precision
     TODO: explicit so it does not convert without warn
     VERIFY :   template <cpp::enable_if_t<fputil::get_fp_type<Double>() ==
                                 fputil::FPType::IEEE754_Binary64,
                             int> = 0>
  */
  constexpr Float128(double x) {
    FPBits<double> x_bits(x);
    uint64_t val = x_bits.uintval();
    bits = fputil::cast<UInt128>(val) << 64;
  }

  constexpr operator double() const {
    uint64_t val = static_cast<uint64_t>(bits >> 64U);
    return cpp::bit_cast<double>(val);
  }
};

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H
