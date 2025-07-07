//===-- High and Low Excepts for expf16 functions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_FLOAT16_CONSTANTS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_FLOAT16_CONSTANTS_H

#include "src/__support/FPUtil/except_value_utils.h"

namespace LIBC_NAMESPACE_DECL {

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
static constexpr fputil::ExceptValues<float16, 2> EXPF16_EXCEPTS_LO = {{
    // (input, RZ output, RU offset, RD offset, RN offset)
    // x = 0x1.de4p-8, expf16(x) = 0x1.01cp+0 (RZ)
    {0x1f79U, 0x3c07U, 1U, 0U, 0U},
    // x = 0x1.73cp-6, expf16(x) = 0x1.05cp+0 (RZ)
    {0x25cfU, 0x3c17U, 1U, 0U, 0U},
}};

static constexpr fputil::ExceptValues<float16, 3> EXPF16_EXCEPTS_HI = {{
    // (input, RZ output, RU offset, RD offset, RN offset)
    // x = 0x1.c34p+0, expf16(x) = 0x1.74cp+2 (RZ)
    {0x3f0dU, 0x45d3U, 1U, 0U, 1U},
    // x = -0x1.488p-5, expf16(x) = 0x1.ebcp-1 (RZ)
    {0xa922U, 0x3bafU, 1U, 0U, 0U},
    // x = -0x1.55p-5, expf16(x) = 0x1.ebp-1 (RZ)
    {0xa954U, 0x3bacU, 1U, 0U, 0U},
}};
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_FLOAT16_CONSTANTS_H
