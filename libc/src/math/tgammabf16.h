//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the bfloat16 tgamma function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_TGAMMABF16_H
#define LLVM_LIBC_SRC_MATH_TGAMMABF16_H

#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

bfloat16 tgammabf16(bfloat16 x);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_TGAMMABF16_H
