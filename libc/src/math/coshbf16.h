//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Public declaration of the coshbf16 entrypoint.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_COSHBF16_H
#define LLVM_LIBC_SRC_MATH_COSHBF16_H

#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

bfloat16 coshbf16(bfloat16 x);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_COSHBF16_H
