//===-- Implementation header for cprojf128 ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/properties/complex_types.h"

#if defined(LIBC_TYPES_HAS_CFLOAT128)

#ifndef LLVM_LIBC_SRC_COMPLEX_CPROJF128_H
#define LLVM_LIBC_SRC_COMPLEX_CPROJF128_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

cfloat128 cprojf128(cfloat128 x);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_COMPLEX_CPROJF128_H

#endif // LIBC_TYPES_HAS_CFLOAT128
