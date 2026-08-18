//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __unordsf2 implementation as
/// builtins::unordsf2 so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_UNORDSF2_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_UNORDSF2_H

#include "src/__support/builtins/cmp_helper.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Unordered comparison of float; mirrors compiler-rt's __unordsf2.
LIBC_INLINE int unordsf2(float a, float b) { return cmp_unord(a, b); }

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_UNORDSF2_H
