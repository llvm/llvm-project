//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __ledf2 implementation as builtins::ledf2 so
/// that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_LEDF2_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_LEDF2_H

#include "src/__support/builtins/cmp_helper.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Less-equal comparison of double; mirrors compiler-rt's __ledf2.
LIBC_INLINE int ledf2(double a, double b) { return cmp_le(a, b); }

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_LEDF2_H
