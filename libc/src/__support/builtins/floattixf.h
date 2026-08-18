//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __floattixf implementation as
/// builtins::floattixf so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_FLOATTIXF_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_FLOATTIXF_H

#include "src/__support/macros/properties/types.h"

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80) &&                          \
    defined(LIBC_TYPES_HAS_INT128)

#include "hdr/stdint_proxy.h"
#include "src/__support/builtins/floatint_helper.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// long double <- __int128_t conversion, round to nearest.
// Mirrors compiler-rt's __floattixf.
LIBC_INLINE long double floattixf(__int128_t x) {
  return floatint<long double>(x);
}

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80 && LIBC_TYPES_HAS_INT128

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_FLOATTIXF_H
