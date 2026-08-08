//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __floattitf implementation as
/// builtins::floattitf so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_FLOATTITF_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_FLOATTITF_H

#include "include/llvm-libc-types/float128.h"
#include "src/__support/macros/properties/types.h"

#if defined(LIBC_TYPES_HAS_FLOAT128) && defined(LIBC_TYPES_HAS_INT128)

#include "hdr/stdint_proxy.h"
#include "src/__support/builtins/floatint_helper.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// float128 <- __int128_t conversion, round to nearest.
// Mirrors compiler-rt's __floattitf.
LIBC_INLINE float128 floattitf(__int128_t x) {
  return floatint<float128>(x);
}

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT128 && LIBC_TYPES_HAS_INT128

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_FLOATTITF_H
