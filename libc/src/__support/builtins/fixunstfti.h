//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __fixunstfti implementation as
/// builtins::fixunstfti so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSTFTI_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSTFTI_H

#include "include/llvm-libc-types/float128.h"
#include "src/__support/macros/properties/types.h"

#if defined(LIBC_TYPES_HAS_FLOAT128) && defined(LIBC_TYPES_HAS_INT128)

#include "hdr/stdint_proxy.h"
#include "src/__support/builtins/fixint_helper.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Truncating float128 -> __uint128_t conversion, saturating on overflow.
// Mirrors compiler-rt's __fixunstfti.
LIBC_INLINE __uint128_t fixunstfti(float128 x) {
  return fixuint<__uint128_t>(x);
}

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT128 && LIBC_TYPES_HAS_INT128

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSTFTI_H
