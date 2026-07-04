//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __fixunsdfti implementation as
/// builtins::fixunsdfti so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSDFTI_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSDFTI_H

#include "src/__support/macros/properties/types.h"

#ifdef LIBC_TYPES_HAS_INT128

#include "hdr/stdint_proxy.h"
#include "src/__support/builtins/fixint_helper.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Truncating double -> __uint128_t conversion, saturating on overflow.
// Mirrors compiler-rt's __fixunsdfti.
LIBC_INLINE __uint128_t fixunsdfti(double x) { return fixuint<__uint128_t>(x); }

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_INT128

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSDFTI_H
