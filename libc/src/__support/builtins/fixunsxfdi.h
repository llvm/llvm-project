//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __fixunsxfdi implementation as
/// builtins::fixunsxfdi so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSXFDI_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSXFDI_H

#include "src/__support/macros/properties/types.h"

#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

#include "hdr/stdint_proxy.h"
#include "src/__support/builtins/fixint_helper.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Truncating long double -> uint64_t conversion, saturating on overflow.
// Mirrors compiler-rt's __fixunsxfdi.
LIBC_INLINE uint64_t fixunsxfdi(long double x) {
  return fixuint<uint64_t>(x);
}

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_FIXUNSXFDI_H
