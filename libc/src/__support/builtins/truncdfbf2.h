//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __truncdfbf2 implementation as
/// builtins::truncdfbf2 so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_TRUNCDFBF2_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_TRUNCDFBF2_H

#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Truncate double to bfloat16; mirrors compiler-rt's __truncdfbf2.
LIBC_INLINE uint16_t truncdfbf2(double x) {
  return fputil::cast<bfloat16>(x).bits;
}

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_TRUNCDFBF2_H
