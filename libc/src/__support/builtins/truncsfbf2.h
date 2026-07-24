//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __truncsfbf2 implementation as
/// builtins::truncsfbf2 so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_TRUNCSFBF2_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_TRUNCSFBF2_H

#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Truncate float to bfloat16; mirrors compiler-rt's __truncsfbf2.
LIBC_INLINE uint16_t truncsfbf2(float x) {
  return fputil::cast<bfloat16>(x).bits;
}

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_TRUNCSFBF2_H
