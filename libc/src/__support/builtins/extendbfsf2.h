//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header exposes LLVM-libc's __extendbfsf2 implementation as
/// builtins::extendbfsf2 so that it can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_EXTENDBFSF2_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_EXTENDBFSF2_H

#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Extend bfloat16 to float; mirrors compiler-rt's __extendbfsf2.
LIBC_INLINE float extendbfsf2(uint16_t bits) {
  bfloat16 x;
  x.bits = bits;
  return fputil::cast<float>(x);
}

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_EXTENDBFSF2_H
