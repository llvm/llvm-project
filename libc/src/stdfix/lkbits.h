//===-- Implementation header for lkbits ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDFIX_LKBITS_H
#define LLVM_LIBC_SRC_STDFIX_LKBITS_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/macros/config.h"
namespace LIBC_NAMESPACE_DECL {

long accum lkbits(int_lk_t x);

}
#endif
