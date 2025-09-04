//===-- Implementation header for arc4random ---------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ARC4RANDOM_H
#define LLVM_LIBC_SRC_STDLIB_ARC4RANDOM_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

uint32_t arc4random();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_ARC4RANDOM_H
