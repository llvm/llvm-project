//===-- Implementation header for arc4random_uniform -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ARC4RANDOM_UNIFORM_H
#define LLVM_LIBC_SRC_STDLIB_ARC4RANDOM_UNIFORM_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

uint32_t arc4random_uniform(uint32_t upper_bound);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_ARC4RANDOM_UNIFORM_H
