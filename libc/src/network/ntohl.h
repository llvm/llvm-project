//===-- Implementation header of ntohl --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NETWORK_NTOHL_H
#define LLVM_LIBC_SRC_NETWORK_NTOHL_H

#include <stdint.h>

namespace LIBC_NAMESPACE {

uint32_t ntohl(uint32_t netlong);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_NETWORK_NTOHL_H
