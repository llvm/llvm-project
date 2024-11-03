//===-- Implementation header of htonl --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NETWORK_HTONL_H
#define LLVM_LIBC_SRC_NETWORK_HTONL_H

#include <stdint.h>

namespace __llvm_libc {

uint32_t htonl(uint32_t hostlong);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_NETWORK_HTONL_H
