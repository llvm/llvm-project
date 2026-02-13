//===-- Implementation header of inet_addr ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ARPA_INET_INET_ADDR_H
#define LLVM_LIBC_SRC_ARPA_INET_INET_ADDR_H

#include "include/llvm-libc-types/in_addr_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

in_addr_t inet_addr(const char *cp);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_ARPA_INET_INET_ADDR_H
