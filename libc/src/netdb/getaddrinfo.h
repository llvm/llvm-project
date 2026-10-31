//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for getaddrinfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NETDB_GETADDRINFO_H
#define LLVM_LIBC_SRC_NETDB_GETADDRINFO_H

#include "hdr/types/struct_addrinfo.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int getaddrinfo(const char *__restrict nodename,
                const char *__restrict servname,
                const struct addrinfo *__restrict hints,
                struct addrinfo **__restrict res);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NETDB_GETADDRINFO_H
