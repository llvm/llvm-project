//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getaddrinfo.
///
//===----------------------------------------------------------------------===//

#include "src/netdb/getaddrinfo.h"
#include "hdr/errno_macros.h"
#include "hdr/netdb_macros.h"
#include "hdr/types/struct_addrinfo.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getaddrinfo,
                   ([[maybe_unused]] const char *__restrict nodename,
                    [[maybe_unused]] const char *__restrict servname,
                    [[maybe_unused]] const struct addrinfo *__restrict hints,
                    [[maybe_unused]] struct addrinfo **__restrict res)) {
  // TODO: Implement getaddrinfo.
  libc_errno = ENOSYS;
  return EAI_SYSTEM;
}

} // namespace LIBC_NAMESPACE_DECL
