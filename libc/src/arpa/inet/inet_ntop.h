//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of inet_ntop.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ARPA_INET_INET_NTOP_H
#define LLVM_LIBC_SRC_ARPA_INET_INET_NTOP_H

#include "hdr/types/socklen_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

const char *inet_ntop(int af, const void *__restrict src, char *__restrict dst,
                      socklen_t size);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_ARPA_INET_INET_NTOP_H
