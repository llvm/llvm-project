//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of inet_ntoa.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ARPA_INET_INET_NTOA_H
#define LLVM_LIBC_SRC_ARPA_INET_INET_NTOA_H

#include "hdr/types/struct_in_addr.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *inet_ntoa(struct in_addr in);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_ARPA_INET_INET_NTOA_H
