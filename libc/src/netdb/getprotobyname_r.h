//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for getprotobyname_r.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NETDB_GETPROTOBYNAME_R_H
#define LLVM_LIBC_SRC_NETDB_GETPROTOBYNAME_R_H

#include "hdr/types/size_t.h"
#include "hdr/types/struct_protoent.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#ifndef LIBC_COPT_PUBLIC_PACKAGING
namespace netdb {

// Overrides the protocols database path on the calling thread for testing.
void set_protocols_path(const char *path);

} // namespace netdb
#endif // LIBC_COPT_PUBLIC_PACKAGING

int getprotobyname_r(const char *__restrict name,
                     struct protoent *__restrict result_buf,
                     char *__restrict buf, size_t buflen,
                     struct protoent **__restrict result);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NETDB_GETPROTOBYNAME_R_H
