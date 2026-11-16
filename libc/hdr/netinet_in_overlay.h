//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Including netinet/in.h in overlay mode.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_HDR_NETINET_IN_OVERLAY_H
#define LLVM_LIBC_HDR_NETINET_IN_OVERLAY_H

#ifdef LIBC_FULL_BUILD
#error "This header should only be included in overlay mode"
#endif

#include <netinet/in.h>

// The system <netinet/in.h> may define the byte order conversion functions as
// function-like macros (e.g. when optimizing). These would expand inside our
// own declarations and uses (e.g. `LIBC_NAMESPACE::htons(x)`), so undefine
// them. The include guard of <netinet/in.h> ensures that they don't get
// redefined by a subsequent (possibly transitive) include of the header.
#undef htonl
#undef htons
#undef ntohl
#undef ntohs

#endif // LLVM_LIBC_HDR_NETINET_IN_OVERLAY_H
