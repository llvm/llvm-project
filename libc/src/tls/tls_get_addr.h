//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for __tls_get_addr
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TLS_TLS_GET_ADDR_H
#define LLVM_LIBC_SRC_TLS_TLS_GET_ADDR_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// What the compiler passes: which module the thread local belongs to, and
// where it sits inside that module's block. The layout is fixed by the ABI.
struct TlsIndex {
  unsigned long ti_module;
  unsigned long ti_offset;
};

void *__tls_get_addr(TlsIndex *index);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TLS_TLS_GET_ADDR_H
