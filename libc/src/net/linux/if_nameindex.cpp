//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of if_nameindex.
///
//===----------------------------------------------------------------------===//

#include "src/net/if_nameindex.h"
#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/OSUtil/linux/network_syscall_policy.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/net/linux/if_nameindex_impl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct if_nameindex *, if_nameindex, ()) {
  ErrorOr<struct if_nameindex *> result =
      net::if_nameindex<net::DefaultNetworkSyscallPolicy>();
  if (!result.has_value()) {
    libc_errno = result.error();
    return nullptr;
  }
  return *result;
}

} // namespace LIBC_NAMESPACE_DECL
