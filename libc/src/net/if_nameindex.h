//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for if_nameindex.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NET_IF_NAMEINDEX_H
#define LLVM_LIBC_SRC_NET_IF_NAMEINDEX_H

#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

struct if_nameindex *if_nameindex();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NET_IF_NAMEINDEX_H
