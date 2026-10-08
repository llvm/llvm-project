//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for if_freenameindex.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NET_IF_FREENAMEINDEX_H
#define LLVM_LIBC_SRC_NET_IF_FREENAMEINDEX_H

#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

/// Frees the memory allocated by if_nameindex().
///
/// \param ptr Pointer to the array of if_nameindex structures returned by
///        if_nameindex(). If ptr is nullptr, this function does nothing.
void if_freenameindex(struct if_nameindex *ptr);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NET_IF_FREENAMEINDEX_H
