//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for getifaddrs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_IFADDRS_GETIFADDRS_H
#define LLVM_LIBC_SRC_IFADDRS_GETIFADDRS_H

#include "hdr/types/struct_ifaddrs.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int getifaddrs(struct ifaddrs **ifap);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_IFADDRS_GETIFADDRS_H
