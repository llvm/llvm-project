//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of getpriority
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_RESOURCE_GETPRIORITY_H
#define LLVM_LIBC_SRC_SYS_RESOURCE_GETPRIORITY_H

#include "hdr/types/id_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int getpriority(int which, id_t who);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_RESOURCE_GETPRIORITY_H
