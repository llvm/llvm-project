//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for getgroups function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_GETGROUPS_H
#define LLVM_LIBC_SRC_UNISTD_GETGROUPS_H

#include "hdr/types/gid_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int getgroups(int size, gid_t *list);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UNISTD_GETGROUPS_H
