//===-- Implementation header for fchown ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_FCHOWN_H
#define LLVM_LIBC_SRC_UNISTD_FCHOWN_H

#include "hdr/types/gid_t.h"
#include "hdr/types/uid_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int fchown(int fildes, uid_t owner, gid_t group);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UNISTD_FCHOWN_H
