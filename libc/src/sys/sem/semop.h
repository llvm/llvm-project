//===-- Implementation header for semop -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SEM_SEMOP_H
#define LLVM_LIBC_SRC_SYS_SEM_SEMOP_H

#include "hdr/types/size_t.h"
#include "hdr/types/struct_sembuf.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int semop(int semid, struct sembuf *sops, size_t nsops);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_SEM_SEMOP_H
