//===-- Implementation header of dladdr ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_DLFCN_DLADDR_H
#define LLVM_LIBC_SRC_DLFCN_DLADDR_H

namespace LIBC_NAMESPACE {

#include "include/llvm-libc-types/Dl_info.h"

int dladdr(const void *,Dl_info *);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_DLFCN_DLADDR_H
