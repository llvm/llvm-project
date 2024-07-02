//===-- Implementation header for getrusage ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_RESOURCE_GETRUSAGE_H
#define LLVM_LIBC_SRC_SYS_RESOURCE_GETRUSAGE_H

#include <sys/resource.h>

namespace LIBC_NAMESPACE {

int getrusage(int who, struct rusage *usage);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_RESOURCE_GETRUSAGE_H
