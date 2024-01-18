//===-- Implementation header for uname -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_UTSNAME_UNAME_H
#define LLVM_LIBC_SRC_SYS_UTSNAME_UNAME_H

#include <sys/utsname.h>

namespace LIBC_NAMESPACE {

int uname(struct utsname *name);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_UTSNAME_UNAME_H
