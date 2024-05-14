//===-- Implementation header for pathconf_utils ----------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_PREAD_H
#define LLVM_LIBC_SRC_UNISTD_PREAD_H

#include <unistd.h>

namespace LIBC_NAMESPACE {

static long filesizebits(const struct statfs &s);
static long link_max(const struct statfs &s);
static long _2_symlinks(const struct statfs &s);
static long pathconfig(const struct fstatfs &s, int name);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_UNISTD_PREAD_H
