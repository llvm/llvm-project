//===-- Implementation header for getopt ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_GETOPT_H
#define LLVM_LIBC_SRC_UNISTD_GETOPT_H

#include <stdio.h>
#include <unistd.h>

namespace LIBC_NAMESPACE {

namespace impl {
void set_getopt_state(char **, int *, int *, unsigned *, int *, FILE *);
}

int getopt(int argc, char *const argv[], const char *optstring);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_UNISTD_GETOPT_H
