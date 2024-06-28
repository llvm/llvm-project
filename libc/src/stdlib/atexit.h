//===-- Implementation header for atexit ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ATEXIT_H
#define LLVM_LIBC_SRC_STDLIB_ATEXIT_H

#include "hdr/types/atexithandler_t.h"
#include "src/stdlib/exit_handler.h"

namespace LIBC_NAMESPACE {

int __cxa_atexit(AtExitCallback *callback, void *payload, void *);

void __cxa_finalize(void *dso);

int atexit(__atexithandler_t);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_ATEXIT_H
