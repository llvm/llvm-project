//===-- Implementation header for setjmp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SETJMP_SETJMP_IMPL_H
#define LLVM_LIBC_SRC_SETJMP_SETJMP_IMPL_H

// This header has the _impl prefix in its name to avoid conflict with the
// public header setjmp.h which is also included. here.
#include "hdr/types/jmp_buf.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int setjmp(jmp_buf buf);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SETJMP_SETJMP_IMPL_H
