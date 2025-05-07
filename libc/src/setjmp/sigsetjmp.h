//===-- Implementation header for sigsetjmp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SETJMP_SIGSETJMP_H
#define LLVM_LIBC_SRC_SETJMP_SIGSETJMP_H

#include "hdr/types/jmp_buf.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/compiler.h"

namespace LIBC_NAMESPACE_DECL {

#ifdef LIBC_COMPILER_IS_GCC
[[gnu::nothrow]]
#endif
[[gnu::returns_twice]] int
sigsetjmp(sigjmp_buf buf, int savesigs);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SETJMP_SIGSETJMP_H
