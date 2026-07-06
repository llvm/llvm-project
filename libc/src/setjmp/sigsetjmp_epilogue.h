//===-- Implementation header for sigsetjmp epilogue ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SETJMP_SIGSETJMP_EPILOGUE_H
#define LLVM_LIBC_SRC_SETJMP_SIGSETJMP_EPILOGUE_H

#include "hdr/types/sigjmp_buf.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {
[[gnu::returns_twice]] int sigsetjmp_epilogue(sigjmp_buf buffer, int retval);
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SETJMP_SIGSETJMP_EPILOGUE_H
