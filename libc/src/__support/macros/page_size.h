//===-- Convenient page size macros -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_PAGE_SIZE_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_PAGE_SIZE_H

#define LIBC_PAGE_SIZE_SYSTEM 0
#define LIBC_PAGE_SIZE_4K (4 * 1024)
#define LIBC_PAGE_SIZE_16K (16 * 1024)
#define LIBC_PAGE_SIZE_64K (64 * 1024)

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_PAGE_SIZE_H
