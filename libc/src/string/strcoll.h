//===-- Implementation header for strcoll -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRCOLL_H
#define LLVM_LIBC_SRC_STRING_STRCOLL_H

namespace LIBC_NAMESPACE {

int strcoll(const char *left, const char *right);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_STRCOLL_H
