//===-- Implementation header for strsignal ----------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRSIGNAL_H
#define LLVM_LIBC_SRC_STRING_STRSIGNAL_H

namespace __llvm_libc {

char *strsignal(int sig_num);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_STRSIGNAL_H
