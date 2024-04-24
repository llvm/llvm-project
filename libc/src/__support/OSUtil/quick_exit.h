//===---------- Implementation of a quick exit function ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_QUICK_EXIT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_QUICK_EXIT_H

namespace LIBC_NAMESPACE {

[[noreturn]] void quick_exit(int status);

}

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_QUICK_EXIT_H
