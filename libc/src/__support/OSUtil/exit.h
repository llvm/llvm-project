//===------------ Implementation of an exit function ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_EXIT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_EXIT_H

namespace LIBC_NAMESPACE::internal {

[[noreturn]] void exit(int status);

}

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_EXIT_H
