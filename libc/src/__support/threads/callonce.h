//===-- Types related to the callonce function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_CALLONCE_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_CALLONCE_H

namespace __llvm_libc {

struct CallOnceFlag;
using CallOnceCallback = void(void);

int callonce(CallOnceFlag *flag, CallOnceCallback *callback);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_CALLONCE_H
