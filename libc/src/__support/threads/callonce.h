//===-- Types related to the callonce function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace __llvm_libc {

struct CallOnceFlag;
using CallOnceCallback = void(void);

int callonce(CallOnceFlag *flag, CallOnceCallback *callback);

} // namespace __llvm_libc
