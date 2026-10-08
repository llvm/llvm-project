//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cxxabi.h"
#include "abort_message.h"

namespace __cxxabiv1 {
extern "C" {
[[noreturn]] _LIBCXXABI_FUNC_VIS
void __cxa_pure_virtual(void) {
  __abort_message("Pure virtual function called!");
}

[[noreturn]] _LIBCXXABI_FUNC_VIS
void __cxa_deleted_virtual(void) {
  __abort_message("Deleted virtual function called!");
}
} // extern "C"
} // abi
