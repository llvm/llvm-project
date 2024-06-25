//===-------- Baremetal implementation of an exit function ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/exit.h"

// This is intended to be provided by the vendor.
extern "C" [[noreturn]] void __llvm_libc_exit(int status);

namespace LIBC_NAMESPACE::internal {

[[noreturn]] void exit(int status) { __llvm_libc_exit(status); }

} // namespace LIBC_NAMESPACE::internal
