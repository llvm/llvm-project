//===----- Baremetal implementation of a quick exit function ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/quick_exit.h"

// This is intended to be provided by the vendor.
[[noreturn]] extern "C" void __llvm_libc_quick_exit(int status);

namespace LIBC_NAMESPACE {

[[noreturn]] void quick_exit(int status) { __llvm_libc_quick_exit(status); }

} // namespace LIBC_NAMESPACE
