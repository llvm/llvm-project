//===----- Baremetal implementation of a quick exit function ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_QUICK_EXIT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_QUICK_EXIT_H

namespace LIBC_NAMESPACE {

// This is intended to be provided by the vendor.
extern "C" void __llvm_libc_quick_exit(int status);

void quick_exit(int status) { __llvm_libc_quick_exit(status); }

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_QUICK_EXIT_H
