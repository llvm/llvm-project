//===-- Internal header for __stack_chk_fail --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_COMPILER___STACK_CHK_FAIL_H
#define LLVM_LIBC_SRC_COMPILER___STACK_CHK_FAIL_H

namespace LIBC_NAMESPACE {

[[noreturn]] void __stack_chk_fail();

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_COMPILER___STACK_CHK_FAIL_H
