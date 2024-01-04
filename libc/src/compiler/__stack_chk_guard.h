//===-- Internal header for __stack_chk_guard -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_COMPILER___STACK_CHK_GUARD_H
#define LLVM_LIBC_SRC_COMPILER___STACK_CHK_GUARD_H

#include <stdint.h>

// The compiler will emit calls implicitly to a non-namespaced version.
// TODO: can we additionally provide a namespaced alias so that tests can
// explicitly call the namespaced variant rather than the non-namespaced
// definition?
extern "C" uintptr_t __stack_chk_guard;

#endif // LLVM_LIBC_SRC_COMPILER___STACK_CHK_GUARD_H
