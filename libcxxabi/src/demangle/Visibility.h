//===--- Visibility.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file is contains noop definitions of the DEMANGLE_ABI macro defined in
// llvm/include/llvm/Demangle/Visibility.h.
//===----------------------------------------------------------------------===//

#ifndef LIBCXXABI_DEMANGLE_VISIBILITY_H
#define LIBCXXABI_DEMANGLE_VISIBILITY_H

// The DEMANGLE_ABI macro resolves to nothing when building libc++abi. Only
// the llvm copy defines DEMANGLE_ABI as a visibility attribute.
#define DEMANGLE_ABI

#endif
