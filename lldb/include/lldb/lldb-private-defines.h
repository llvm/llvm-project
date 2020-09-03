//===-- lldb-private-defines.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_LLDB_PRIVATE_DEFINES_H
#define LLDB_LLDB_PRIVATE_DEFINES_H

#if defined(__cplusplus)

// Include Compiler.h here so we don't define LLVM_FALLTHROUGH and then
// Compiler.h later tries to redefine it.
#include "llvm/Support/Compiler.h"

#ifndef LLVM_FALLTHROUGH

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif

/// \macro LLVM_FALLTHROUGH
/// Marks an empty statement preceding a deliberate switch fallthrough.
#if __has_cpp_attribute(clang::fallthrough)
#define LLVM_FALLTHROUGH [[clang::fallthrough]]
#else
#define LLVM_FALLTHROUGH
#endif

#endif // ifndef LLVM_FALLTHROUGH

#endif // #if defined(__cplusplus)

#endif // LLDB_LLDB_PRIVATE_DEFINES_H
