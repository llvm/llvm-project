//===-- Definition of the _Exit proxy -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_HDR_FUNC_EXIT_H
#define LLVM_LIBC_HDR_FUNC_EXIT_H

#ifdef LIBC_FULL_BUILD
extern "C" [[noreturn]] void _Exit(int) noexcept;

#else // Overlay mode

#include "hdr/stdlib_overlay.h"

#endif

#endif // LLVM_LIBC_HDR_EXIT_H
