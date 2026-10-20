//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Proxy header for __nftw_func_t.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_HDR_TYPES___NFTW_FUNC_T_H
#define LLVM_LIBC_HDR_TYPES___NFTW_FUNC_T_H

#ifdef LIBC_FULL_BUILD

#include "include/llvm-libc-types/__nftw_func_t.h"

#else // Overlay mode

#include <ftw.h>

#endif // LIBC_FULL_BUILD

#endif // LLVM_LIBC_HDR_TYPES___NFTW_FUNC_T_H
