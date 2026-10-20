//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of type __nftw_func_t.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES___NFTW_FUNC_T_H
#define LLVM_LIBC_TYPES___NFTW_FUNC_T_H

#include "struct_FTW.h"
#include "struct_stat.h"

typedef int (*__nftw_func_t)(const char *, const struct stat *, int,
                             struct FTW *);

#endif // LLVM_LIBC_TYPES___NFTW_FUNC_T_H
