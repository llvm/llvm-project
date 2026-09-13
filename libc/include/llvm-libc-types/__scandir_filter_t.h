//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of __scandir_filter type.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES___SCANDIR_FILTER_T_H
#define LLVM_LIBC_TYPES___SCANDIR_FILTER_T_H

#include "struct_dirent.h"

typedef int (*__scandir_filter_t)(const struct dirent *);

#endif // LLVM_LIBC_TYPES___SCANDIR_FILTER_H
