//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of prfpregset_t type.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_PRFPREGSET_T_H
#define LLVM_LIBC_TYPES_PRFPREGSET_T_H

#include "elf_fpregset_t.h"

typedef elf_fpregset_t prfpregset_t;

#endif // LLVM_LIBC_TYPES_PRFPREGSET_T_H
