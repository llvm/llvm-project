//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of elf_gregset_t type.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_ELF_GREGSET_T_H
#define LLVM_LIBC_TYPES_ELF_GREGSET_T_H

#include "../llvm-libc-macros/sys-procfs-macros.h"
#include "elf_greg_t.h"
#include "struct_user_regs_struct.h"

typedef elf_greg_t elf_gregset_t[ELF_NGREG];

#endif // LLVM_LIBC_TYPES_ELF_GREGSET_T_H
