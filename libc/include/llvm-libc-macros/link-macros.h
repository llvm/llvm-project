//===-- Definition of macros to for extra dynamic linker functionality ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_LINK_MACROS_H
#define LLVM_LIBC_MACROS_LINK_MACROS_H

#include "elf-macros.h"

#ifdef __LP64__
#define ElfW(type) Elf64_##type
#else
#define ElfW(type) Elf32_##type
#endif

struct link_map {
  ElfW(Addr) l_addr;
  char *l_name;
  ElfW(Dyn) * l_ld;
  struct link_map *l_next, *l_prev;
};

struct r_debug {
  int r_version;
  struct link_map *r_map;
  ElfW(Addr) r_brk;
  enum { RT_CONSISTENT, RT_ADD, RT_DELETE } r_state;
  ElfW(Addr) r_ldbase;
};

#endif
