/* Function to ignore certain symbol matches for machine-specific reasons.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _ELF_MACHINE_SYM_NO_MATCH_H
#define _ELF_MACHINE_SYM_NO_MATCH_H

#include <link.h>
#include <stdbool.h>

/* This can be customized to ignore certain symbols during lookup in
   case there are machine-specific rules to disregard some
   symbols.  */
static inline bool
elf_machine_sym_no_match (const ElfW(Sym) *sym)
{
  return false;
}

#endif /* _ELF_MACHINE_SYM_NO_MATCH_H */
