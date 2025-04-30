/* Invoke the early initialization function in libc.so.
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

#include <assert.h>
#include <ldsodefs.h>
#include <libc-early-init.h>
#include <link.h>
#include <stddef.h>

void
_dl_call_libc_early_init (struct link_map *libc_map, _Bool initial)
{
  /* There is nothing to do if we did not actually load libc.so.  */
  if (libc_map == NULL)
    return;

  const ElfW(Sym) *sym
    = _dl_lookup_direct (libc_map, "__libc_early_init",
                         0x069682ac, /* dl_new_hash output.  */
                         "GLIBC_PRIVATE",
                         0x0963cf85); /* _dl_elf_hash output.  */
  assert (sym != NULL);
  __typeof (__libc_early_init) *early_init
    = DL_SYMBOL_ADDRESS (libc_map, sym);
  early_init (initial);
}
