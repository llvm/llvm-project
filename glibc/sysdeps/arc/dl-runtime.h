/* Helpers for On-demand PLT fixup for shared objects.  ARC version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* PLT jump into resolver passes PC of PLTn, while _dl_fixup expects the
   address of corresponding .rela.plt entry.

    - @plt0: runtime pc of first plt entry (DT_PLTGOT)
    - @pltn: runtime pc of plt entry being resolved
    - @size: size of .plt.rela entry (unused).  */
static inline uintptr_t
reloc_index (uintptr_t plt0, uintptr_t pltn, size_t size)
{
  unsigned long int idx = pltn - plt0;

  /* PLT trampoline is 16 bytes.  */
  idx /= 16;

  /* Exclude PLT0 and PLT1.  */
  return idx - 2;
}

static inline uintptr_t
reloc_offset (uintptr_t plt0, uintptr_t pltn)
{
  size_t sz = sizeof (ElfW(Rela));
  return reloc_index (plt0, pltn, sz) * sz;
}
