/* Helpers for On-demand PLT fixup for shared objects.  HPAA version.
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
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */

/* Clear PA_GP_RELOC bit in relocation offset.  */
static inline uintptr_t
reloc_offset (uintptr_t plt0, uintptr_t pltn)
{
  return pltn & ~PA_GP_RELOC;
}

static inline uintptr_t
reloc_index (uintptr_t plt0, uintptr_t pltn, size_t size)
{
  return (pltn & ~PA_GP_RELOC )/ size;
}
