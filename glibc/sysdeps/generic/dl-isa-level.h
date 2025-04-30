/* Support for reading ISA level in /etc/ld.so.cache files written by
   Linux ldconfig.  Generic version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

/* Return true if the ISA level in ENTRY is compatible with CPU.  */
static inline bool
dl_cache_hwcap_isa_level_compatible (struct file_entry_new *entry)
{
  return true;
}
