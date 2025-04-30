/* Early initialization of libc.so.
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

#ifndef _LIBC_EARLY_INIT_H
#define _LIBC_EARLY_INIT_H

struct link_map;

/* If LIBC_MAP is not NULL, look up the __libc_early_init symbol in it
   and call this function, with INITIAL as the argument.  */
void _dl_call_libc_early_init (struct link_map *libc_map, _Bool initial)
  attribute_hidden;

/* In the shared case, this function is defined in libc.so and invoked
   from ld.so (or on the fist static dlopen) after complete relocation
   of a new loaded libc.so, but before user-defined ELF constructors
   run.  In the static case, this function is called directly from the
   startup code.  If INITIAL is true, the libc being initialized is
   the libc for the main program.  INITIAL is false for libcs loaded
   for audit modules, dlmopen, and static dlopen.  */
void __libc_early_init (_Bool initial);

#endif /* _LIBC_EARLY_INIT_H */
