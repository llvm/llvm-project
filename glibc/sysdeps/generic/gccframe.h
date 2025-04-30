/* Definition of object in frame unwind info.  Generic version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>

struct dwarf_fde;
struct fde_vector;

struct object
{
  void *pc_begin;
  void *tbase;
  void *dbase;
  union {
    struct dwarf_fde *single;
    struct dwarf_fde **array;
    struct fde_vector *sort;
  } u;

  union {
    struct {
      unsigned long sorted : 1;
      unsigned long from_array : 1;
      unsigned long mixed_encoding : 1;
      unsigned long encoding : 8;
      /* ??? Wish there was an easy way to detect a 64-bit host here;
	 we've got 32 bits left to play with... */
      unsigned long count : 21;
    } b;
    size_t i;
  } s;

  struct object *next;
};
