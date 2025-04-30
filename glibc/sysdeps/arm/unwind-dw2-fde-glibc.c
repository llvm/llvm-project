/* Dummy exception handling and frame unwind runtime interface routines.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.

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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* ARM uses setjmp-longjmp exceptions.  However, previous versions of
   GNU libc exported some DWARF-2 exception handling support routines.
   They are not necessary, but older (or broken) configurations of GCC
   will do so.  Even though all references to these are weak, because
   they refer to versioned symbols, they must be provided.  */

#include <stdlib.h>
#include <unwind.h>
#include <unwind-dw2-fde.h>

/* These may be called from startup code, but don't need to do
   anything.  */

void __register_frame_info_bases (void *a1, struct object *a2,
				  void *a3, void *a4)
{
}

void __register_frame_info (void *a1, struct object *a2)
{
}

void __register_frame (void *a1)
{
}

void __register_frame_info_table_bases (void *a1, struct object *a2,
					void *a3, void *a4)
{
}

void __register_frame_info_table (void *a1, struct object *a2)
{
}

void __register_frame_table (void *a1)
{
}

void *__deregister_frame_info (void *a1)
{
  return NULL;
}

void *__deregister_frame_info_bases (void *a1)
{
  return NULL;
}

void __deregister_frame (void *a1)
{
}

/* This should not be called.  */

fde *
_Unwind_Find_FDE (void *a1, struct dwarf_eh_bases *a2)
{
  abort ();
}
