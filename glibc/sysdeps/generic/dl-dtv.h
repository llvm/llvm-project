/* Generic declarations for DTV-based TLS handling in the dynamic linker.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#ifndef _DL_DTV_H
#define _DL_DTV_H

struct dtv_pointer
{
  void *val;                    /* Pointer to data, or TLS_DTV_UNALLOCATED.  */
  void *to_free;                /* Unaligned pointer, for deallocation.  */
};

/* Type for the dtv.  */
typedef union dtv
{
  size_t counter;
  struct dtv_pointer pointer;
} dtv_t;

/* Value used for dtv entries for which the allocation is delayed.  */
#define TLS_DTV_UNALLOCATED ((void *) -1l)

#endif /* _DLT_DTV_H */
