/* Definition of object in frame unwind info.  ia64 version.
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

/* This must match what's in frame.h in gcc. */

struct object
{
  void *pc_base;        /* This field will be set by find_fde. */
  void *pc_begin;
  void *pc_end;
  void *fde_begin;
  void *fde_end;
  void *fde_array;
  __SIZE_TYPE__ count;
  struct object *next;
};
