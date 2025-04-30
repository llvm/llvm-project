/* Memory related definitions for program modules.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifndef _XMALLOC_H
#define _XMALLOC_H	1

#include <sys/cdefs.h>
#include <stddef.h>

/* Prototypes for a few program-wide used functions.  */
extern void *xmalloc (size_t n)
  __attribute_malloc__ __attribute_alloc_size__ ((1)) __attr_dealloc_free
  __returns_nonnull;
extern void *xcalloc (size_t n, size_t s)
  __attribute_malloc__ __attribute_alloc_size__ ((1, 2)) __attr_dealloc_free
  __returns_nonnull;
extern void *xrealloc (void *o, size_t n)
  __attribute_malloc__ __attribute_alloc_size__ ((2)) __attr_dealloc_free;
extern char *xstrdup (const char *) __attribute_malloc__ __attr_dealloc_free
  __returns_nonnull;

#endif /* xmalloc.h */
