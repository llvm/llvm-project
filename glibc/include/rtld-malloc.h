/* Redirection of malloc inside the dynamic linker.
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

/* The dynamic linker needs to use its own minimal malloc before libc
   has been relocated, and the libc malloc afterwards.  The active
   malloc implementation is reached via the __rtld_* function pointers
   declared below.  They are initialized to the minimal malloc by
   __rtld_malloc_init_stubs, and set to the final implementation by
   __rtld_malloc_init_real.  */

#ifndef _RTLD_MALLOC_H
#define _RTLD_MALLOC_H

#if IS_IN (rtld)

extern __typeof (calloc) *__rtld_calloc attribute_hidden;
extern __typeof (free) *__rtld_free attribute_hidden;
extern __typeof (malloc) *__rtld_malloc attribute_hidden;
extern __typeof (realloc) *__rtld_realloc attribute_hidden;

/* Wrapper functions which call through the function pointers above.
   Note that it is not supported to take the address of those
   functions.  Instead the function pointers must be used
   directly.  */

__extern_inline void *
calloc (size_t a, size_t b)
{
  return __rtld_calloc (a, b);
}

__extern_inline void
free (void *ptr)
{
   __rtld_free (ptr);
}

__extern_inline void *
malloc (size_t size)
{
  return __rtld_malloc (size);
}

__extern_inline void *
realloc (void *ptr, size_t size)
{
  return __rtld_realloc (ptr, size);
}

/* Called after the first self-relocation to activate the minimal malloc
   implementation.  */
void __rtld_malloc_init_stubs (void) attribute_hidden;

/* Return false if the active malloc is the ld.so minimal malloc, true
   if it is the full implementation from libc.so.  */
_Bool __rtld_malloc_is_complete (void) attribute_hidden;

/* Called shortly before the final self-relocation (when RELRO
   variables are still writable) to activate the real malloc
   implementation.  MAIN_MAP is the link map of the executable.  */
struct link_map;
void __rtld_malloc_init_real (struct link_map *main_map) attribute_hidden;

#else /* !IS_IN (rtld) */

/* This allows static/non-rtld builds to get a pointer to the
   functions, in the same way that is required inside rtld.  */
# define __rtld_calloc (&calloc)
# define __rtld_free (&free)
# define __rtld_malloc (&malloc)
# define __rtld_realloc (&realloc)

#endif /* !IS_IN (rtld) */
#endif /* _RTLD_MALLOC_H */
