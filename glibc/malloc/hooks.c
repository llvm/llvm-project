/* Compatibility code for malloc debugging and state management.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Wolfram Gloger <wg@malloc.de>, 2001.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#ifndef weak_variable
# define weak_variable weak_function
#endif

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_24)
void (*__malloc_initialize_hook) (void);
compat_symbol (libc, __malloc_initialize_hook,
	       __malloc_initialize_hook, GLIBC_2_0);
#endif

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_34)
void weak_variable (*__after_morecore_hook) (void) = NULL;
compat_symbol (libc, __after_morecore_hook, __after_morecore_hook, GLIBC_2_0);
void *(*__morecore)(ptrdiff_t);
compat_symbol (libc, __morecore, __morecore, GLIBC_2_0);

void weak_variable (*__free_hook) (void *, const void *) = NULL;
void *weak_variable (*__malloc_hook) (size_t, const void *) = NULL;
void *weak_variable (*__realloc_hook) (void *, size_t, const void *) = NULL;
void *weak_variable (*__memalign_hook) (size_t, size_t, const void *) = NULL;
compat_symbol (libc, __free_hook, __free_hook, GLIBC_2_0);
compat_symbol (libc, __malloc_hook, __malloc_hook, GLIBC_2_0);
compat_symbol (libc, __realloc_hook, __realloc_hook, GLIBC_2_0);
compat_symbol (libc, __memalign_hook, __memalign_hook, GLIBC_2_0);
#endif

/*
 * Local variables:
 * c-basic-offset: 2
 * End:
 */
