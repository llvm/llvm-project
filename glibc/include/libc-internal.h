/* Internal prototype declarations that don't fit anywhere else.
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

#ifndef _LIBC_INTERNAL
# define _LIBC_INTERNAL 1

#include <hp-timing.h>

/* Initialize the `__libc_enable_secure' flag.  */
extern void __libc_init_secure (void);

/* Discover the tick frequency of the machine if something goes wrong,
   we return 0, an impossible hertz.  */
extern int __profile_frequency (void);
libc_hidden_proto (__profile_frequency)

/* Hooks for the instrumenting functions.  */
extern void __cyg_profile_func_enter (void *this_fn, void *call_site);
extern void __cyg_profile_func_exit (void *this_fn, void *call_site);

/* Free all allocated resources.  */
extern void __libc_freeres (void);
libc_hidden_proto (__libc_freeres)

/* Free resources stored in thread-local variables on thread exit.  */
extern void __libc_thread_freeres (void)
#if PTHREAD_IN_LIBC
  attribute_hidden
#endif
  ;

/* Define and initialize `__progname' et. al.  */
extern void __init_misc (int, char **, char **) attribute_hidden;

# if IS_IN (rtld)
extern __typeof (__profile_frequency) __profile_frequency attribute_hidden;
# endif

#ifdef SHARED
/* True if this libc belongs to the initially loaded program (i.e., it
   is not for an audit module, not loaded via dlmopen, and not loaded
   via static dlopen either).  */
extern _Bool __libc_initial attribute_hidden;
#endif

#endif /* _LIBC_INTERNAL  */
