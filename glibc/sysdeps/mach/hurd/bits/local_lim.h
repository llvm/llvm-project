/* Minimum guaranteed maximum values for system limits.  Hurd version.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

/* GNU has no arbitrary fixed limits on most of these things, so we
   don't define the macros.  Some things are unlimited.  Some are in
   fact limited but the limit is run-time dependent and fetched with
   `sysconf' or `pathconf'.  */

/* This one value is actually constrained by the `struct dirent'
   layout, in which the `d_namlen' member is only 8 bits wide.  */

#define NAME_MAX	255

/* POSIX.1 requires that we define NGROUPS_MAX (though none of the others
   is required).  GNU allows any number of supplementary groups,
   dynamically allocated.  So we pick a number which seems vaguely
   suitable, and `sysconf' will return a number at least as large.  */

#define NGROUPS_MAX	256

/* The number of data keys per process.  */
#define _POSIX_THREAD_KEYS_MAX	128

/* Controlling the iterations of destructors for thread-specific data.  */
#define _POSIX_THREAD_DESTRUCTOR_ITERATIONS	4

/* The number of threads per process.  */
#define _POSIX_THREAD_THREADS_MAX	64

/* Maximum value the semaphore can have.  */
#define SEM_VALUE_MAX   (2147483647)
