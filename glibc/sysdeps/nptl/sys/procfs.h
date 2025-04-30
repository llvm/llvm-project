/* Types used by thread_db callback interface.  Stub version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef _SYS_PROCFS_H
#define _SYS_PROCFS_H	1

#include <sys/types.h>

/* The rest of this file provides the types for emulation of the
   Solaris <proc_service.h> interfaces that should be implemented by
   users of libthread_db.  */

/* Addresses.  */
typedef void *psaddr_t;

/* Register set types used in thread_db interface.  */
typedef struct prregset prgregset_t;
typedef struct prfpregset prfpregset_t;

/* Type used for lightweight-process (i.e. thread) IDs.  */
typedef __pid_t lwpid_t;


#endif  /* sys/procfs.h */
