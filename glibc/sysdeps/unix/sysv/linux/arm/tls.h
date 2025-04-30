/* Definition for thread-local data handling.  ARM/Linux version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#ifndef _LINUX_ARM_TLS_H
#define _LINUX_ARM_TLS_H	1

/* Almost everything is OS-independent and common for all NPTL on ARM.  */
#include <sysdeps/arm/nptl/tls.h>

#ifndef __ASSEMBLER__

/* Get system call information.  */
# include <sysdep.h>

/* Code to initially initialize the thread pointer.  This might need
   special attention since 'errno' is not yet available and if the
   operation can cause a failure 'errno' must not be touched.  */
# define TLS_INIT_TP(tcbp) \
  ({ long int result_var;						\
     result_var = INTERNAL_SYSCALL_CALL (set_tls, (tcbp));		\
     INTERNAL_SYSCALL_ERROR_P (result_var)				\
       ? "unknown error" : NULL; })

#endif /* __ASSEMBLER__ */

#endif  /* tls.h */
