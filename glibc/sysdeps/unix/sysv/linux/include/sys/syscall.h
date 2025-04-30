/* Wrapper header for <sys/syscall.h>.  Linux version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.

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
   <http://www.gnu.org/licenses/>.  */

#ifndef _SYSCALL_H

#ifdef _ISOMAC
# include <sysdeps/unix/sysv/linux/sys/syscall.h>
#else /* !_ISOMAC */
/* Use the built-in system call list, not <asm/unistd.h>, which may
   not list all the system call numbers we need.  */
# define _SYSCALL_H
# include <arch-syscall.h>
#endif /* !_ISOMAC */

#endif /* _SYSCALL_H */
