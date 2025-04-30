/* Internal defenitions and declarations for pseudo terminal functions.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Zack Weinberg <zack@rabi.phys.columbia.edu>, 1998.

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

#ifndef _PTY_PRIVATE_H
#define _PTY_PRIVATE_H 1

/* The group slave pseudo terminals belong to.  */
#define TTY_GROUP "tty"

/* The file descriptor connected to the master pseudo terminal.  */
#define PTY_FILENO 3

/* Path to the helper program that implements `grantpt' in user space.  */
#define _PATH_PT_CHOWN LIBEXECDIR "/pt_chown"

/* Test whether given TTY is really a Unix98 pseudo terminal.  */
/* #define unix98_pseudo_p(Dev) ... */

/* Exit codes for the helper program.  */
enum  /* failure modes */
{
  FAIL_EBADF = 1,
  FAIL_EINVAL,
  FAIL_EACCES,
  FAIL_EXEC,
  FAIL_ENOMEM
};

#endif /* pty-private.h  */
