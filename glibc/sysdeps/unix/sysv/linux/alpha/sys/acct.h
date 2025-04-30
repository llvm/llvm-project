/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef _SYS_ACCT_H

#define _SYS_ACCT_H	1
#include <features.h>

#include <bits/types/time_t.h>

__BEGIN_DECLS

#define ACCT_COMM 16

struct acct
  {
    char ac_comm[ACCT_COMM];		/* Accounting command name.  */
    time_t ac_utime;			/* Accounting user time.  */
    time_t ac_stime;			/* Accounting system time.  */
    time_t ac_etime;			/* Accounting elapsed time.  */
    time_t ac_btime;			/* Beginning time.  */
    unsigned int ac_uid;		/* Accounting user ID.  */
    unsigned int ac_gid;		/* Accounting group ID.  */
    unsigned int ac_tty;		/* Controlling tty.  */
    /* Please note that the value of the `ac_tty' field, a device number,
       is encoded differently in the kernel and for the libc dev_t type.  */
    char ac_flag;			/* Accounting flag.  */
    long int ac_minflt;			/* Accounting minor pagefaults.  */
    long int ac_majflt;			/* Accounting major pagefaults.  */
    long int ac_exitcode;		/* Accounting process exitcode.  */
  };

enum
  {
    AFORK = 0001,		/* Has executed fork, but no exec.  */
    ASU = 0002,			/* Used super-user privileges.  */
    ACORE = 0004,		/* Dumped core.  */
    AXSIG = 0010		/* Killed by a signal.  */
  };

#define AHZ     100


/* Switch process accounting on and off.  */
extern int acct (const char *__filename) __THROW;

__END_DECLS

#endif	/* sys/acct.h */
