/* Define list of all signal numbers and their names.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <stddef.h>
#include <signal.h>
#include <libintl.h>

const char *const __sys_siglist[NSIG] =
{
#define init_sig(sig, abbrev, desc)   [sig] = desc,
#include <siglist.h>
#undef init_sig
};
libc_hidden_def (__sys_siglist)

const char *const __sys_sigabbrev[NSIG] =
{
#define init_sig(sig, abbrev, desc)   [sig] = abbrev,
#include <siglist.h>
#undef init_sig
};
libc_hidden_def (__sys_sigabbrev)

#include <siglist-compat.c>
