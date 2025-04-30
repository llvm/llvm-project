/* mtrace API for `malloc'.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
                 Written April 2, 1991 by John Gilmore of Cygnus Support.
                 Based on mcheck.c by Mike Haertel.

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

#if !IS_IN (libc)
# include "mtrace-impl.c"
#else
# include <shlib-compat.h>
# include <libc-symbols.h>
#endif

#if IS_IN (libc) && SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_34)
/* Compatibility symbols that were introduced to help break at allocation sites
   for specific memory allocations.  This is unusable with ASLR, although gdb
   may allow predictable allocation addresses.  Even then, gdb has watchpoint
   and conditional breakpoint support which should provide the same
   functionality without having this kludge.  These symbols are preserved in
   case some applications ended up linking against them but they don't actually
   do anything anymore; not that they did much before anyway.  */

void *mallwatch;
compat_symbol (libc, mallwatch, mallwatch, GLIBC_2_0);

void
tr_break (void)
{
}
compat_symbol (libc, tr_break, tr_break, GLIBC_2_0);
#endif


void
mtrace (void)
{
#if !IS_IN (libc)
  do_mtrace ();
#endif
}

void
muntrace (void)
{
#if !IS_IN (libc)
  do_muntrace ();
#endif
}
