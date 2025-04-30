/* Low-level statistical profiling support function.  Linux/Sparc64 version.
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

#include <signal.h>

#include <sysdeps/unix/sysv/linux/profil-counter.h>

#ifndef __profil_counter
# include <shlib-compat.h>
# if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_31)
void
__profil_counter_global (int signo, struct sigcontext *si)
{
#ifdef __arch64__
  profil_count (si->sigc_regs.tpc);
#else
  profil_count (si->si_regs.pc);
#endif
}
compat_symbol (libc, __profil_counter_global, profil_counter, GLIBC_2_0);
# endif
#endif
