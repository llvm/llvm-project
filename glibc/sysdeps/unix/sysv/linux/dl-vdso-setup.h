/* ELF symbol initialization functions for VDSO objects.  Linux version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _DL_VDSO_INIT_H
#define _DL_VDSO_INIT_H

/* Initialize the VDSO functions pointers.  */
static inline void __attribute__ ((always_inline))
setup_vdso_pointers (void)
{
#ifdef HAVE_CLOCK_GETTIME_VSYSCALL
  GLRO(dl_vdso_clock_gettime) = dl_vdso_vsym (HAVE_CLOCK_GETTIME_VSYSCALL);
#endif
#ifdef HAVE_CLOCK_GETTIME64_VSYSCALL
  GLRO(dl_vdso_clock_gettime64) = dl_vdso_vsym (HAVE_CLOCK_GETTIME64_VSYSCALL);
#endif
#ifdef HAVE_GETTIMEOFDAY_VSYSCALL
  GLRO(dl_vdso_gettimeofday) = dl_vdso_vsym (HAVE_GETTIMEOFDAY_VSYSCALL);
#endif
#ifdef HAVE_TIME_VSYSCALL
  GLRO(dl_vdso_time) = dl_vdso_vsym (HAVE_TIME_VSYSCALL);
#endif
#ifdef HAVE_GETCPU_VSYSCALL
  GLRO(dl_vdso_getcpu) = dl_vdso_vsym (HAVE_GETCPU_VSYSCALL);
#endif
#ifdef HAVE_CLOCK_GETRES_VSYSCALL
  GLRO(dl_vdso_clock_getres) = dl_vdso_vsym (HAVE_CLOCK_GETRES_VSYSCALL);
#endif
#ifdef HAVE_CLOCK_GETRES64_VSYSCALL
  GLRO(dl_vdso_clock_getres_time64) = dl_vdso_vsym (HAVE_CLOCK_GETRES64_VSYSCALL);
#endif
#ifdef HAVE_GET_TBFREQ
  GLRO(dl_vdso_get_tbfreq) = dl_vdso_vsym (HAVE_GET_TBFREQ);
#endif
#ifdef HAVE_SIGTRAMP_RT64
  GLRO(dl_vdso_sigtramp_rt64) = dl_vdso_vsym (HAVE_SIGTRAMP_RT64);
#endif
#ifdef HAVE_SIGTRAMP_RT32
  GLRO(dl_vdso_sigtramp_rt32) = dl_vdso_vsym (HAVE_SIGTRAMP_RT32);
#endif
#ifdef HAVE_SIGTRAMP_32
  GLRO(dl_vdso_sigtramp_32) = dl_vdso_vsym (HAVE_SIGTRAMP_32);
#endif
}

#endif
