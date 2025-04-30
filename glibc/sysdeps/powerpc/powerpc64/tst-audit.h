/* Definitions for testing PLT entry/exit auditing.  PowerPC64 version.

   Copyright (C) 2012-2021 Free Software Foundation, Inc.

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

#if _CALL_ELF != 2
#define pltenter la_ppc64_gnu_pltenter
#define pltexit la_ppc64_gnu_pltexit
#define La_regs La_ppc64_regs
#define La_retval La_ppc64_retval
#define int_retval lrv_r3
#else
#define pltenter la_ppc64v2_gnu_pltenter
#define pltexit la_ppc64v2_gnu_pltexit
#define La_regs La_ppc64v2_regs
#define La_retval La_ppc64v2_retval
#define int_retval lrv_r3
#endif
