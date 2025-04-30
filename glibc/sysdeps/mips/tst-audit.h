/* Definitions for testing PLT entry/exit auditing.  ARM version.

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

#include <sgidefs.h>

#if _MIPS_SIM == _ABIO32
#define pltenter la_mips_o32_gnu_pltenter
#define pltexit la_mips_o32_gnu_pltexit
#define La_regs La_mips_32_regs
#define La_retval La_mips_32_retval
#else
#if _MIPS_SIM == _ABIN32
#define pltenter la_mips_n32_gnu_pltenter
#define pltexit la_mips_n32_gnu_pltexit
#else
#define pltenter la_mips_n64_gnu_pltenter
#define pltexit la_mips_n64_gnu_pltexit
#endif
#define La_regs La_mips_64_regs
#define La_retval La_mips_64_retval
#endif
#define int_retval lrv_v0
