/* Stub version of processor capability information handling macros.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#ifndef _DL_PROCINFO_H
#define _DL_PROCINFO_H	1

/* We cannot provide a general printing function.  */
#define _dl_procinfo(type, word) -1

/* There are no hardware capabilities defined.  */
#define _dl_hwcap_string(idx) ""

/* By default there is no important hardware capability.  */
#define HWCAP_IMPORTANT (0)

/* There're no platforms to filter out.  */
#define _DL_HWCAP_PLATFORM 0

/* We don't have any hardware capabilities.  */
#define _DL_HWCAP_COUNT 0

#define _dl_string_hwcap(str) (-1)

#define _dl_string_platform(str) (-1)

#endif /* dl-procinfo.h */
