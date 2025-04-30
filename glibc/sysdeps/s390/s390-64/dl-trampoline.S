/* PLT trampolines.  s390x version.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

	.text
/* Create variant of _dl_runtime_resolve/profile for machines before z13.
   No vector registers are saved/restored.  */
#include <dl-trampoline.h>

#if defined HAVE_S390_VX_ASM_SUPPORT
/* Create variant of _dl_runtime_resolve/profile for z13 and newer.
   The vector registers are saved/restored, too.*/
# define _dl_runtime_resolve _dl_runtime_resolve_vx
# define _dl_runtime_profile _dl_runtime_profile_vx
# define RESTORE_VRS
# include <dl-trampoline.h>
#endif
