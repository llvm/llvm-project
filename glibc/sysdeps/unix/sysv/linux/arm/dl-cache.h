/* Support for reading /etc/ld.so.cache files written by Linux ldconfig.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <ldconfig.h>

/* In order to support the transition from unmarked objects
   to marked objects we must treat unmarked objects as
   compatible with either FLAG_ARM_LIBHF or FLAG_ARM_LIBSF.  */
#ifdef __ARM_PCS_VFP
# define _dl_cache_check_flags(flags) \
  ((flags) == (FLAG_ARM_LIBHF | FLAG_ELF_LIBC6) \
   || (flags) == FLAG_ELF_LIBC6)
#else
# define _dl_cache_check_flags(flags) \
  ((flags) == (FLAG_ARM_LIBSF | FLAG_ELF_LIBC6) \
   || (flags) == FLAG_ELF_LIBC6)
#endif

#include_next <dl-cache.h>
