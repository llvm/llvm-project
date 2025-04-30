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

#if ((defined __mips_nan2008 && !defined HAVE_MIPS_NAN2008) \
     || (!defined __mips_nan2008 && defined HAVE_MIPS_NAN2008))
# error "Configuration inconsistency: __mips_nan2008 != HAVE_MIPS_NAN2008, overridden CFLAGS?"
#endif

/* Redefine the cache ID for new ABIs and 2008 NaN support; legacy o32
   keeps using the generic check.  */
#ifdef __mips_nan2008
# if _MIPS_SIM == _ABIO32
#  define _DL_CACHE_DEFAULT_ID	(FLAG_MIPS_LIB32_NAN2008 | FLAG_ELF_LIBC6)
# elif _MIPS_SIM == _ABI64
#  define _DL_CACHE_DEFAULT_ID	(FLAG_MIPS64_LIBN64_NAN2008 | FLAG_ELF_LIBC6)
# elif _MIPS_SIM == _ABIN32
#  define _DL_CACHE_DEFAULT_ID	(FLAG_MIPS64_LIBN32_NAN2008 | FLAG_ELF_LIBC6)
# endif
#else
# if _MIPS_SIM == _ABI64
#  define _DL_CACHE_DEFAULT_ID	(FLAG_MIPS64_LIBN64 | FLAG_ELF_LIBC6)
# elif _MIPS_SIM == _ABIN32
#  define _DL_CACHE_DEFAULT_ID	(FLAG_MIPS64_LIBN32 | FLAG_ELF_LIBC6)
# endif
#endif

#ifdef _DL_CACHE_DEFAULT_ID
# define _dl_cache_check_flags(flags) \
  ((flags) == _DL_CACHE_DEFAULT_ID)
#endif

#define add_system_dir(dir) \
  do								\
    {								\
      size_t len = strlen (dir);				\
      char path[len + 3];					\
      memcpy (path, dir, len + 1);				\
      if (len >= 6						\
	  && (! memcmp (path + len - 6, "/lib64", 6)		\
	      || ! memcmp (path + len - 6, "/lib32", 6)))	\
	{							\
	  len -= 2;						\
	  path[len] = '\0';					\
	}							\
      add_dir (path);						\
      if (len >= 4 && ! memcmp (path + len - 4, "/lib", 4))	\
	{							\
	  memcpy (path + len, "32", 3);				\
	  add_dir (path);					\
	  memcpy (path + len, "64", 3);				\
	  add_dir (path);					\
	}							\
    } while (0)

#include_next <dl-cache.h>
